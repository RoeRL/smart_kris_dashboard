// Step 1: mic selection + audio capture into a thread-safe queue.
// Equivalent of: audio_queue, audio_callback(), pilih_mikrofon() in the Python version.

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

constexpr int SAMPLE_RATE = 16000;
constexpr int CHANNELS = 1;
constexpr int BLOCK_SIZE = 1024;  // frames per block, same as blocksize=1024 in the Python version

constexpr double SILENCE_DURATION = 1.8;     // seconds of quiet that end an utterance
constexpr double PRE_ROLL_DURATION = 0.4;    // seconds of audio kept from before speech starts
constexpr double MIN_SPEECH_DURATION = 0.5;  // utterances shorter than this are ignored
constexpr double CALIBRATION_SECONDS = 3.0;

using AudioBlock = std::vector<float>;

// Thread-safe queue: the audio thread pushes, the main thread pops.
// Equivalent of Python's queue.Queue().
class AudioQueue {
public:
    void push(AudioBlock block) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(block));
        }
        cv_.notify_one();
    }

    // Waits up to timeout_ms for a block. Returns false on timeout.
    bool pop(AudioBlock& out, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [this] { return !queue_.empty(); })) {
            return false;
        }
        out = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<AudioBlock> queue_;
};

static std::atomic<bool> g_running{true};
static void onSigint(int) { g_running = false; }

// Called by miniaudio on its own audio thread for every new block of mic data.
// Keep it short: copy the samples and push them. No heavy work here.
static void dataCallback(ma_device* device, void* /*output*/, const void* input, ma_uint32 frameCount) {
    if (input == nullptr) return;
    auto* queue = static_cast<AudioQueue*>(device->pUserData);
    const float* samples = static_cast<const float*>(input);
    queue->push(AudioBlock(samples, samples + frameCount * CHANNELS));
}

// Lists capture devices and asks the user to pick one. Returns the index, or -1 on failure.
static int pilihMikrofon(ma_context& context, ma_device_info*& captureInfos, ma_uint32& captureCount) {
    ma_device_info* playbackInfos = nullptr;
    ma_uint32 playbackCount = 0;
    if (ma_context_get_devices(&context, &playbackInfos, &playbackCount,
                               &captureInfos, &captureCount) != MA_SUCCESS) {
        std::cerr << "Gagal mengambil daftar perangkat audio.\n";
        return -1;
    }
    if (captureCount == 0) {
        std::cerr << "Tidak ada perangkat input audio ditemukan.\n";
        return -1;
    }

    std::cout << "\n=======================================================\n"
              << "         DAFTAR PERANGKAT AUDIO INPUT\n"
              << "=======================================================\n";
    for (ma_uint32 i = 0; i < captureCount; ++i) {
        std::cout << "  [" << i << "]  " << captureInfos[i].name
                  << (captureInfos[i].isDefault ? "  <- [DEFAULT]" : "") << "\n";
    }
    std::cout << "=======================================================\n\n";

    while (true) {
        std::cout << "Masukkan nomor INDEX mic yang ingin digunakan: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) return -1;  // EOF / Ctrl+D
        try {
            int choice = std::stoi(line);
            if (choice >= 0 && choice < static_cast<int>(captureCount)) {
                std::cout << "\nMenggunakan perangkat: " << captureInfos[choice].name << "\n\n";
                return choice;
            }
            std::cout << "Index tidak valid. Coba lagi.\n";
        } catch (...) {
            std::cout << "Harap masukkan angka.\n";
        }
    }
}

struct Calibration {
    float threshold;           // volume above this counts as speech
    std::vector<float> noise;  // raw ambient audio, kept as the noise profile for noise reduction later
};

// Records ambient noise for a few seconds, then derives the silence threshold from it.
// Equivalent of kalibrasi_noise() in the Python version.
static Calibration kalibrasiNoise(AudioQueue& queue, double seconds) {
    using clock = std::chrono::steady_clock;

    std::cout << "\n[!] JANGAN BICARA DULU! Mengukur noise ruangan selama " << seconds << " detik...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(800));  // let the user finish speaking
    queue.clear();                                                 // drop anything captured meanwhile

    std::vector<float> noise;
    AudioBlock block;
    const auto start = clock::now();
    while (g_running && std::chrono::duration<double>(clock::now() - start).count() < seconds) {
        if (queue.pop(block, 50)) noise.insert(noise.end(), block.begin(), block.end());
    }

    if (noise.empty()) {
        std::cout << "Tidak ada data kalibrasi, menggunakan nilai default.\n";
        return {0.003f, {}};
    }

    // 99th percentile of |samples|. nth_element avoids sorting the whole buffer.
    // Note: numpy interpolates between neighbours, this takes the nearest rank. Negligible difference.
    std::vector<float> magnitudes(noise.size());
    std::transform(noise.begin(), noise.end(), magnitudes.begin(), [](float s) { return std::fabs(s); });
    const size_t k = static_cast<size_t>(0.99 * (magnitudes.size() - 1));
    std::nth_element(magnitudes.begin(), magnitudes.begin() + k, magnitudes.end());
    const float noisePeak = magnitudes[k];
    const float threshold = std::max(noisePeak * 2.0f, 0.001f);

    std::cout << "Kalibrasi selesai! Noise peak: " << noisePeak << " -> Threshold: " << threshold << "\n";
    return {threshold, std::move(noise)};
}

int main() {
    std::signal(SIGINT, onSigint);

    // The context must stay alive as long as the device is in use,
    // because the device ID we pass lives inside the context's device list.
    ma_context context;
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS) {
        std::cerr << "Gagal inisialisasi audio context.\n";
        return 1;
    }

    ma_device_info* captureInfos = nullptr;
    ma_uint32 captureCount = 0;
    int micIndex = pilihMikrofon(context, captureInfos, captureCount);
    if (micIndex < 0) {
        ma_context_uninit(&context);
        return 1;
    }

    AudioQueue audioQueue;

    // 16 kHz, mono, float32: what Whisper expects. miniaudio converts/resamples if the mic differs.
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.pDeviceID = &captureInfos[micIndex].id;
    config.capture.format = ma_format_f32;
    config.capture.channels = CHANNELS;
    config.sampleRate = SAMPLE_RATE;
    config.periodSizeInFrames = BLOCK_SIZE;
    config.dataCallback = dataCallback;
    config.pUserData = &audioQueue;

    ma_device device;
    if (ma_device_init(&context, &config, &device) != MA_SUCCESS) {
        std::cerr << "Gagal membuka mikrofon.\n";
        ma_context_uninit(&context);
        return 1;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Gagal memulai capture.\n";
        ma_device_uninit(&device);
        ma_context_uninit(&context);
        return 1;
    }

    Calibration calib = kalibrasiNoise(audioQueue, CALIBRATION_SECONDS);

    std::cout << "\n=======================================================\n"
              << ">>>  SISTEM SIAP! Silakan mulai berbicara...\n"
              << ">>>  Ucapkan 1 kalimat utuh, lalu diam sejenak\n"
              << ">>>  Tekan Ctrl+C untuk berhenti\n"
              << "=======================================================\n\n";

    using clock = std::chrono::steady_clock;
    const size_t preRollSize = static_cast<size_t>(PRE_ROLL_DURATION * SAMPLE_RATE / BLOCK_SIZE) + 1;

    std::deque<AudioBlock> preRoll;        // rolling window of the most recent quiet blocks
    std::vector<AudioBlock> audioBuffer;   // blocks of the current utterance
    bool isSpeaking = false;
    clock::time_point speechStart, lastVoice;

    AudioBlock block;
    while (g_running) {
        if (!audioQueue.pop(block, 100)) continue;

        const auto now = clock::now();
        float volume = 0.0f;
        for (float s : block) volume = std::max(volume, std::fabs(s));

        if (volume > calib.threshold) {
            // Loud block: speech starts (or continues).
            if (!isSpeaking) {
                isSpeaking = true;
                speechStart = now;
                std::cout << "Mendengar...\r" << std::flush;
                audioBuffer.assign(preRoll.begin(), preRoll.end());  // prepend the pre-roll
            }
            audioBuffer.push_back(block);
            lastVoice = now;

        } else if (isSpeaking) {
            // Quiet block while speaking: keep it, and check whether the silence is long enough to end.
            audioBuffer.push_back(block);
            if (std::chrono::duration<double>(now - lastVoice).count() > SILENCE_DURATION) {
                const double speechDuration = std::chrono::duration<double>(lastVoice - speechStart).count();

                if (speechDuration < MIN_SPEECH_DURATION) {
                    std::cout << "Terlalu singkat! (" << speechDuration << "s). Abaikan.\n\n";
                } else {
                    // Flatten the blocks into one contiguous utterance.
                    std::vector<float> utterance;
                    for (const auto& b : audioBuffer) utterance.insert(utterance.end(), b.begin(), b.end());

                    std::cout << "\nUcapan terdeteksi: " << utterance.size() << " sampel ("
                              << static_cast<double>(utterance.size()) / SAMPLE_RATE << " detik)\n";
                    // TODO step 3: normalize -> whisper.cpp transcription.
                    // TODO step 4: noise reduction using calib.noise.
                }

                audioBuffer.clear();
                isSpeaking = false;
                audioQueue.clear();  // drop audio that piled up while we were busy
            }

        } else {
            // Idle: keep only the last preRollSize blocks.
            preRoll.push_back(block);
            if (preRoll.size() > preRollSize) preRoll.pop_front();
        }
    }

    std::cout << "\n\nSistem dihentikan oleh pengguna.\n";
    ma_device_uninit(&device);
    ma_context_uninit(&context);
    return 0;
}

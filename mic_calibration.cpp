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
#include <vector>

constexpr int SAMPLE_RATE = 16000;
constexpr int CHANNELS = 1;
constexpr int BLOCK_SIZE = 1024;  // frames per block, same as blocksize=1024 in the Python version

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

    std::cout << "Capture berjalan. Bicara untuk melihat level suara. Ctrl+C untuk berhenti.\n";

    // Temporary test loop: shows the peak volume per block, like `volume = np.max(np.abs(data))`.
    // This will be replaced by the calibration + VAD logic in the next step.
    AudioBlock block;
    while (g_running) {
        if (!audioQueue.pop(block, 100)) continue;

        float peak = 0.0f;
        for (float s : block) peak = std::max(peak, std::fabs(s));

        int bars = std::min(40, static_cast<int>(peak * 80));
        std::cout << "\r[" << std::string(bars, '#') << std::string(40 - bars, ' ') << "] "
                  << peak << "      " << std::flush;
    }

    std::cout << "\n\nSistem dihentikan oleh pengguna.\n";
    ma_device_uninit(&device);
    ma_context_uninit(&context);
    return 0;
}

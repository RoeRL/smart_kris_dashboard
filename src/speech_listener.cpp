// SpeechListener implementation: mic selection, capture, noise calibration, speech/silence detection,
// noise reduction, and Whisper transcription (whisper.cpp). The public interface is in speech_listener.h.

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "whisper.h"

#include "speech_listener.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <memory>
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

// --- Step 3 settings ---
constexpr double TRAILING_KEEP_SECONDS = 0.3;  // keep this much silence after the speech, trim the rest
constexpr double MIN_WHISPER_SECONDS = 1.0;    // pad shorter audio with silence up to this length
constexpr float CONFIDENCE_THRESHOLD = 0.45f;  // below this, warn the user

// Same prompt as the Python version. NOTE: it is about IoT control words (nyalakan, relay, ...),
// not the KRIS dashboard. Consider replacing it with KRIS vocabulary later (ruang, suhu, kelembapan, ...).
constexpr const char* INITIAL_PROMPT =
    "Sistem kontrol IoT berbahasa Indonesia. "
    "Perintah yang mungkin diucapkan: nyalakan, matikan, buka, tutup, "
    "hidupkan, nonaktifkan, kirim data, terima data, cek status, "
    "frekuensi radio, sensor suhu, sensor cahaya, motor, relay, "
    "lampu, kipas, pintu, pompa, RF, transmit, receive, channel.";

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

// Called by miniaudio on its own audio thread for every new block of mic data.
// Keep it short: copy the samples and push them. No heavy work here.
static void dataCallback(ma_device* device, void* /*output*/, const void* input, ma_uint32 frameCount) {
    if (input == nullptr) return;
    auto* queue = static_cast<AudioQueue*>(device->pUserData);
    const float* samples = static_cast<const float*>(input);
    queue->push(AudioBlock(samples, samples + frameCount * CHANNELS));
}

// Fetches the capture devices. The pointers stay valid as long as `context` is alive.
static bool ambilDaftarMic(ma_context& context, ma_device_info*& captureInfos, ma_uint32& captureCount) {
    ma_device_info* playbackInfos = nullptr;
    ma_uint32 playbackCount = 0;
    if (ma_context_get_devices(&context, &playbackInfos, &playbackCount,
                               &captureInfos, &captureCount) != MA_SUCCESS) {
        std::cerr << "Gagal mengambil daftar perangkat audio.\n";
        return false;
    }
    if (captureCount == 0) {
        std::cerr << "Tidak ada perangkat input audio ditemukan.\n";
        return false;
    }
    return true;
}

// Prints the devices and asks the user to pick one. Returns the index, or -1 on failure.
static int pilihMikrofon(ma_device_info* captureInfos, ma_uint32 captureCount) {
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
static Calibration kalibrasiNoise(AudioQueue& queue, double seconds, const std::atomic<bool>& running) {
    using clock = std::chrono::steady_clock;

    std::cout << "\n[!] JANGAN BICARA DULU! Mengukur noise ruangan selama " << seconds << " detik...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(800));  // let the user finish speaking
    queue.clear();                                                 // drop anything captured meanwhile

    std::vector<float> noise;
    AudioBlock block;
    const auto start = clock::now();
    while (running && std::chrono::duration<double>(clock::now() - start).count() < seconds) {
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

// ============================================================================
// Step 4: noise reduction. Stationary spectral gating, a port of noisereduce's
// SpectralGateStationary (stationary=True), using the ambient noise recorded
// during calibration as the noise profile.
//
// How it works:
//  1. STFT of the calibration noise -> per-frequency threshold = mean + 1.5 * std (in dB).
//  2. STFT of the utterance. Every time/frequency cell below its threshold is "noise".
//  3. Noise cells are attenuated (to 15%), the mask is smoothed, then the audio is rebuilt (ISTFT).
// ============================================================================
namespace nr {

using cd = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;
constexpr int N_FFT = 512;
constexpr int WIN_LENGTH = 400;
constexpr int HOP_LENGTH = 100;
constexpr int N_BINS = N_FFT / 2 + 1;                                // 257
constexpr double PROP_DECREASE = 0.85;                               // same as the Python call
constexpr double N_STD_THRESH = 1.5;                                 // noisereduce default
constexpr double TOP_DB = 80.0;                                      // dB floor below each bin's peak
constexpr int N_GRAD_FREQ = 500 * (N_FFT / 2) / SAMPLE_RATE;         // 500 Hz smoothing -> 8 bins
constexpr int N_GRAD_TIME = 50 * SAMPLE_RATE / (HOP_LENGTH * 1000);  // 50 ms smoothing -> 8 frames

struct Profile {
    bool valid = false;
    std::vector<double> thresh;  // dB threshold per frequency bin
};

struct Spec {
    int frames = 0;
    std::vector<cd> data;  // frame-major: data[t * N_BINS + f]
};

// In-place radix-2 FFT. a.size() must be a power of two.
static void fft(std::vector<cd>& a, bool inverse) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const cd wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                const cd u = a[i + k];
                const cd v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& x : a) x /= static_cast<double>(n);
    }
}

// Periodic Hann window (what scipy.signal.stft uses by default).
static std::vector<double> hannWindow() {
    std::vector<double> w(WIN_LENGTH);
    for (int n = 0; n < WIN_LENGTH; ++n) w[n] = 0.5 - 0.5 * std::cos(2.0 * kPi * n / WIN_LENGTH);
    return w;
}

// STFT with zero boundary padding of WIN_LENGTH/2 on both sides (scipy's default, padded=False).
static Spec stft(const std::vector<float>& x) {
    const int n = static_cast<int>(x.size());
    const int pad = WIN_LENGTH / 2;
    const auto win = hannWindow();

    Spec s;
    s.frames = n / HOP_LENGTH + 1;
    s.data.resize(static_cast<size_t>(s.frames) * N_BINS);

    std::vector<cd> buf(N_FFT);
    for (int t = 0; t < s.frames; ++t) {
        const int start = t * HOP_LENGTH - pad;
        for (int i = 0; i < N_FFT; ++i) {
            double v = 0.0;
            if (i < WIN_LENGTH) {
                const int idx = start + i;
                if (idx >= 0 && idx < n) v = x[idx] * win[i];
            }
            buf[i] = cd(v, 0.0);  // frame is zero-padded from 400 to 512 samples
        }
        fft(buf, false);
        for (int f = 0; f < N_BINS; ++f) s.data[static_cast<size_t>(t) * N_BINS + f] = buf[f];
    }
    return s;
}

// Magnitude in dB, floored at (peak of that frequency bin over time) - 80 dB. Same as _amp_to_db().
static std::vector<double> toDb(const Spec& s) {
    const double eps = 2.220446049250313e-16;
    std::vector<double> db(s.data.size());
    for (size_t i = 0; i < db.size(); ++i) db[i] = 20.0 * std::log10(std::abs(s.data[i]) + eps);
    for (int f = 0; f < N_BINS; ++f) {
        double peak = -1e300;
        for (int t = 0; t < s.frames; ++t) peak = std::max(peak, db[static_cast<size_t>(t) * N_BINS + f]);
        for (int t = 0; t < s.frames; ++t) {
            double& v = db[static_cast<size_t>(t) * N_BINS + f];
            v = std::max(v, peak - TOP_DB);
        }
    }
    return db;
}

// Builds the noise profile from the calibration audio.
static Profile makeProfile(const std::vector<float>& noise) {
    Profile p;
    if (noise.size() < static_cast<size_t>(N_FFT)) return p;  // too short to learn anything

    const Spec s = stft(noise);
    const auto db = toDb(s);
    p.thresh.resize(N_BINS);
    for (int f = 0; f < N_BINS; ++f) {
        double mean = 0.0;
        for (int t = 0; t < s.frames; ++t) mean += db[static_cast<size_t>(t) * N_BINS + f];
        mean /= s.frames;
        double var = 0.0;
        for (int t = 0; t < s.frames; ++t) {
            const double d = db[static_cast<size_t>(t) * N_BINS + f] - mean;
            var += d * d;
        }
        var /= s.frames;  // population std, like np.std
        p.thresh[f] = mean + std::sqrt(var) * N_STD_THRESH;
    }
    p.valid = true;
    return p;
}

// Triangle of length 2n+1 (peak in the middle), normalized to sum 1.
static std::vector<double> normalizedTriangle(int n) {
    std::vector<double> v(2 * n + 1);
    for (int i = 0; i <= n; ++i) v[i] = v[2 * n - i] = (i + 1.0) / (n + 1.0);
    double sum = 0.0;
    for (double x : v) sum += x;
    for (double& x : v) x /= sum;
    return v;
}

// Smooths the mask in place. The 2D filter in noisereduce is an outer product of two triangles,
// so it can be applied as two 1D passes (frequency, then time) with identical results.
// Edges are treated as zeros, like fftconvolve(mode="same").
static void smoothMask(std::vector<double>& mask, int frames) {
    const auto fv = normalizedTriangle(N_GRAD_FREQ);
    const auto tv = normalizedTriangle(N_GRAD_TIME);
    std::vector<double> tmp(mask.size(), 0.0);

    for (int t = 0; t < frames; ++t) {
        for (int f = 0; f < N_BINS; ++f) {
            double s = 0.0;
            for (int k = -N_GRAD_FREQ; k <= N_GRAD_FREQ; ++k) {
                const int ff = f + k;
                if (ff >= 0 && ff < N_BINS) s += fv[k + N_GRAD_FREQ] * mask[static_cast<size_t>(t) * N_BINS + ff];
            }
            tmp[static_cast<size_t>(t) * N_BINS + f] = s;
        }
    }
    for (int t = 0; t < frames; ++t) {
        for (int f = 0; f < N_BINS; ++f) {
            double s = 0.0;
            for (int k = -N_GRAD_TIME; k <= N_GRAD_TIME; ++k) {
                const int tt = t + k;
                if (tt >= 0 && tt < frames) s += tv[k + N_GRAD_TIME] * tmp[static_cast<size_t>(tt) * N_BINS + f];
            }
            mask[static_cast<size_t>(t) * N_BINS + f] = s;
        }
    }
}

// Applies the noise reduction to one utterance. Returns a cleaned copy (same length).
static std::vector<float> reduce(const std::vector<float>& audio, const Profile& profile) {
    if (!profile.valid || audio.empty()) return audio;

    // Pad with zeros up to a multiple of the hop size so the whole signal is covered.
    const size_t n0 = audio.size();
    const size_t n = ((n0 + HOP_LENGTH - 1) / HOP_LENGTH) * HOP_LENGTH;
    std::vector<float> x(audio);
    x.resize(n, 0.0f);

    Spec s = stft(x);
    const auto db = toDb(s);

    // 1.0 where the signal is above the noise threshold, (1 - PROP_DECREASE) where it is noise.
    std::vector<double> mask(db.size());
    for (int t = 0; t < s.frames; ++t) {
        for (int f = 0; f < N_BINS; ++f) {
            const size_t i = static_cast<size_t>(t) * N_BINS + f;
            const double above = (db[i] > profile.thresh[f]) ? 1.0 : 0.0;
            mask[i] = above * PROP_DECREASE + (1.0 - PROP_DECREASE);
        }
    }
    smoothMask(mask, s.frames);
    for (size_t i = 0; i < s.data.size(); ++i) s.data[i] *= mask[i];

    // Inverse STFT: inverse FFT per frame, window, overlap-add, divide by the summed squared windows.
    const int pad = WIN_LENGTH / 2;
    const auto win = hannWindow();
    std::vector<double> acc(n + 2 * pad, 0.0), norm(n + 2 * pad, 0.0);
    std::vector<cd> buf(N_FFT);
    for (int t = 0; t < s.frames; ++t) {
        for (int f = 0; f < N_BINS; ++f) buf[f] = s.data[static_cast<size_t>(t) * N_BINS + f];
        for (int f = N_BINS; f < N_FFT; ++f) buf[f] = std::conj(buf[N_FFT - f]);  // mirror for a real signal
        fft(buf, true);
        for (int i = 0; i < WIN_LENGTH; ++i) {
            const size_t pos = static_cast<size_t>(t) * HOP_LENGTH + i;
            if (pos >= acc.size()) break;
            acc[pos] += buf[i].real() * win[i];
            norm[pos] += win[i] * win[i];
        }
    }

    std::vector<float> out(n0);
    for (size_t i = 0; i < n0; ++i) {
        const size_t pos = i + pad;
        out[i] = (norm[pos] > 1e-10) ? static_cast<float>(acc[pos] / norm[pos]) : 0.0f;
    }
    return out;
}

}  // namespace nr

struct Transcription {
    std::string text;
    float confidence;  // 0..1
};

// Scales the audio so its loudest sample is 0.90. Equivalent of the normalization in preprocess_audio().
static void normalisasiAudio(std::vector<float>& audio) {
    float peak = 0.0f;
    for (float s : audio) peak = std::max(peak, std::fabs(s));
    if (peak > 0.001f) {
        const float gain = 0.90f / peak;
        for (float& s : audio) s *= gain;
    }
}

// Sends one utterance to Whisper and returns the text plus a confidence score.
// Equivalent of transkripsi() in the Python version.
static Transcription transkripsi(whisper_context* ctx, std::vector<float> audio) {
    normalisasiAudio(audio);

    // Whisper wants at least ~1 second of audio, so pad very short clips with silence.
    const size_t minSamples = static_cast<size_t>(MIN_WHISPER_SECONDS * SAMPLE_RATE);
    if (audio.size() < minSamples) audio.resize(minSamples, 0.0f);

    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    p.beam_search.beam_size = 5;                 // beam_size=5
    p.language = "id";                           // language="id"
    p.translate = false;
    p.initial_prompt = INITIAL_PROMPT;           // initial_prompt=...
    p.no_context = true;                         // condition_on_previous_text=False
    p.temperature = 0.0f;                        // temperature=0.0
    p.temperature_inc = 0.0f;                    // no temperature fallback (Python passed a single 0.0)
    p.no_speech_thold = 0.6f;                    // no_speech_threshold=0.6
    p.logprob_thold = -1.0f;                     // log_prob_threshold=-1.0
    p.entropy_thold = 2.4f;                      // closest to compression_ratio_threshold=2.4 (not identical)
    p.n_threads = static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
    p.print_progress = false;
    p.print_realtime = false;
    p.print_special = false;
    p.print_timestamps = false;

    if (whisper_full(ctx, p, audio.data(), static_cast<int>(audio.size())) != 0) {
        std::cerr << "Whisper gagal memproses audio.\n";
        return {"", 0.0f};
    }

    // Collect the text and a confidence score. Python used exp(mean of segment avg_logprob);
    // here each segment's avg logprob is the mean over its normal (non-special) tokens.
    const whisper_token eot = whisper_token_eot(ctx);
    std::string text;
    double sumSegLogprob = 0.0;
    int segCount = 0;

    const int nSegments = whisper_full_n_segments(ctx);
    for (int i = 0; i < nSegments; ++i) {
        std::string seg = whisper_full_get_segment_text(ctx, i);
        const size_t first = seg.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;  // empty segment
        const size_t last = seg.find_last_not_of(" \t\r\n");
        seg = seg.substr(first, last - first + 1);

        if (!text.empty()) text += " ";
        text += seg;

        double sum = 0.0;
        int n = 0;
        const int nTokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < nTokens; ++j) {
            const whisper_token_data td = whisper_full_get_token_data(ctx, i, j);
            if (td.id >= eot) continue;  // skip special tokens (timestamps, end-of-text, ...)
            sum += td.plog;
            ++n;
        }
        if (n > 0) {
            sumSegLogprob += sum / n;
            ++segCount;
        }
    }

    float confidence = 0.0f;
    if (segCount > 0) {
        confidence = static_cast<float>(std::exp(sumSegLogprob / segCount));
        confidence = std::min(1.0f, std::max(0.0f, confidence));
    }
    return {text, confidence};
}

// Prints one result block (used when verbose).
static void tampilkanHasil(const std::string& label, const Transcription& hasil, double detik) {
    std::cout << "-------------------------------------------------------\n";
    if (!label.empty()) std::cout << "  >> " << label << "\n";
    if (!hasil.text.empty()) {
        const int filled = static_cast<int>(hasil.confidence * 20);
        std::cout << "  [HASIL]      : " << hasil.text << "\n"
                  << "  [CONFIDENCE] : [" << std::string(filled, '#') << std::string(20 - filled, '.') << "] "
                  << static_cast<int>(hasil.confidence * 1000) / 10.0 << "%\n";
        if (hasil.confidence < CONFIDENCE_THRESHOLD) {
            std::cout << "  !! Kepercayaan rendah. Coba ulangi lebih jelas.\n";
        }
    } else {
        std::cout << "  [Sistem] : Tidak ada teks terdeteksi. Coba bicara lebih keras/jelas.\n";
    }
    std::cout << "  (waktu Whisper: " << detik << " detik)\n";
}

class SpeechListener::Impl {
public:
    explicit Impl(ListenerConfig cfg) : cfg_(std::move(cfg)) {}
    ~Impl() { shutdown(); }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Opens the mic, loads the Whisper model and calibrates the room noise. Returns false on failure.
    bool init() {
        if (ma_context_init(nullptr, 0, nullptr, &context_) != MA_SUCCESS) {
            std::cerr << "Gagal inisialisasi audio context.\n";
            return false;
        }
        contextReady_ = true;

        ma_device_info* infos = nullptr;
        ma_uint32 count = 0;
        if (!ambilDaftarMic(context_, infos, count)) return false;

        int mic = cfg_.micIndex;
        if (mic < 0) {
            mic = pilihMikrofon(infos, count);
        } else if (mic >= static_cast<int>(count)) {
            std::cerr << "Index mic tidak valid: " << mic << "\n";
            return false;
        } else if (cfg_.verbose) {
            std::cout << "\nMenggunakan perangkat: " << infos[mic].name << "\n\n";
        }
        if (mic < 0) return false;

        // Load the Whisper model once (takes a few seconds).
        if (cfg_.verbose) std::cout << "Memuat model Whisper dari '" << cfg_.modelPath << "'...\n";
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = false;  // CPU only, like DEVICE="cpu" in the Python version
        whisper_.reset(whisper_init_from_file_with_params(cfg_.modelPath.c_str(), cparams));
        if (!whisper_) {
            std::cerr << "Gagal memuat model. Cek path: " << cfg_.modelPath << "\n";
            return false;
        }
        if (cfg_.verbose) std::cout << "Model siap!\n\n";

        // 16 kHz, mono, float32: what Whisper expects. miniaudio converts/resamples if the mic differs.
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.capture.pDeviceID = &infos[mic].id;
        config.capture.format = ma_format_f32;
        config.capture.channels = CHANNELS;
        config.sampleRate = SAMPLE_RATE;
        config.periodSizeInFrames = BLOCK_SIZE;
        config.dataCallback = dataCallback;
        config.pUserData = &queue_;
        if (ma_device_init(&context_, &config, &device_) != MA_SUCCESS) {
            std::cerr << "Gagal membuka mikrofon.\n";
            return false;
        }
        deviceReady_ = true;
        if (ma_device_start(&device_) != MA_SUCCESS) {
            std::cerr << "Gagal memulai capture.\n";
            return false;
        }

        calib_ = kalibrasiNoise(queue_, CALIBRATION_SECONDS, running_);
        noiseProfile_ = nr::makeProfile(calib_.noise);
        if (!noiseProfile_.valid && cfg_.nrMode != NrMode::Off) {
            std::cout << "Data noise kalibrasi tidak cukup, noise reduction dimatikan.\n";
            cfg_.nrMode = NrMode::Off;
        }
        if (cfg_.verbose) {
            std::cout << "Noise reduction: "
                      << (cfg_.nrMode == NrMode::On    ? "AKTIF"
                          : cfg_.nrMode == NrMode::Off ? "MATI (--no-nr)"
                                                       : "MODE BANDING (--compare)")
                      << "\n\n=======================================================\n"
                      << ">>>  SISTEM SIAP! Silakan mulai berbicara...\n"
                      << ">>>  Ucapkan 1 kalimat utuh, lalu diam sejenak\n"
                      << ">>>  Tekan Ctrl+C untuk berhenti\n"
                      << "=======================================================\n\n";
        }
        return true;
    }

    // Blocks until one sentence has been recognized (text not empty) and stores it in `out`.
    // Returns false when stop() was called.
    bool listen(Recognition& out) {
        using clock = std::chrono::steady_clock;
        const size_t preRollSize = static_cast<size_t>(PRE_ROLL_DURATION * SAMPLE_RATE / BLOCK_SIZE) + 1;

        // Ignore anything captured while nobody was listening (e.g. while the caller was busy).
        queue_.clear();
        preRoll_.clear();

        std::vector<AudioBlock> audioBuffer;  // blocks of the current utterance
        bool isSpeaking = false;
        size_t quietBlocks = 0;  // quiet blocks since the last loud one (the trailing silence)
        clock::time_point speechStart, lastVoice;

        AudioBlock block;
        while (running_) {
            if (!queue_.pop(block, 100)) continue;

            const auto now = clock::now();
            float volume = 0.0f;
            for (float s : block) volume = std::max(volume, std::fabs(s));

            if (volume > calib_.threshold) {
                // Loud block: speech starts (or continues).
                if (!isSpeaking) {
                    isSpeaking = true;
                    speechStart = now;
                    if (cfg_.verbose) std::cout << "Mendengar...\r" << std::flush;
                    audioBuffer.assign(preRoll_.begin(), preRoll_.end());  // prepend the pre-roll
                }
                audioBuffer.push_back(block);
                lastVoice = now;
                quietBlocks = 0;

            } else if (isSpeaking) {
                // Quiet block while speaking: keep it, and check whether the silence is long enough to end.
                audioBuffer.push_back(block);
                ++quietBlocks;
                if (std::chrono::duration<double>(now - lastVoice).count() > SILENCE_DURATION) {
                    const double speechDuration = std::chrono::duration<double>(lastVoice - speechStart).count();

                    bool got = false;
                    Recognition r;
                    if (speechDuration < MIN_SPEECH_DURATION) {
                        if (cfg_.verbose) std::cout << "Terlalu singkat! (" << speechDuration << "s). Abaikan.\n\n";
                    } else {
                        // Trim most of the trailing silence, keeping a short tail.
                        const size_t keepBlocks =
                            static_cast<size_t>(TRAILING_KEEP_SECONDS * SAMPLE_RATE / BLOCK_SIZE) + 1;
                        if (quietBlocks > keepBlocks) {
                            audioBuffer.resize(audioBuffer.size() - (quietBlocks - keepBlocks));
                        }

                        // Flatten the blocks into one contiguous utterance.
                        std::vector<float> utterance;
                        for (const auto& b : audioBuffer) utterance.insert(utterance.end(), b.begin(), b.end());

                        if (cfg_.verbose) {
                            std::cout << "\nMemproses teks... ("
                                      << static_cast<double>(utterance.size()) / SAMPLE_RATE << " detik audio)\n";
                        }
                        r = transcribeUtterance(std::move(utterance));
                        got = !r.text.empty();
                    }

                    audioBuffer.clear();
                    isSpeaking = false;
                    queue_.clear();  // drop audio that piled up while we were busy
                    if (got) {
                        out = std::move(r);
                        return true;
                    }
                }

            } else {
                // Idle: keep only the last preRollSize blocks.
                preRoll_.push_back(block);
                if (preRoll_.size() > preRollSize) preRoll_.pop_front();
            }
        }
        return false;
    }

    // Makes listen() return false. Safe to call from a signal handler.
    void stop() { running_ = false; }

private:
    void shutdown() {
        if (deviceReady_) {
            ma_device_uninit(&device_);
            deviceReady_ = false;
        }
        if (contextReady_) {
            ma_context_uninit(&context_);
            contextReady_ = false;
        }
    }

    // Noise reduction runs on the raw audio (same scale as the calibration noise);
    // transkripsi() normalizes the volume afterwards.
    Recognition transcribeUtterance(std::vector<float> utterance) {
        auto run = [this](const std::string& label, std::vector<float> audio) {
            const auto t0 = std::chrono::steady_clock::now();
            Transcription r = transkripsi(whisper_.get(), std::move(audio));
            const double detik = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (cfg_.verbose) tampilkanHasil(label, r, detik);
            return r;
        };

        Transcription hasil{"", 0.0f};
        if (cfg_.nrMode == NrMode::Off) {
            hasil = run("", std::move(utterance));
        } else {
            std::vector<float> bersih = nr::reduce(utterance, noiseProfile_);
            if (cfg_.nrMode == NrMode::Compare) run("TANPA noise reduction", std::move(utterance));
            hasil = run(cfg_.nrMode == NrMode::Compare ? "DENGAN noise reduction" : "", std::move(bersih));
        }
        if (cfg_.verbose) std::cout << "-------------------------------------------------------\n\n";

        Recognition out;
        out.text = hasil.text;
        out.confidence = hasil.confidence;
        return out;
    }

    ListenerConfig cfg_;
    std::atomic<bool> running_{true};
    bool contextReady_ = false;
    bool deviceReady_ = false;
    ma_context context_;
    ma_device device_;
    std::unique_ptr<whisper_context, decltype(&whisper_free)> whisper_{nullptr, &whisper_free};
    AudioQueue queue_;
    Calibration calib_{0.003f, {}};
    nr::Profile noiseProfile_;
    std::deque<AudioBlock> preRoll_;  // rolling window of the most recent quiet blocks
};

// ---- public interface (declared in speech_listener.h): thin wrappers around Impl ----

SpeechListener::SpeechListener(ListenerConfig config) : impl_(new Impl(std::move(config))) {}
SpeechListener::~SpeechListener() = default;
bool SpeechListener::init() { return impl_->init(); }
bool SpeechListener::listen(Recognition& out) { return impl_->listen(out); }
void SpeechListener::stop() { impl_->stop(); }

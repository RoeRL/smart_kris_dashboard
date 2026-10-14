#pragma once
// SpeechListener: Indonesian speech recognition as a reusable module.
// Microphone -> noise reduction -> Whisper -> plain std::string.
//
// Usage from any C++ file:
//
//   #include "speech_listener.h"
//
//   SpeechListener listener;
//   if (!listener.init()) return 1;      // opens the mic, loads the model, calibrates noise
//   Recognition r;
//   while (listener.listen(r)) {         // blocks until one sentence is recognized
//       handle(r.text);                  // r.text is a plain std::string
//   }
//
// Keep ONE SpeechListener per program (there is one microphone).

#include <memory>
#include <string>

enum class NrMode {
    On,       // clean the audio, then transcribe (default)
    Off,      // transcribe the raw audio
    Compare,  // transcribe both and print both (the cleaned one is returned)
};

struct ListenerConfig {
    std::string modelPath = "models/ggml-small.bin";  // relative to where the program is run from
    int micIndex = -1;                                // -1 = show the device list and ask
    NrMode nrMode = NrMode::On;
    bool verbose = true;                              // print progress and result details
};

struct Recognition {
    std::string text;         // the recognized sentence
    float confidence = 0.0f;  // 0..1. Low values are NOT filtered, the caller decides.
};

class SpeechListener {
public:
    explicit SpeechListener(ListenerConfig config = {});
    ~SpeechListener();
    SpeechListener(const SpeechListener&) = delete;
    SpeechListener& operator=(const SpeechListener&) = delete;

    // Opens the mic, loads the Whisper model, calibrates room noise (stay quiet for ~3 s).
    // Returns false on failure.
    bool init();

    // Blocks until one sentence is recognized (never returns empty text). Returns false after stop().
    // Audio captured while nobody was calling listen() is discarded.
    bool listen(Recognition& out);

    // Makes listen() return false. Safe to call from a signal handler.
    void stop();

private:
    class Impl;  // all the heavy parts (miniaudio, whisper) live in speech_listener.cpp
    std::unique_ptr<Impl> impl_;
};

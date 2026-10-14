// Minimal demo of the SpeechListener module. This file is NOT part of the module:
// it only shows how any other code can use it.
// Usage: ./build/voice_stt [model.bin] [--mic N] [--no-nr | --compare]

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "speech_listener.h"

static SpeechListener* g_listener = nullptr;
static void onSigint(int) {
    if (g_listener) g_listener->stop();
}

int main(int argc, char** argv) {
    ListenerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-nr") cfg.nrMode = NrMode::Off;
        else if (arg == "--compare") cfg.nrMode = NrMode::Compare;
        else if (arg == "--mic" && i + 1 < argc) cfg.micIndex = std::atoi(argv[++i]);
        else cfg.modelPath = arg;
    }

    SpeechListener listener(cfg);
    g_listener = &listener;
    std::signal(SIGINT, onSigint);

    if (!listener.init()) return 1;

    Recognition r;
    while (listener.listen(r)) {
        // This is where the decision tree gets its input: a plain string.
        std::cout << ">>> [KE DECISION TREE] \"" << r.text << "\"  (confidence " << r.confidence << ")\n\n";
    }

    std::cout << "\nSistem dihentikan oleh pengguna.\n";
    return 0;
}

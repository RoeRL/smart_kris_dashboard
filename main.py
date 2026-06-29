import queue
import sys
import time
import sounddevice as sd
import numpy as np
import noisereduce as nr
from faster_whisper import WhisperModel

# ==============================================================================
# Ni buat konfigurasi modelnya, tar sesuaiin aja
# Jenis mobelnya ini:
#   - "small"   → Ringan (CPU), download ~240 MB, akurasi ajah     
#   - "medium"  → Seimbang (CPU), download ~500 MB, akurasi bagus banget tapi ya rada lemot
#   - "large-v3"→ Gacor (CPU/GPU), download ~1.5 GB, akurasinya paling bagus tapi ya lemot tergantung CPU/GPU
# Untuk mengganti: ubah nilai MODEL_NAME di bawah ini
# ==============================================================================
MODEL_NAME      = "small"        #Ganti ke "medium" atau "large-v3" buat akurasi lebih tinggi
COMPUTE_TYPE    = "int8"      
DEVICE          = "cpu"    

SAMPLE_RATE         = 16000  
CHANNELS            = 1      
SILENCE_DURATION    = 1.8  
PRE_ROLL_DURATION   = 0.4    
MIN_SPEECH_DURATION = 0.5     
CONFIDENCE_THRESHOLD = 0.45   

INITIAL_PROMPT = (
    "Sistem kontrol IoT berbahasa Indonesia. "
    "Perintah yang mungkin diucapkan: nyalakan, matikan, buka, tutup, "
    "hidupkan, nonaktifkan, kirim data, terima data, cek status, "
    "frekuensi radio, sensor suhu, sensor cahaya, motor, relay, "
    "lampu, kipas, pintu, pompa, RF, transmit, receive, channel."
)

audio_queue       = queue.Queue()
noise_profile     = None 
silence_threshold = 0.003 

def audio_callback(indata, frames, time_info, status):
    """Dipanggil oleh sounddevice setiap ada blok audio baru."""
    if status:
        print(f"[Audio Status]: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def pilih_mikrofon():
    """Tampilkan daftar perangkat audio input dan minta pengguna memilih."""
    print("\n" + "="*55)
    print("         DAFTAR PERANGKAT AUDIO INPUT")
    print("="*55)
    devices = sd.query_devices()
    input_devices = []
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((idx, dev['name']))
            is_default = " ← [DEFAULT]" if idx == sd.default.device[0] else ""
            print(f"  [{idx:2d}]  {dev['name']}{is_default}")
    print("="*55 + "\n")

    while True:
        try:
            raw = input("Masukkan nomor INDEX mic yang ingin digunakan: ").strip()
            pilihan = int(raw)
            if any(pilihan == d[0] for d in input_devices):
                print(f"\n✔  Menggunakan perangkat: {devices[pilihan]['name']}\n")
                return pilihan
            print("Index tidak valid. Coba lagi.")
        except ValueError:
            print("Harap masukkan angka.")

def kalibrasi_noise(durasi_detik: float = 3.0) -> tuple[float, np.ndarray]:
    """
    Rekam audio ambient selama beberapa detik untuk:
    1. Mengukur silence threshold secara otomatis
    2. Membuat noise profile untuk noisereduce
    """
    print(f"\n[!] JANGAN BICARA DULU! Mengukur noise ruangan selama {durasi_detik} detik...")
    time.sleep(0.8)

    while not audio_queue.empty():
        audio_queue.get()

    calibration_frames = []
    start = time.time()
    while time.time() - start < durasi_detik:
        if not audio_queue.empty():
            calibration_frames.append(audio_queue.get())
        else:
            time.sleep(0.005)

    if not calibration_frames:
        print("⚠  Tidak ada data kalibrasi, menggunakan nilai default.")
        return 0.003, None

    calib_audio = np.concatenate(calibration_frames, axis=0).ravel().astype(np.float32)
    noise_peak = float(np.percentile(np.abs(calib_audio), 99))
    threshold = max(noise_peak * 2.0, 0.001)

    print(f"✔  Kalibrasi selesai! Noise peak: {noise_peak:.5f} → Threshold: {threshold:.5f}")
    return threshold, calib_audio

def preprocess_audio(audio: np.ndarray, noise_profile: np.ndarray | None) -> np.ndarray:
    """
    Bersihkan audio sebelum dikirim ke Whisper:
    1. Noise reduction menggunakan profil noise yang direkam saat kalibrasi
    2. Normalisasi amplitude
    """
    audio = audio.astype(np.float32)

    if noise_profile is not None:
        try:
            audio = nr.reduce_noise(
                y=audio,
                y_noise=noise_profile,
                sr=SAMPLE_RATE,
                stationary=False,       
                prop_decrease=0.85,     
                n_fft=512,
                win_length=400,
                hop_length=100,
                time_constant_s=2.0,
            )
        except Exception as e:
            pass

    peak = np.max(np.abs(audio))
    if peak > 0.001:
        audio = audio * (0.90 / peak)

    return audio

def transkripsi(model: WhisperModel, audio: np.ndarray) -> tuple[str, float]:
    """
    Kirim audio ke Whisper dan kembalikan (teks, confidence_score).
    """
    segments, info = model.transcribe(
        audio,
        language="id",                   
        beam_size=5,                     
        best_of=5,                         
        temperature=0.0,                   
        vad_filter=True,                  
        vad_parameters={
            "threshold": 0.45,             
            "min_speech_duration_ms": 200,
            "max_speech_duration_s": 30,   
            "min_silence_duration_ms": 500, 
            "speech_pad_ms": 300,           
        },
        initial_prompt=INITIAL_PROMPT,       
        condition_on_previous_text=False,    
        no_speech_threshold=0.6,             
        compression_ratio_threshold=2.4,     
        log_prob_threshold=-1.0,             
        word_timestamps=True,                
    )

    teks_parts   = []
    total_logprob = 0.0
    count         = 0

    for seg in segments:
        t = seg.text.strip()
        if t:
            teks_parts.append(t)
            total_logprob += seg.avg_logprob
            count += 1

    hasil_teks = " ".join(teks_parts).strip()

    if count > 0:
        avg_lp     = total_logprob / count
        confidence = float(np.clip(np.exp(avg_lp), 0.0, 1.0))
    else:
        confidence = 0.0

    return hasil_teks, confidence


def main():
    global noise_profile, silence_threshold
    id_mic = pilih_mikrofon()
    print(f"Memuat model Whisper '{MODEL_NAME}' (offline, {DEVICE}/{COMPUTE_TYPE})...")
    print(f"(Jika pertama kali, model akan di-download otomatis ~{'240 MB' if MODEL_NAME == 'small' else '500 MB' if MODEL_NAME == 'medium' else '1.5 GB'})")
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"Model '{MODEL_NAME}' siap!\n")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=id_mic,
        callback=audio_callback,
        blocksize=1024,
        dtype='float32',
    ):
        silence_threshold, noise_profile = kalibrasi_noise(durasi_detik=3.0)

        print("\n" + "="*55)
        print(">>>  SISTEM SIAP! Silakan mulai berbicara...")
        print(">>>  Ucapkan 1 kalimat utuh, lalu diam sejenak")
        print(">>>  Tekan Ctrl+C untuk berhenti")
        print("="*55 + "\n")
        pre_roll_size    = int(PRE_ROLL_DURATION * SAMPLE_RATE / 1024) + 1
        pre_roll_buffer  = [] 
        audio_buffer     = []   
        is_speaking      = False
        speech_start_time = 0.0
        last_voice_time  = 0.0

        try:
            while True:
                if audio_queue.empty():
                    time.sleep(0.005)
                    continue

                data         = audio_queue.get()
                current_time = time.time()
                volume       = float(np.max(np.abs(data)))

                if volume > silence_threshold:
                    if not is_speaking:
                        is_speaking       = True
                        speech_start_time = current_time
                        print("Mendengar...", end="\r", flush=True)
                        audio_buffer = list(pre_roll_buffer)

                    audio_buffer.append(data)
                    last_voice_time = current_time

                elif is_speaking:
                    audio_buffer.append(data) 
                    if current_time - last_voice_time > SILENCE_DURATION:
                        speech_duration = last_voice_time - speech_start_time
                        if speech_duration < MIN_SPEECH_DURATION:
                            print(f"Terlalu singkat! ({speech_duration:.1f}s). Abaikan.\n")
                        else:
                            print("\nMemproses teks...                        ")
                            audio_data = np.concatenate(audio_buffer, axis=0).ravel().astype(np.float32)
                            audio_clean = preprocess_audio(audio_data, noise_profile)
                            hasil_teks, confidence = transkripsi(model, audio_clean)
                            print("━"*55)
                            if hasil_teks:
                                conf_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
                                conf_pct = confidence * 100
                                print(f"  [HASIL]      : {hasil_teks}")
                                print(f"  [CONFIDENCE] : [{conf_bar}] {conf_pct:.1f}%")
                                if confidence < CONFIDENCE_THRESHOLD:
                                    print(f"  ⚠  Kepercayaan rendah ({conf_pct:.0f}%). Coba ulangi lebih jelas.")
                            else:
                                print("  [Sistem] : Tidak ada teks terdeteksi. Coba bicara lebih keras/jelas.")
                            print("━"*55 + "\n")

                        audio_buffer  = []
                        is_speaking   = False
                        while not audio_queue.empty():
                            audio_queue.get()

                else:
                    pre_roll_buffer.append(data)
                    if len(pre_roll_buffer) > pre_roll_size:
                        pre_roll_buffer.pop(0)

        except KeyboardInterrupt:
            print("\n\n✔  Sistem dihentikan oleh pengguna.")

if __name__ == "__main__":
    main()
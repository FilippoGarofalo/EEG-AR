import time
import threading
import argparse
import os
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample_poly

import librosa
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    audio_path: str = ""

    # Wav2Vec expects 16k
    model_sr: int = 16000

    # analysis window + hop (tradeoff latency vs smoothness)
    window_sec: float = 0.50   # analyze 500 ms at a time
    hop_sec: float = 0.25      # update 4 times per second

    # OSC
    osc_ip: str = "127.0.0.1"
    osc_port: int = 9000        # Python → Processing (analysis output)
    osc_listen_port: int = 9001 # Processing → Python (commands: load/play/stop)
    osc_addr: str = "/brain/lobes"   # sends 4 floats

    # smoothing on the sender side
    smooth_alpha: float = 0.15          # 0..1 ; higher = snappier, lower = smoother
    frontal_smooth_alpha: float = 0.45  # frontal channel is snappier to punch on drum hits

    # STFT params for feature extraction
    n_fft: int = 512
    hop_length: int = 128


# -----------------------------
# Utility: safe normalization
# -----------------------------
class RunningMinMax:
    """Keeps robust-ish min/max online to map values into [0,1]."""
    def __init__(self, decay=0.999):
        self.decay = decay
        self.min = None
        self.max = None

    def update(self, x: float):
        if self.min is None:
            self.min = x
            self.max = x
            return
        # slow drift (prevents exploding normalization)
        self.min = min(self.min * self.decay + x * (1 - self.decay), x)
        self.max = max(self.max * self.decay + x * (1 - self.decay), x)

    def norm01(self, x: float) -> float:
        if self.min is None or self.max is None:
            return 0.0
        denom = (self.max - self.min)
        if denom < 1e-9:
            return 0.0
        y = (x - self.min) / denom
        return float(np.clip(y, 0.0, 1.0))


# -----------------------------
# Core system
# -----------------------------
class BrainOSCPlayer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.osc = SimpleUDPClient(cfg.osc_ip, cfg.osc_port)

        # Load audio file (as float32, always 2D to capture stereo if present)
        audio_raw, sr = sf.read(cfg.audio_path, dtype="float32", always_2d=True)
        self._has_stereo = audio_raw.shape[1] >= 2
        audio = np.mean(audio_raw, axis=1)  # mono for playback and mono analysis
        self.file_sr = sr
        self.audio = audio

        # We keep a 16k mono version for analysis (resample once)
        if sr != cfg.model_sr:
            g = np.gcd(sr, cfg.model_sr)
            up = cfg.model_sr // g
            down = sr // g
            self.audio_16k = resample_poly(audio, up, down).astype(np.float32)
            if self._has_stereo:
                L = resample_poly(audio_raw[:, 0], up, down).astype(np.float32)
                R = resample_poly(audio_raw[:, 1], up, down).astype(np.float32)
                self.audio_stereo_16k = np.stack([L, R], axis=1)  # (N, 2)
            else:
                self.audio_stereo_16k = None
        else:
            self.audio_16k = audio
            self.audio_stereo_16k = audio_raw[:, :2].astype(np.float32) if self._has_stereo else None

        self.duration = len(self.audio) / self.file_sr

        # transport / state
        self._start_time = None
        self._stop_flag = threading.Event()
        self._analysis_thread = None

        # smoothing / normalization
        self._smoothed = np.zeros(4, dtype=np.float32)
        self._norm = [RunningMinMax() for _ in range(4)]

    # -------- transport helpers --------
    def _t_audio(self) -> float:
        if self._start_time is None:
            return 0.0
        return max(0.0, time.time() - self._start_time)

    def _extract_segment_16k(self, t_center: float) -> np.ndarray:
        """Extract window centered at t_center from 16k audio, pad if needed."""
        win = self.cfg.window_sec
        half = win / 2.0

        start_t = t_center - half
        end_t = t_center + half

        start_i = int(round(start_t * self.cfg.model_sr))
        end_i = int(round(end_t * self.cfg.model_sr))

        # pad logic
        pad_left = max(0, -start_i)
        pad_right = max(0, end_i - len(self.audio_16k))

        start_i = max(0, start_i)
        end_i = min(len(self.audio_16k), end_i)

        seg = self.audio_16k[start_i:end_i]
        if pad_left > 0 or pad_right > 0:
            seg = np.pad(seg, (pad_left, pad_right), mode="constant")

        # guard: exact length
        target_len = int(round(win * self.cfg.model_sr))
        if len(seg) != target_len:
            seg = seg[:target_len] if len(seg) > target_len else np.pad(seg, (0, target_len - len(seg)))
        return seg.astype(np.float32)

    def _extract_stereo_segment_16k(self, t_center: float) -> np.ndarray | None:
        """Extract stereo window centered at t_center. Returns (N,2) or None if mono source."""
        if self.audio_stereo_16k is None:
            return None
        win = self.cfg.window_sec
        half = win / 2.0
        start_i = int(round((t_center - half) * self.cfg.model_sr))
        end_i   = int(round((t_center + half) * self.cfg.model_sr))
        pad_left  = max(0, -start_i)
        pad_right = max(0, end_i - len(self.audio_stereo_16k))
        start_i = max(0, start_i)
        end_i   = min(len(self.audio_stereo_16k), end_i)
        seg = self.audio_stereo_16k[start_i:end_i]
        if pad_left > 0 or pad_right > 0:
            seg = np.pad(seg, ((pad_left, pad_right), (0, 0)), mode="constant")
        target_len = int(round(win * self.cfg.model_sr))
        if len(seg) != target_len:
            seg = seg[:target_len] if len(seg) > target_len else np.pad(seg, ((0, target_len - len(seg)), (0, 0)))
        return seg.astype(np.float32)

    # -------- Audio features -> lobe proxies --------
    def _lobes_from_features(self, seg_16k: np.ndarray, seg_stereo: np.ndarray | None = None) -> np.ndarray:
        """
        Maps audio features to 4 brain-lobe proxies using established
        neuroscience correlates.
        """
        sr      = self.cfg.model_sr
        n_fft   = self.cfg.n_fft
        hop     = self.cfg.hop_length
        S       = np.abs(librosa.stft(seg_16k, n_fft=n_fft, hop_length=hop))
        freqs   = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # HPSS computed once and shared by temporal and frontal
        S_harm, S_perc = librosa.decompose.hpss(S)

        # ── Temporal lobe (auditory cortex / superior temporal gyrus) ─────
        # MFCCs capture timbral/phonemic content processed by the auditory cortex.
        # A voicing score boosts the response when singing is present: it measures
        # how much harmonic energy is concentrated in the singing frequency range
        # (180–3500 Hz). Pure percussion or sub-bass keep this score low; a voiced
        # human voice (or melodic instrument in that range) raises it significantly.
        # Bilateral STG processes both the linguistic and melodic stream of singing.
        # Ref: Hickok & Poeppel 2007; Mesgarani et al. 2014; Zatorre et al. 2002
        mfccs      = librosa.feature.mfcc(y=seg_16k, sr=sr, n_mfcc=13,
                                           n_fft=n_fft, hop_length=hop)
        mfcc_score = float(np.mean(np.abs(mfccs[1:])))  # skip coeff-0 (DC energy)
        vocal_mask = (freqs >= 180) & (freqs <= 3500)
        harm_total = float(np.mean(S_harm)) + 1e-9
        harm_vocal = float(np.mean(S_harm[vocal_mask, :])) if vocal_mask.any() else 0.0
        voicing_score = float(np.clip(harm_vocal / harm_total, 0.0, 1.0))
        # Scale: 1.0 (no voice) → up to 2.0 (strong vocal presence)
        temporal   = mfcc_score * (1.0 + voicing_score)

        # ── Parietal lobe (posterior parietal cortex / "where" dorsal stream) ────
        # The posterior parietal cortex is part of the auditory dorsal pathway
        # responsible for sound localisation and spatial processing. Stereo width
        # (Mid/Side ratio) is a direct acoustic correlate: wide/decorrelated audio
        # activates spatial processing, mono audio does not.
        # Loudness (RMS) is retained as a general salience modulator.
        # Ref: Griffiths & Warren 2002; Zatorre et al. 2002
        rms = float(np.mean(librosa.feature.rms(
                        y=seg_16k, frame_length=n_fft, hop_length=hop)))
        if seg_stereo is not None:
            L, R   = seg_stereo[:, 0], seg_stereo[:, 1]
            mid    = (L + R) * 0.5
            side   = (L - R) * 0.5
            rms_mid  = float(np.sqrt(np.mean(mid  ** 2))) + 1e-9
            rms_side = float(np.sqrt(np.mean(side ** 2)))
            stereo_width = float(np.clip(rms_side / rms_mid, 0.0, 1.5))
        else:
            stereo_width = 0.0
        parietal = 0.5 * rms + 0.5 * stereo_width

        # ── Frontal lobe (premotor / prefrontal cortex) ───────────────────
        # S_perc reused from HPSS above (no recomputation).
        # Peak onset (not mean) ensures a single strong drum hit fully activates
        # the lobe. A kick-drum band component (~60-200 Hz) is blended in.
        # Ref: Grahn & Brett 2007; Zatorre et al. 2007
        onset_env     = librosa.onset.onset_strength(S=S_perc, sr=sr, hop_length=hop)
        frontal_onset = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
        kick_mask     = (freqs >= 60) & (freqs <= 200)
        kick_energy   = float(np.mean(S_perc[kick_mask, :])) if kick_mask.any() else 0.0
        frontal       = 0.70 * frontal_onset + 0.30 * kick_energy

        # ── Occipital lobe (cross-modal / auditory imagery) ───────────────
        # Frame-to-frame change in spectral contrast encodes spectral novelty;
        # cross-modal occipital activation is driven by auditory surprise
        # and mental imagery of sound.
        # Ref: Zatorre et al. 2010
        contrast  = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=80.0)
        occipital = float(np.mean(np.abs(np.diff(contrast, axis=1))))

        return np.array([temporal, parietal, frontal, occipital], dtype=np.float32)

    # -------- OSC send --------
    def _send_osc(self, t_analysis: float, values01: np.ndarray):
        # one message with 4 floats + optional time if you want
        self.osc.send_message(self.cfg.osc_addr, values01.tolist())
        # Optional: if your 3D side wants explicit time
        # self.osc.send_message("/brain/time", float(t_analysis))

    # -------- analysis loop --------
    def _analysis_loop(self):
        last_bucket = -1

        while not self._stop_flag.is_set():
            t = self._t_audio()
            if t >= self.duration:
                break

            bucket = int(t / self.cfg.hop_sec)
            if bucket != last_bucket:
                t_analysis = bucket * self.cfg.hop_sec

                seg        = self._extract_segment_16k(t_analysis)
                seg_stereo = self._extract_stereo_segment_16k(t_analysis)
                raw = self._lobes_from_features(seg, seg_stereo)

                # update running normalization
                for i in range(4):
                    self._norm[i].update(float(raw[i]))
                normed = np.array([self._norm[i].norm01(float(raw[i])) for i in range(4)], dtype=np.float32)

                # smoothing (sender-side, helps reduce jitter)
                # frontal channel uses a snappier alpha to punch on drum hits
                a = self.cfg.smooth_alpha
                a_f = self.cfg.frontal_smooth_alpha
                alphas = np.array([a, a, a_f, a], dtype=np.float32)
                self._smoothed = (1 - alphas) * self._smoothed + alphas * normed

                self._send_osc(t_analysis, self._smoothed)

                last_bucket = bucket

            time.sleep(0.01)

        # at end, send zeros to reset visuals
        self._send_osc(self._t_audio(), np.zeros(4, dtype=np.float32))

    # -------- public API --------
    def play(self):
        """Start playback + analysis synced to the same transport."""
        self.stop()  # ensure clean state

        self._stop_flag.clear()
        self._start_time = time.time()

        # Start audio (non-blocking)
        sd.play(self.audio, self.file_sr)

        # Start analysis thread
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

    def stop(self):
        self._stop_flag.set()
        try:
            sd.stop()
        except Exception:
            pass
        self._start_time = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-AR Brain OSC Player")
    parser.add_argument("audio_path", nargs="?", help="Path to the audio file (WAV, optional)")
    args = parser.parse_args()

    cfg = Config()
    current_player: BrainOSCPlayer | None = None
    player_lock = threading.Lock()

    # ── OSC command handlers ──────────────────────────────────
    def handle_load(address: str, path: str):
        global current_player
        path = path.strip().strip("'\"")
        if not os.path.exists(path):
            print(f"  [load] File not found: {path}")
            return
        print(f"  [load] {path}")
        with player_lock:
            if current_player is not None:
                current_player.stop()
            cfg.audio_path = path
            try:
                current_player = BrainOSCPlayer(cfg)
                print(f"  [load] Ready. Send /brain/play to start.")
            except Exception as e:
                print(f"  [load] Error: {e}")
                current_player = None

    def handle_play(address: str, *args):
        global current_player
        with player_lock:
            if current_player is None:
                print("  [play] No file loaded yet — send /brain/load first.")
                return
            print("  [play] Starting playback.")
            current_player.play()

    def handle_stop(address: str, *args):
        global current_player
        with player_lock:
            if current_player is not None:
                print("  [stop] Stopping playback.")
                current_player.stop()

    dispatcher = Dispatcher()
    dispatcher.map("/brain/load", handle_load)
    dispatcher.map("/brain/play", handle_play)
    dispatcher.map("/brain/stop", handle_stop)

    server = ThreadingOSCUDPServer(("0.0.0.0", cfg.osc_listen_port), dispatcher)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print()
    print("=" * 52)
    print("  EEG-AR Brain OSC Player")
    print("=" * 52)
    print(f"  Listening for commands on port : {cfg.osc_listen_port}")
    print(f"  Sending analysis to            : {cfg.osc_ip}:{cfg.osc_port}")
    print(f"  OSC address                    : {cfg.osc_addr}")
    print("-" * 52)
    print("  OSC commands (from Processing):")
    print("    /brain/load <path>  → load a WAV file")
    print("    /brain/play         → start playback")
    print("    /brain/stop         → stop playback")
    print("-" * 52)
    print("  CTRL+C to quit.")
    print("=" * 52)
    print()

    # If a path was passed on the CLI, load it immediately
    if args.audio_path:
        handle_load("/brain/load", args.audio_path)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        with player_lock:
            if current_player is not None:
                current_player.stop()
        server.shutdown()
        print("\n  Stopped. Goodbye!")
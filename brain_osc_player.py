import time
import threading
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample_poly

import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from pythonosc.udp_client import SimpleUDPClient


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    audio_path: str = "/Users/alessandrolillo/Desktop/EEG-AR/AMICI.wav"

    # Wav2Vec expects 16k
    model_sr: int = 16000

    # analysis window + hop (tradeoff latency vs smoothness)
    window_sec: float = 0.50   # analyze 500 ms at a time
    hop_sec: float = 0.25      # update 4 times per second

    # OSC
    osc_ip: str = "127.0.0.1"
    osc_port: int = 9000
    osc_addr: str = "/brain/lobes"   # sends 4 floats

    # smoothing on the sender side
    smooth_alpha: float = 0.15  # 0..1 ; higher = snappier, lower = smoother

    # layer groups (wav2vec2-base has 12 transformer layers + conv features)
    temporal_layers: tuple = (2, 5)   # [2,3,4]
    parietal_layers: tuple = (6, 9)   # [6,7,8]
    frontal_layers: tuple = (10, 12)  # [10,11]
    # occipital is derived (cross-coupling proxy)


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

        # Load audio file (as float32, mono)
        audio, sr = sf.read(cfg.audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)  # mono
        self.file_sr = sr
        self.audio = audio

        # We keep a 16k version for analysis (resample once)
        if sr != cfg.model_sr:
            # resample_poly is high-quality and efficient
            g = np.gcd(sr, cfg.model_sr)
            up = cfg.model_sr // g
            down = sr // g
            self.audio_16k = resample_poly(audio, up, down).astype(np.float32)
        else:
            self.audio_16k = audio

        self.duration = len(self.audio) / self.file_sr

        # Wav2Vec
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h",
            output_hidden_states=True
        )
        self.model.eval()

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

    # -------- Wav2Vec -> lobe proxies --------
    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(t * t) + 1e-12)
    
    def _lobes_from_wav2vec(self, seg_16k):

        inputs = self.processor(seg_16k, sampling_rate=self.cfg.model_sr, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)

        hidden = out.hidden_states
        

        def layer_cat(l0, l1):
            return torch.cat([hidden[k] for k in range(l0, l1)], dim=-1)

        def stats(t):
            mean = torch.mean(torch.abs(t))
            var  = torch.var(t)
            return mean.item(), var.item()

        t_feat = layer_cat(*self.cfg.temporal_layers)
        p_feat = layer_cat(*self.cfg.parietal_layers)
        f_feat = layer_cat(*self.cfg.frontal_layers)

        t_mean, t_var = stats(t_feat)
        p_mean, p_var = stats(p_feat)
        f_mean, f_var = stats(f_feat)

        temporal  = 0.7*t_mean + 0.3*t_var
        parietal  = p_var
        frontal   = f_mean
        occipital = abs(f_mean - t_mean) + abs(f_var - t_var)

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

                seg = self._extract_segment_16k(t_analysis)
                raw = self._lobes_from_wav2vec(seg)

                # update running normalization
                for i in range(4):
                    self._norm[i].update(float(raw[i]))
                normed = np.array([self._norm[i].norm01(float(raw[i])) for i in range(4)], dtype=np.float32)

                # smoothing (sender-side, helps reduce jitter)
                a = self.cfg.smooth_alpha
                self._smoothed = (1 - a) * self._smoothed + a * normed

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
    cfg = Config()
    player = BrainOSCPlayer(cfg)

    print("Press ENTER to PLAY, CTRL+C to stop.")
    input()
    player.play()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        player.stop()
        print("Stopped.")
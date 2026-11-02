# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : f0_extractor.py
# เวอร์ชั่นโค้ด (Version)      : 2.7 (Studio-Quality Pitch & Breath Preservation Edition)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/extract/f0_extractor.py
# หน้าที่ (Function/Role)     : โมดูลแกนหลักสำหรับสกัดความถี่มูลฐาน (F0) รองรับ 6 อัลกอริทึม
#                             ปรับจูนความแม่นยำสูง (High-Fidelity) พร้อมระบบป้องกัน VRAM รั่วไหล 100%
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & Staff
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================
from __future__ import annotations

import os
import sys
import stat
import time
import tempfile
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import parselmouth
import pyworld

SUPPORTED_METHODS = {"pm", "harvest", "dio", "crepe", "mangio-crepe", "rmvpe", "rmvpe_gpu"}


def log_bilingual(exp_dir: Path, th_msg: str, en_msg: str) -> None:
    """ระบบบันทึก Log 2 ภาษา พร้อมเขียนลงไฟล์ (Bilingual Logger)"""
    text = f"[TH] {th_msg}\n[EN] {en_msg}"
    try:
        print(text, flush=True)
    except Exception:
        pass
    try:
        log_path = exp_dir / "extract_f0_feature.log"
        with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(text + "\n")
    except Exception:
        pass


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_rmvpe_model() -> str:
    env_root = os.environ.get("rmvpe_root", "")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / "rmvpe.pt")
    candidates.extend([
        Path.cwd() / "assets" / "rmvpe" / "rmvpe.pt",
        Path.cwd() / "rmvpe.pt",
        Path(__file__).resolve().parents[4] / "assets" / "rmvpe" / "rmvpe.pt",
    ])
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(Path("assets") / "rmvpe" / "rmvpe.pt")


def _save_npy_atomic(output_base: Path, array: np.ndarray) -> None:
    """บันทึกไฟล์ NumPy อย่างปลอดภัย ป้องกันไฟล์พังเมื่อโปรแกรมดับกระทันหัน"""
    output_path = Path(str(output_base) + ".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npy",
            prefix=output_path.stem + ".",
            dir=str(output_path.parent),
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            np.save(temp_file, array, allow_pickle=False)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


class FeatureInput:
    def __init__(
        self,
        samplerate: int = 16000,
        hop_size: int = 160,
        device: Any = "cpu",
        is_half: bool = False,
        exp_dir: Path | None = None,
    ):
        self.fs = int(samplerate)
        self.hop = int(hop_size)
        self.device = device
        self.is_half = bool(is_half)
        self.exp_dir = exp_dir or Path.cwd()
        self.model_rmvpe = None

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log1p(self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log1p(self.f0_max / 700)

    def _load_audio(self, path: str) -> np.ndarray:
        from infer.lib.audio import load_audio
        audio = load_audio(path, self.fs)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("Audio contains no samples")
        if audio.size < 1024:
            audio = np.pad(audio, (0, 1024 - audio.size), mode="constant")
        return audio

    def _align_length(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
        if target_length <= 0:
            return np.zeros(0, dtype=np.float32)
        if f0.size > target_length:
            return f0[:target_length]
        if f0.size < target_length:
            return np.pad(f0, (0, target_length - f0.size), mode="constant", constant_values=0)
        return f0

    def _compute_crepe(self, audio: np.ndarray, expected_length: int, step_factor: int = 1) -> np.ndarray:
        try:
            import torch
            import torchcrepe
            torch.set_num_threads(2)
            dev = str(self.device) if str(self.device).startswith(("cuda", "cpu")) else ("cuda" if torch.cuda.is_available() else "cpu")
            audio_tensor = torch.from_numpy(np.copy(audio))[None].float()
            hop_step = self.hop * max(1, int(step_factor))
            f0, pd = torchcrepe.predict(
                audio_tensor,
                self.fs,
                hop_step,
                self.f0_min,
                self.f0_max,
                "full",
                batch_size=256,
                device=dev,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            
            # [Studio-Quality Tune] ลดเกณฑ์ Periodicity ลงเหลือ 0.05 เพื่อดักจับลมหายใจและลูกคอที่แผ่วเบา
            f0[pd < 0.05] = 0 
            
            pitch = f0[0].detach().cpu().numpy()
            if pitch.size != expected_length and expected_length > 0:
                pitch = np.interp(
                    np.linspace(0, len(pitch), expected_length, endpoint=False),
                    np.arange(len(pitch)),
                    pitch,
                )
            return self._align_length(pitch, expected_length)
        except Exception:
            return self._compute_rmvpe(audio)

    def _compute_rmvpe(self, audio: np.ndarray) -> np.ndarray:
        if self.model_rmvpe is None:
            from infer.lib.rmvpe import RMVPE
            model_p = _find_rmvpe_model()
            log_bilingual(
                self.exp_dir,
                f"🎙️ กำลังโหลดโมเดลสกัดระดับเสียง RMVPE: {model_p}",
                f"Loading RMVPE pitch extraction model: {model_p} on {self.device}"
            )
            self.model_rmvpe = RMVPE(model_p, is_half=self.is_half, device=self.device)
            
        # [Studio-Quality Tune] ปรับ threshold ให้ละเอียดขึ้น (0.03 -> 0.02) เพื่อจับเอกลักษณ์เสียงแผ่ว
        return self.model_rmvpe.infer_from_audio(audio, thred=0.02)

    def compute_f0(self, path: str, method: str) -> np.ndarray:
        method = method.lower().strip()
        if method == "rmvpe_gpu":
            method = "rmvpe"
        if method not in SUPPORTED_METHODS:
            method = "rmvpe"

        audio = self._load_audio(path)
        expected_length = audio.shape[0] // self.hop

        if method == "pm":
            pitch = (
                parselmouth.Sound(audio, self.fs)
                .to_pitch_ac(
                    time_step=self.hop / self.fs,
                    voicing_threshold=0.45, # [Studio-Quality Tune] ไวต่อเสียงลมหายใจมากขึ้น (0.6 -> 0.45)
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pitch = self._align_length(pitch, expected_length)
        elif method in {"harvest", "dio"}:
            audio_double = np.ascontiguousarray(audio, dtype=np.float64)
            frame_period = 1000 * self.hop / self.fs
            if method == "harvest":
                pitch, times = pyworld.harvest(
                    audio_double, fs=self.fs, f0_ceil=self.f0_max,
                    f0_floor=self.f0_min, frame_period=frame_period,
                )
            else:
                pitch, times = pyworld.dio(
                    audio_double, fs=self.fs, f0_ceil=self.f0_max,
                    f0_floor=self.f0_min, frame_period=frame_period,
                )
            pitch = pyworld.stonemask(audio_double, pitch, times, self.fs)
        elif method == "crepe":
            pitch = self._compute_crepe(audio, expected_length, step_factor=1)
        elif method == "mangio-crepe":
            pitch = self._compute_crepe(audio, expected_length, step_factor=2)
        else:
            pitch = self._compute_rmvpe(audio)

        pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
        pitch[~np.isfinite(pitch)] = 0
        pitch[pitch < 0] = 0
        return pitch

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        f0 = np.maximum(f0, 0)

        f0_mel = 1127 * np.log1p(f0 / 700)
        voiced = f0_mel > 0
        f0_mel[voiced] = (
            (f0_mel[voiced] - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1
        )
        f0_mel = np.clip(f0_mel, 1, self.f0_bin - 1)
        coarse = np.rint(f0_mel).astype(np.int64)
        return coarse

    def process_paths(
        self,
        paths: Sequence[tuple[str, str, str]],
        method: str,
        worker_id: int = 0,
    ) -> tuple[int, int, int]:
        if not paths:
            log_bilingual(self.exp_dir, "ไม่มีไฟล์เสียงให้ประมวลผล (No F0 tasks to do)", "No F0 tasks to do")
            return 0, 0, 0

        total = len(paths)
        succ_c, skip_c, fail_c = 0, 0, 0

        for index, (input_path, coarse_base, continuous_base) in enumerate(paths):
            fname = os.path.basename(input_path)
            try:
                coarse_path = Path(coarse_base + ".npy")
                continuous_path = Path(continuous_base + ".npy")

                if coarse_path.is_file() and continuous_path.is_file() and coarse_path.stat().st_size > 64 and continuous_path.stat().st_size > 64:
                    skip_c += 1
                    continue

                continuous_f0 = self.compute_f0(input_path, method)
                coarse_f0 = self.coarse_f0(continuous_f0)

                _save_npy_atomic(Path(continuous_base), continuous_f0)
                _save_npy_atomic(Path(coarse_base), coarse_f0)
                succ_c += 1

                pct = ((index + 1) / total) * 100.0
                log_bilingual(
                    self.exp_dir,
                    f"⏳ [เทรดที่ {worker_id}] ความคืบหน้า: [{index + 1}/{total}] ({pct:.1f}%) | ไฟล์: {fname}",
                    f"[Worker {worker_id}] Progress: [{index + 1}/{total}] ({pct:.1f}%) | File: {fname}"
                )
            except Exception as e:
                fail_c += 1
                log_bilingual(
                    self.exp_dir,
                    f"❌ ล้มเหลวที่ไฟล์ {fname}: {e}",
                    f"Failed at {fname}: {traceback.format_exc()}"
                )
            finally:
                # [Zero-Leak] คืนค่าหน่วยความจำทันทีหลังจบแต่ละไฟล์
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        return succ_c, skip_c, fail_c


def _worker(
    paths: Sequence[tuple[str, str, str]],
    method: str,
    exp_dir: str,
    device: Any,
    is_half: bool,
    worker_id: int = 0,
) -> None:
    worker = FeatureInput(device=device, is_half=is_half, exp_dir=Path(exp_dir))
    worker.process_paths(paths, method, worker_id=worker_id)


def _collect_paths(exp_dir: Path) -> list[tuple[str, str, str]]:
    input_root = exp_dir / "1_16k_wavs"
    coarse_root = exp_dir / "2a_f0"
    continuous_root = exp_dir / "2b-f0nsf"

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    coarse_root.mkdir(parents=True, exist_ok=True)
    continuous_root.mkdir(parents=True, exist_ok=True)

    valid_audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    paths = []
    for input_path in sorted(input_root.iterdir()):
        if not input_path.is_file():
            continue
        base_lower = input_path.name.lower()
        if input_path.suffix.lower() not in valid_audio_exts or base_lower in ("desktop.ini", "thumbs.db"):
            try:
                if not os.path.islink(str(input_path)):
                    os.chmod(str(input_path), stat.S_IWRITE)
                input_path.unlink(missing_ok=True)
                log_bilingual(exp_dir, f"ลบทิ้งไฟล์ขยะ: {input_path.name}", f"[AUTO-PURGE] Deleted non-audio file: {input_path.name}")
            except Exception:
                pass
            continue
        if "spec" in base_lower:
            continue
        paths.append(
            (
                str(input_path),
                str(coarse_root / input_path.name),
                str(continuous_root / input_path.name),
            )
        )
    return paths


def run_extraction(
    exp_dir: str,
    method: str,
    workers: int = 1,
    shard_count: int | None = None,
    shard_index: int = 0,
    device: Any = "cpu",
    is_half: bool = False,
) -> int:
    start_t = time.time()
    start_str = datetime.now().strftime("%H:%M:%S")
    exp_path = Path(exp_dir).expanduser().resolve()
    method_clean = method.lower().strip()
    if method_clean == "rmvpe_gpu":
        method_clean = "rmvpe"
    if method_clean not in SUPPORTED_METHODS:
        method_clean = "rmvpe"

    workers = max(1, min(2, int(workers or 1)))
    if method_clean in ("rmvpe", "crepe", "mangio-crepe") and str(device) == "cpu":
        workers = 1

    log_bilingual(
        exp_path,
        f"🎵 เริ่มกระบวนการสกัดระดับเสียง (F0 Extraction) | อัลกอริทึม: {method_clean.upper()} | เวลา: {start_str}",
        f"Starting F0 Pitch Extraction | Method: {method_clean.upper()} | Device: {device} | Threads: {workers} | Start: {start_str}"
    )

    all_paths = _collect_paths(exp_path)

    if shard_count is not None:
        shard_count = max(1, min(2, int(shard_count)))
        shard_index = max(0, min(shard_count - 1, int(shard_index)))
        paths = all_paths
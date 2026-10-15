# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : f0_extractor.py
# เวอร์ชั่นโค้ด (Version)      : 2.0 (Unified Bilingual & Smart Worker Balanced Edition)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/extract/f0_extractor.py
# หน้าที่ (Function/Role)     : แกนกลางสกัดระดับเสียง (F0 Pitch Extractor) รองรับ PM, Harvest, DIO, RMVPE
#                             พร้อมระบบ Auto-Purge ลบไฟล์ขยะ, Atomic NPY Save, Smart RMVPE Memory Guard
#                             และรายงานผลสองภาษา (TH/EN) แบบ Real-time ประสานกับ Program.py ไร้รอยต่อ
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC Project
# ==============================================================================
from __future__ import annotations

import os
import sys
import stat
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

# Configure UTF-8 encoding for standard streams on Windows
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Ensure project root is always in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import parselmouth
import pyworld

SUPPORTED_METHODS = {"pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"}


def log_message(exp_dir: Path, message: Any) -> None:
    """บันทึกข้อความออกทางหน้าจอ (stdout) พร้อม flush=True และเขียนลงไฟล์ extract_f0_feature.log"""
    text = str(message)
    try:
        print(text, flush=True)
    except Exception:
        try:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(text.encode(enc, errors="replace").decode(enc), flush=True)
        except Exception:
            pass

    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "extract_f0_feature.log"
        with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(text + "\n")
    except Exception:
        pass


def log_bilingual(exp_dir: Path, th_msg: str, en_msg: str, use_sep: bool = False) -> None:
    """แสดงผลข้อความสองภาษา (ไทย-อังกฤษ) ตามมาตรฐานเดียวกับ extract_feature_print.py"""
    if use_sep:
        log_message(exp_dir, "-" * 65)
    log_message(exp_dir, f"{th_msg}\n({en_msg})")
    if use_sep:
        log_message(exp_dir, "-" * 65)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_rmvpe_model() -> str:
    """ค้นหาไฟล์โมเดล RMVPE จากทุกตำแหน่งมาตรฐานในโปรเจกต์ RVC และ DiffSinger"""
    env_rmvpe = os.environ.get("rmvpe_root", "")
    candidates = [
        Path(env_rmvpe) / "rmvpe.pt" if env_rmvpe else None,
        Path.cwd() / "assets" / "rmvpe" / "rmvpe.pt",
        Path.cwd() / "rmvpe.pt",
        PROJECT_ROOT / "assets" / "rmvpe" / "rmvpe.pt",
        PROJECT_ROOT / "rmvpe.pt",
        PROJECT_ROOT / "checkpoints" / "rmvpe" / "model.pt",
        Path(r"B:\DiffSinger-2.5.1\checkpoints\rmvpe\model.pt"),
    ]

    for path in candidates:
        if path is not None and path.is_file():
            return str(path)

    return str(PROJECT_ROOT / "assets" / "rmvpe" / "rmvpe.pt")


def _save_npy_atomic(output_base: Path, array: np.ndarray) -> None:
    """บันทึกไฟล์ NumPy (.npy) แบบ Atomic เพื่อป้องกันไฟล์เสียหายหากถูกตัดจบการทำงาน"""
    out_str = str(output_base)
    output_path = Path(out_str if out_str.lower().endswith(".npy") else out_str + ".npy")
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
        if temp_path is not None and temp_path.exists():
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
        self.f0_mel_min = 1127.0 * np.log1p(self.f0_min / 700.0)
        self.f0_mel_max = 1127.0 * np.log1p(self.f0_max / 700.0)

    def _load_audio(self, path: str) -> np.ndarray:
        from infer.lib.audio import load_audio

        audio = load_audio(path, self.fs)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            raise ValueError("ไฟล์เสียงว่างเปล่า (Audio contains no samples)")

        if audio.size < 1600:
            audio = np.pad(audio, (0, 1600 - audio.size), mode="constant")

        return audio

    def _align_length(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """ปรับความยาวของเส้นโค้ง F0 ให้ตรงกับจำนวนเฟรมเป้าหมายอย่างแม่นยำ"""
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)

        if target_length <= 0:
            return np.zeros(0, dtype=np.float32)

        if f0.size > target_length:
            return f0[:target_length]

        if f0.size < target_length:
            return np.pad(
                f0,
                (0, target_length - f0.size),
                mode="constant",
                constant_values=0.0,
            )

        return f0

    def compute_f0(self, path: str, method: str) -> np.ndarray:
        method = method.lower().strip()
        if method == "rmvpe_gpu":
            method = "rmvpe"

        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"ไม่รองรับอัลกอริทึม F0 {method!r}; กรุณาเลือกจาก {', '.join(sorted(SUPPORTED_METHODS))}"
            )

        audio = self._load_audio(path)
        expected_length = audio.shape[0] // self.hop

        if method == "pm":
            pitch = (
                parselmouth.Sound(audio, self.fs)
                .to_pitch_ac(
                    time_step=self.hop / self.fs,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pitch = self._align_length(pitch, expected_length)

        elif method in {"harvest", "dio"}:
            audio_double = np.ascontiguousarray(audio, dtype=np.float64)
            frame_period = 1000.0 * self.hop / self.fs

            if method == "harvest":
                pitch, times = pyworld.harvest(
                    audio_double,
                    fs=self.fs,
                    f0_ceil=self.f0_max,
                    f0_floor=self.f0_min,
                    frame_period=frame_period,
                )
            else:
                pitch, times = pyworld.dio(
                    audio_double,
                    fs=self.fs,
                    f0_ceil=self.f0_max,
                    f0_floor=self.f0_min,
                    frame_period=frame_period,
                )

            pitch = pyworld.stonemask(audio_double, pitch, times, self.fs)
            pitch = self._align_length(pitch, expected_length)

        else:  # rmvpe
            if self.model_rmvpe is None:
                from infer.lib.rmvpe import RMVPE

                rmvpe_ckpt = _find_rmvpe_model()
                if not os.path.isfile(rmvpe_ckpt):
                    raise FileNotFoundError(
                        f"ไม่พบไฟล์โมเดล RMVPE ที่ {rmvpe_ckpt} (RMVPE model not found)"
                    )
                log_bilingual(
                    self.exp_dir,
                    f"🎙️ กำลังโหลดโมเดลสกัดระดับเสียง RMVPE: {rmvpe_ckpt}",
                    f"Loading RMVPE pitch extraction model: {rmvpe_ckpt} on {self.device}"
                )
                self.model_rmvpe = RMVPE(
                    rmvpe_ckpt,
                    is_half=self.is_half,
                    device=self.device,
                )

            pitch = self.model_rmvpe.infer_from_audio(audio, thred=0.03)
            pitch = self._align_length(pitch, expected_length)

        pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
        pitch[~np.isfinite(pitch)] = 0.0
        pitch[pitch < 0.0] = 0.0
        return pitch

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        f0 = np.maximum(f0, 0.0)

        f0_mel = 1127.0 * np.log1p(f0 / 700.0)
        voiced = f0_mel > 0.0
        f0_mel[voiced] = (
            (f0_mel[voiced] - self.f0_mel_min)
            * (self.f0_bin - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1.0
        )

        f0_mel = np.clip(f0_mel, 1.0, float(self.f0_bin - 1))
        coarse = np.rint(f0_mel).astype(np.int64)

        if coarse.size and (coarse.min() < 1 or coarse.max() > 255):
            coarse = np.clip(coarse, 1, 255)

        return coarse

    def process_paths(
        self,
        paths: Sequence[tuple[str, str, str]],
        method: str,
        worker_id: int = 0,
    ) -> tuple[int, int, int]:
        """ประมวลผลรายการไฟล์ของเวิร์กเกอร์นี้ คืนค่า (สำเร็จ, ข้าม, ผิดพลาด)"""
        if not paths:
            log_bilingual(
                self.exp_dir,
                "⚠️ ไม่มีรายการไฟล์เสียงที่ต้องสกัดระดับเสียง F0",
                "no-f0-todo (No audio files queued for F0 extraction)"
            )
            return 0, 0, 0

        total = len(paths)
        progress_interval = max(total // 10, 1)
        processed = 0
        skipped = 0
        failures = 0

        for index, (input_path, coarse_base, continuous_base) in enumerate(paths):
            try:
                coarse_path = Path(coarse_base + ".npy")
                continuous_path = Path(continuous_base + ".npy")

                # ข้ามเฉพาะกรณีที่มีไฟล์อยู่แล้วทั้งคู่และมีขนาดมากกว่า 128 ไบต์ (ไม่เสียหาย)
                if (
                    coarse_path.is_file()
                    and continuous_path.is_file()
                    and coarse_path.stat().st_size > 128
                    and continuous_path.stat().st_size > 128
                ):
                    skipped += 1
                    continue

                continuous_f0 = self.compute_f0(input_path, method)
                coarse_f0 = self.coarse_f0(continuous_f0)

                _save_npy_atomic(Path(continuous_base), continuous_f0)
                _save_npy_atomic(Path(coarse_base), coarse_f0)
                processed += 1

                if index % progress_interval == 0 or index == total - 1:
                    pct = ((index + 1) / total) * 100.0
                    fname = os.path.basename(input_path)
                    log_bilingual(
                        self.exp_dir,
                        f"⏳ [F0 Worker {worker_id}] ความคืบหน้า: [{index + 1}/{total}] ({pct:.1f}%) | ไฟล์: {fname}",
                        f"f0ing,now-{index + 1},all-{total},-{fname}"
                    )

            except KeyboardInterrupt:
                log_bilingual(self.exp_dir, "🛑 หยุดการสกัด F0 ตามคำสั่งผู้ใช้งาน", "F0 extraction interrupted by user")
                break
            except Exception as exc:
                failures += 1
                log_bilingual(
                    self.exp_dir,
                    f"❌ สกัด F0 ขัดข้องที่ไฟล์ {os.path.basename(input_path)}: {exc}",
                    f"f0fail-{index}-{input_path}-{traceback.format_exc()}"
                )

        return processed, skipped, failures


def _worker(
    paths: Sequence[tuple[str, str, str]],
    method: str,
    exp_dir: str,
    device: Any,
    is_half: bool,
    worker_id: int = 0,
) -> None:
    worker = FeatureInput(
        device=device,
        is_half=is_half,
        exp_dir=Path(exp_dir),
    )
    worker.process_paths(paths, method, worker_id=worker_id)


def _collect_paths(exp_dir: Path) -> list[tuple[str, str, str]]:
    input_root = exp_dir / "1_16k_wavs"
    coarse_root = exp_dir / "2a_f0"
    continuous_root = exp_dir / "2b-f0nsf"

    if not input_root.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์เสียง 16kHz (Input directory does not exist): {input_root}")

    coarse_root.mkdir(parents=True, exist_ok=True)
    continuous_root.mkdir(parents=True, exist_ok=True)

    valid_audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    paths = []
    for input_path in sorted(input_root.iterdir()):
        if not input_path.is_file():
            continue

        base_lower = input_path.name.lower()
        # [AUTO-PURGE] ลบไฟล์ขยะที่ไม่ใช่ไฟล์เสียงออกจาก 1_16k_wavs ทันที
        if input_path.suffix.lower() not in valid_audio_exts or base_lower in ("desktop.ini", "thumbs.db", ".ds_store"):
            try:
                os.chmod(str(input_path), stat.S_IWRITE)
                input_path.unlink(missing_ok=True)
                log_message(exp_dir, f"🧹 [AUTO-PURGE] ลบไฟล์ขยะที่ไม่ใช่ไฟล์เสียง (Deleted non-audio file): {input_path.name}")
            except Exception:
                pass
            continue

        if "spec" in base_lower or input_path.stat().st_size <= 512:
            continue

        # รักษามาตรฐานชื่อไฟล์: 2a_f0/<name.wav>.npy และ 2b-f0nsf/<name.wav>.npy
        # ซึ่งตรงกับ clean_stem() ใน Program.py และ train_dataset_inspector.py 100%
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
    """ดำเนินการสกัดระดับเสียง F0 พร้อมบริหารจัดการเธรดและหน่วยความจำอัตโนมัติ"""
    start_time = time.time()
    start_str = datetime.now().strftime("%H:%M:%S")
    exp_path = Path(exp_dir).expanduser().resolve()
    method_clean = method.lower().strip()
    if method_clean == "rmvpe_gpu":
        method_clean = "rmvpe"

    if method_clean not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported F0 method {method!r}; choose from {', '.join(sorted(SUPPORTED_METHODS))}"
        )

    log_bilingual(
        exp_path,
        f"🎵 เริ่มกระบวนการสกัดระดับเสียง (F0 Pitch Extraction v2.0) | อัลกอริทึม: {method_clean.upper()}",
        f"Starting F0 Pitch Extraction | Method: {method_clean.upper()} | Device: {device} | Start: {start_str}",
        use_sep=True
    )

    try:
        all_paths = _collect_paths(exp_path)
    except FileNotFoundError as fnf_err:
        log_bilingual(exp_path, f"❌ {fnf_err}", str(fnf_err), use_sep=True)
        return 1

    if not all_paths:
        log_bilingual(
            exp_path,
            "⚠️ ไม่พบไฟล์เสียงใน 1_16k_wavs สำหรับสกัดระดับเสียง (กรุณารันขั้นตอนที่ 1 ก่อน)",
            "no-f0-todo: No valid audio files found in 1_16k_wavs",
            use_sep=True
        )
        return 0

    # กรณีรันแบบแบ่ง Shard บน GPU (เช่น เรียกจาก extract_f0_rmvpe.py)
    if shard_count is not None:
        shard_count = max(1, int(shard_count))
        shard_index = max(0, min(int(shard_index), shard_count - 1))
        paths = all_paths[shard_index::shard_count]
        worker = FeatureInput(device=device, is_half=is_half, exp_dir=exp_path)
        proc_cnt, skip_cnt, fail_cnt = worker.process_paths(paths, method_clean, worker_id=shard_index)
        elapsed = int(time.time() - start_time)
        mm, ss = divmod(elapsed, 60)
        log_bilingual(
            exp_path,
            f"🎉 สกัดระดับเสียง F0 (Shard {shard_index + 1}/{shard_count}) เสร็จสมบูรณ์! | สำเร็จ: {proc_cnt} | ข้าม: {skip_cnt} | พลาด: {fail_cnt} | เวลา: {mm:02d}:{ss:02d}",
            f"F0 Shard {shard_index + 1}/{shard_count} Completed | Success: {proc_cnt} | Skipped: {skip_cnt} | Failed: {fail_cnt} | Elapsed: {mm:02d}:{ss:02d}",
            use_sep=True
        )
        return 1 if fail_cnt > 0 and proc_cnt == 0 and skip_cnt == 0 else 0

    # ปรับสมดุลเวิร์กเกอร์อัตโนมัติ: หากใช้ RMVPE (Neural Network) ให้รันในโปรเซสเดียวเพื่อไม่ให้โหลดโมเดลซ้ำจนล้น RAM/VRAM
    if method_clean == "rmvpe" or str(device) != "cpu":
        worker_count = 1
    else:
        worker_count = max(1, min(int(workers), len(all_paths), os.cpu_count() or 4))

    if worker_count == 1:
        worker = FeatureInput(
            device=device,
            is_half=is_half,
            exp_dir=exp_path,
        )
        proc_cnt, skip_cnt, fail_cnt = worker.process_paths(all_paths, method_clean, worker_id=0)
        elapsed = int(time.time() - start_time)
        mm, ss = divmod(elapsed, 60)
        log_bilingual(
            exp_path,
            f"🎉 สกัดระดับเสียง F0 เสร็จสมบูรณ์! | สำเร็จ: {proc_cnt} | ข้าม: {skip_cnt} | พลาด: {fail_cnt} | เวลาที่ใช้: {mm:02d}:{ss:02d}",
            f"F0 Extraction Completed | Success: {proc_cnt} | Skipped: {skip_cnt} | Failed: {fail_cnt} | Elapsed: {mm:02d}:{ss:02d}",
            use_sep=True
        )
        return 1 if (fail_cnt > 0 and proc_cnt == 0 and skip_cnt == 0) else 0

    import multiprocessing

    chunks = [all_paths[index::worker_count] for index in range(worker_count)]
    processes = []
    for w_idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        process = multiprocessing.Process(
            target=_worker,
            args=(chunk, method_clean, str(exp_path), device, is_half, w_idx),
        )
        process.start()
        processes.append(process)

    failed_process = False
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed_process = True
            log_message(
                exp_path,
                f"⚠️ เวิร์กเกอร์สกัด F0 สิ้นสุดด้วยรหัส (F0 worker exited with code): {process.exitcode}",
            )

    elapsed = int(time.time() - start_time)
    mm, ss = divmod(elapsed, 60)
    log_bilingual(
        exp_path,
        f"🎉 สกัดระดับเสียง F0 แบบหลายเธรด ({worker_count} Workers) เสร็จสมบูรณ์! | เวลาที่ใช้: {mm:02d}:{ss:02d}",
        f"Multi-process F0 Extraction ({worker_count} Workers) Completed | Elapsed: {mm:02d}:{ss:02d}",
        use_sep=True
    )
    return 1 if failed_process else 0
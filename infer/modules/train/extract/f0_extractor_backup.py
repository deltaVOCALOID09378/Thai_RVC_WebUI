# -*- coding: utf-8 -*-
"""Shared F0 extraction logic for RVC training scripts."""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence


if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import parselmouth
import pyworld


SUPPORTED_METHODS = {"pm", "harvest", "dio", "rmvpe"}


def log_message(exp_dir: Path, message: Any) -> None:
    """Write a message to stdout and the shared extraction log."""
    text = str(message)

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
    """Find the RMVPE checkpoint using the current directory or project root."""
    candidates = [
        Path.cwd() / "assets" / "rmvpe" / "rmvpe.pt",
        Path(__file__).resolve().parents[4] / "assets" / "rmvpe" / "rmvpe.pt",
    ]

    for path in candidates:
        if path.is_file():
            return str(path)

    # Keep the conventional path in the error message if the checkpoint is missing.
    return str(candidates[0])


def _save_npy_atomic(output_base: Path, array: np.ndarray) -> None:
    """Save a NumPy array atomically to output_base + '.npy'."""
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
        # Import lazily so GPU-related wrappers can set their environment first.
        from infer.lib.audio import load_audio

        audio = load_audio(path, self.fs)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        if audio.size == 0:
            raise ValueError("Audio contains no samples")

        return audio

    def _align_length(self, f0: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or trim the pitch track to the expected frame count."""
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
                constant_values=0,
            )

        return f0

    def compute_f0(self, path: str, method: str) -> np.ndarray:
        method = method.lower().strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported F0 method {method!r}; "
                f"choose from {', '.join(sorted(SUPPORTED_METHODS))}"
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
            frame_period = 1000 * self.hop / self.fs

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

        else:  # rmvpe
            if self.model_rmvpe is None:
                from infer.lib.rmvpe import RMVPE

                log_message(self.exp_dir, "Loading RMVPE model")
                self.model_rmvpe = RMVPE(
                    _find_rmvpe_model(),
                    is_half=self.is_half,
                    device=self.device,
                )

            pitch = self.model_rmvpe.infer_from_audio(audio, thred=0.03)

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

        if coarse.size and (coarse.min() < 1 or coarse.max() > 255):
            raise ValueError(
                f"Coarse F0 out of range: min={coarse.min()}, max={coarse.max()}"
            )

        return coarse

    def process_paths(
        self,
        paths: Sequence[tuple[str, str, str]],
        method: str,
    ) -> int:
        """Process this worker's files. Returns the number of failed files."""
        if not paths:
            log_message(self.exp_dir, "no-f0-todo")
            return 0

        log_message(self.exp_dir, f"todo-f0-{len(paths)}")
        progress_interval = max(len(paths) // 5, 1)
        failures = 0

        for index, (input_path, coarse_base, continuous_base) in enumerate(paths):
            try:
                if index % progress_interval == 0:
                    log_message(
                        self.exp_dir,
                        f"f0ing,now-{index},all-{len(paths)},-{input_path}",
                    )

                coarse_path = Path(coarse_base + ".npy")
                continuous_path = Path(continuous_base + ".npy")

                # Resume only when both output files are present.
                if coarse_path.is_file() and continuous_path.is_file():
                    continue

                continuous_f0 = self.compute_f0(input_path, method)
                coarse_f0 = self.coarse_f0(continuous_f0)

                # Write both results only after both arrays have been computed.
                _save_npy_atomic(Path(continuous_base), continuous_f0)
                _save_npy_atomic(Path(coarse_base), coarse_f0)

            except Exception:
                failures += 1
                log_message(
                    self.exp_dir,
                    f"f0fail-{index}-{input_path}-{traceback.format_exc()}",
                )

        return failures


def _worker(
    paths: Sequence[tuple[str, str, str]],
    method: str,
    exp_dir: str,
    device: Any,
    is_half: bool,
) -> None:
    worker = FeatureInput(
        device=device,
        is_half=is_half,
        exp_dir=Path(exp_dir),
    )
    worker.process_paths(paths, method)


def _collect_paths(exp_dir: Path) -> list[tuple[str, str, str]]:
    input_root = exp_dir / "1_16k_wavs"
    coarse_root = exp_dir / "2a_f0"
    continuous_root = exp_dir / "2b-f0nsf"

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    coarse_root.mkdir(parents=True, exist_ok=True)
    continuous_root.mkdir(parents=True, exist_ok=True)

    paths = []
    for input_path in sorted(input_root.iterdir()):
        if not input_path.is_file() or "spec" in str(input_path).lower():
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
    """Run extraction, optionally processing one shard of the input list."""
    exp_path = Path(exp_dir).expanduser().resolve()
    method = method.lower().strip()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported F0 method {method!r}; "
            f"choose from {', '.join(sorted(SUPPORTED_METHODS))}"
        )

    all_paths = _collect_paths(exp_path)

    if shard_count is not None:
        shard_count = int(shard_count)
        shard_index = int(shard_index)

        if shard_count < 1:
            raise ValueError("shard_count must be at least 1")
        if not 0 <= shard_index < shard_count:
            raise ValueError(
                f"shard_index must be between 0 and {shard_count - 1}"
            )

        paths = all_paths[shard_index::shard_count]
        _worker(paths, method, str(exp_path), device, is_half)
        return 0

    if not all_paths:
        log_message(exp_path, "no-f0-todo")
        return 0

    worker_count = max(1, min(int(workers), len(all_paths)))
    chunks = [all_paths[index::worker_count] for index in range(worker_count)]

    if worker_count == 1:
        worker = FeatureInput(
            device=device,
            is_half=is_half,
            exp_dir=exp_path,
        )
        failures = worker.process_paths(chunks[0], method)
        return 1 if failures else 0

    # Import here to keep single-worker and DirectML use lightweight.
    import multiprocessing

    processes = []
    for chunk in chunks:
        process = multiprocessing.Process(
            target=_worker,
            args=(chunk, method, str(exp_path), device, is_half),
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
                f"F0 worker exited with code {process.exitcode}",
            )

    return 1 if failed_process else 0
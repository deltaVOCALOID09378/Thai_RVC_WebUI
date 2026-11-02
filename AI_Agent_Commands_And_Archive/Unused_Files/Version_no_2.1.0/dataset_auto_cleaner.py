# ==============================================================================
# File Name   : dataset_auto_cleaner.py
# Module      : Universal Dataset & Experiment Auto-Cleaner Sentinel
# Project     : Thai RVC Studio & DiffSinger Unified Engine
# Version     : 2.2.0 (PR #3 CodeRabbit Hardened Edition)
# Description : Automatically scans, unlocks, and removes Windows/macOS system
#               junk files (desktop.ini, Thumbs.db, .DS_Store), zero-byte files,
#               and unreadable/corrupted audio files before training/extraction.
#
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by : Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys
import stat
import wave
import shutil
from typing import Dict, Any, Set

try:
    import soundfile as sf
except ImportError:
    sf = None

JUNK_FILENAMES: Set[str] = {
    "desktop.ini",
    "thumbs.db",
    "ehthumbs.db",
    "ehthumbs_vista.db",
    ".ds_store",
    "folder.htt",
    "icon\r",
}

JUNK_EXTENSIONS: Set[str] = {
    ".ini",
    ".db",
    ".tmp",
    ".bak",
    ".lnk",
    ".log",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".DS_Store",
}

VALID_AUDIO_EXTENSIONS: Set[str] = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
}


def _unlock_and_remove(file_path: str) -> bool:
    """Safely removes a file after clearing read-only attributes (skipping chmod on symlinks)."""
    try:
        if not os.path.islink(file_path):
            os.chmod(file_path, stat.S_IWRITE)
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"[Auto-Cleaner][WARN] Could not remove '{file_path}': {e}")
        return False


def _is_audio_readable(file_path: str, ext: str) -> bool:
    """Verifies that an audio file has a readable header and non-zero frames."""
    try:
        if os.path.getsize(file_path) <= 44:
            return False
        if sf is not None:
            info = sf.info(file_path)
            return info.frames > 0 and info.samplerate > 0
        if ext == ".wav":
            with wave.open(file_path, "rb") as wf:
                return wf.getnframes() > 0 and wf.getframerate() > 0
        with open(file_path, "rb") as f:
            header = f.read(16)
            return len(header) >= 4
    except Exception:
        return False


def purge_junk_from_directory(
    target_dir: str,
    strict_audio_only: bool = False,
    verify_audio_headers: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Scans `target_dir` recursively and purges system metadata files, AppleDouble
    files (`._*`), zero-byte files, and optionally non-audio or unreadable audio files.
    """
    report: Dict[str, Any] = {
        "valid_dir": False,
        "target_dir": target_dir,
        "scanned": 0,
        "removed_junk": 0,
        "removed_zero_byte": 0,
        "removed_corrupt_audio": 0,
        "valid_audio_remaining": 0,
        "errors": [],
    }

    if not target_dir or not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        msg = f"[Auto-Cleaner][ERROR] Target path does not exist or is not a directory: '{target_dir}'"
        if verbose:
            print(msg)
        report["errors"].append(msg)
        return report

    report["valid_dir"] = True

    for root, _, files in os.walk(target_dir):
        for fname in files:
            report["scanned"] += 1
            fpath = os.path.join(root, fname)
            lower_name = fname.lower()
            _, ext = os.path.splitext(lower_name)

            # 1. Remove known OS junk files or AppleDouble resource forks
            if lower_name in JUNK_FILENAMES or fname.startswith("._") or ext in JUNK_EXTENSIONS:
                if _unlock_and_remove(fpath):
                    report["removed_junk"] += 1
                    if verbose:
                        print(f"[Auto-Cleaner] Purged system junk file: {fpath}")
                else:
                    report["errors"].append(fpath)
                continue

            # 2. Remove zero-byte files
            try:
                fsize = os.path.getsize(fpath)
            except OSError as e:
                report["errors"].append(f"{fpath}: {e}")
                continue

            if fsize == 0:
                if _unlock_and_remove(fpath):
                    report["removed_zero_byte"] += 1
                    if verbose:
                        print(f"[Auto-Cleaner] Purged 0-byte empty file: {fpath}")
                else:
                    report["errors"].append(fpath)
                continue

            # 3. Check audio extensions and header readability
            if ext in VALID_AUDIO_EXTENSIONS:
                if verify_audio_headers and not _is_audio_readable(fpath, ext):
                    if _unlock_and_remove(fpath):
                        report["removed_corrupt_audio"] += 1
                        if verbose:
                            print(f"[Auto-Cleaner] Purged unreadable/corrupt audio file: {fpath}")
                    else:
                        report["errors"].append(fpath)
                else:
                    report["valid_audio_remaining"] += 1
            elif strict_audio_only:
                if _unlock_and_remove(fpath):
                    report["removed_junk"] += 1
                    if verbose:
                        print(f"[Auto-Cleaner] Purged non-audio file in strict mode: {fpath}")
                else:
                    report["errors"].append(fpath)

    return report


def clean_rvc_experiment_dir(exp_dir: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Cleans an RVC experiment folder (`logs/<exp_name>`), including its root
    `desktop.ini` and sliced audio folders (`0_gt_wavs`, `1_16k_wavs`).
    """
    summary = {
        "exp_dir": exp_dir,
        "valid_dir": False,
        "subfolders_cleaned": {},
    }
    if not exp_dir or not os.path.isdir(exp_dir):
        if verbose:
            print(f"[Auto-Cleaner][WARN] Experiment directory not found: '{exp_dir}'")
        return summary

    summary["valid_dir"] = True

    exp_desktop = os.path.join(exp_dir, "desktop.ini")
    if os.path.exists(exp_desktop) or os.path.islink(exp_desktop):
        try:
            if not os.path.islink(exp_desktop):
                os.chmod(exp_desktop, stat.S_IWRITE)
            os.remove(exp_desktop)
            if verbose:
                print(f"[Auto-Cleaner] Purged root desktop.ini in '{exp_desktop}'")
        except Exception as e:
            print(f"[Auto-Cleaner][WARN] Failed to remove '{exp_desktop}': {e}")

    for sub in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature256", "3_feature768"):
        sub_path = os.path.join(exp_dir, sub)
        if os.path.isdir(sub_path):
            is_wav_dir = sub in ("0_gt_wavs", "1_16k_wavs")
            summary["subfolders_cleaned"][sub] = purge_junk_from_directory(
                sub_path,
                strict_audio_only=is_wav_dir,
                verify_audio_headers=is_wav_dir,
                verbose=verbose,
            )

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dataset_auto_cleaner.py <target_directory> [--strict-audio]")
        sys.exit(1)

    target = sys.argv[1]
    strict_flag = "--strict-audio" in sys.argv[2:]

    if not os.path.exists(target) or not os.path.isdir(target):
        print(f"[Auto-Cleaner][ERROR] Invalid target directory: '{target}'")
        sys.exit(1)

    result = purge_junk_from_directory(target, strict_audio_only=strict_flag, verify_audio_headers=True, verbose=True)
    clean_rvc_experiment_dir(target, verbose=True)
    print(
        f"[Auto-Cleaner][DONE] Scanned={result['scanned']} | "
        f"JunkRemoved={result['removed_junk']} | "
        f"ZeroByteRemoved={result['removed_zero_byte']} | "
        f"CorruptAudioRemoved={result['removed_corrupt_audio']} | "
        f"ValidAudio={result['valid_audio_remaining']}"
    )
    sys.exit(0 if result["valid_dir"] else 1)

# -*- coding: utf-8 -*-
# ==============================================================================
# File: multispeaker.py
# Path: tools\multispeaker.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 2.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# Description: Multi-Speaker Dataset Manifest Builder and Validator (Thai & English)

import hashlib
import json
import os
import re


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
}
SPEAKER_ID_MIN = 0
SPEAKER_ID_MAX = 109
MANIFEST_VERSION = 1
SPEAKER_DIR_RE = re.compile(r"^(.+)_(\d+)_(\d+)$")


class ManifestError(Exception):
    def __init__(self, key, *values):
        self.key = key
        self.values = values
        super().__init__(key, *values)


def audio_files(folder):
    result = []
    if not os.path.isdir(folder):
        return result
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
                result.append(os.path.abspath(os.path.join(root, name)))
    return result


def _manifest_entry(path, speaker_name, speaker_id, repeat, index):
    digest = hashlib.sha1(
        os.path.normcase(os.path.abspath(path)).encode("utf8")
    ).hexdigest()[:10]
    return {
        "path": os.path.abspath(path),
        "speaker_name": speaker_name,
        "speaker_id": int(speaker_id),
        "repeat": int(repeat),
        "output_key": "ms%04d_s%03d_%s" % (index, int(speaker_id), digest),
    }


def build_manifest_from_root(root):
    root = os.path.abspath(str(root or "").strip())
    if not os.path.isdir(root):
        raise ManifestError("ไม่พบโฟลเดอร์หลักของชุดข้อมูลหลายผู้พูด (Multi-speaker root folder does not exist): %s", root)
    entries = []
    invalid = []
    names_by_id = {}
    child_dirs = [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]
    if not child_dirs:
        raise ManifestError("ไม่พบโฟลเดอร์ย่อยในชุดข้อมูลหลายผู้พูด (No subdirectories found in multi-speaker root folder)")
    for child in child_dirs:
        name = os.path.basename(child)
        match = SPEAKER_DIR_RE.match(name)
        if not match:
            invalid.append(name)
            continue
        speaker_name = match.group(1).strip()
        speaker_id = int(match.group(2))
        repeat = int(match.group(3))
        files = audio_files(child)
        invalid_name = (
            not speaker_name
            or "|" in speaker_name
            or "\n" in speaker_name
            or "\r" in speaker_name
        )
        inconsistent = (
            speaker_id in names_by_id and names_by_id[speaker_id] != speaker_name
        )
        if (
            invalid_name
            or speaker_id < SPEAKER_ID_MIN
            or speaker_id > SPEAKER_ID_MAX
            or repeat < 1
            or not files
            or inconsistent
        ):
            invalid.append(name)
            continue
        names_by_id[speaker_id] = speaker_name
        for path in files:
            entries.append(
                _manifest_entry(
                    path, speaker_name, speaker_id, repeat, len(entries)
                )
            )
    if invalid:
        raise ManifestError(
            "โฟลเดอร์ย่อยหลายผู้พูดไม่ถูกต้อง (รูปแบบต้องเป็น ชื่อ_ID_รอบ เช่น Singer_0_1, ID 0~109, รอบ > 0): %s",
            ", ".join(invalid),
        )
    if not entries:
        raise ManifestError("ไม่พบไฟล์เสียงที่ใช้งานได้ในโฟลเดอร์หลายผู้พูด (No valid audio found in multi-speaker root folder)")
    return {
        "version": MANIFEST_VERSION,
        "source": "folder_scan",
        "root": root,
        "speakers": [
            {"id": speaker_id, "name": names_by_id[speaker_id]}
            for speaker_id in sorted(names_by_id)
        ],
        "entries": entries,
    }


def build_manifest_from_rows(rows, root=""):
    entries = []
    valid_rows = []
    invalid_rows = []
    names_by_id = {}
    for row_index, row in enumerate(rows, 1):
        path, speaker_name, speaker_id, repeat = [
            "" if value is None else str(value).strip() for value in row
        ]
        if not path and not speaker_name and not speaker_id and not repeat:
            continue
        try:
            speaker_id_int = int(float(speaker_id))
            repeat_int = int(float(repeat))
            if str(speaker_id_int) != speaker_id and str(float(speaker_id_int)) != speaker_id:
                raise ValueError
            if str(repeat_int) != repeat and str(float(repeat_int)) != repeat:
                raise ValueError
            if speaker_id_int < SPEAKER_ID_MIN or speaker_id_int > SPEAKER_ID_MAX or repeat_int < 1:
                raise ValueError
        except (TypeError, ValueError):
            invalid_rows.append(row_index)
            continue
        invalid_name = (
            not speaker_name
            or "|" in speaker_name
            or "\n" in speaker_name
            or "\r" in speaker_name
        )
        inconsistent = (
            speaker_id_int in names_by_id
            and names_by_id[speaker_id_int] != speaker_name
        )
        path = os.path.abspath(path)
        files = audio_files(path)
        if invalid_name or inconsistent or not files:
            invalid_rows.append(row_index)
            continue
        names_by_id[speaker_id_int] = speaker_name
        valid_rows.append(
            {
                "path": path,
                "speaker_name": speaker_name,
                "speaker_id": speaker_id_int,
                "repeat": repeat_int,
            }
        )
        for file_path in files:
            entries.append(
                _manifest_entry(
                    file_path,
                    speaker_name,
                    speaker_id_int,
                    repeat_int,
                    len(entries),
                )
            )
    if not entries:
        raise ManifestError("ไม่มีแถวข้อมูลชุดฝึกสอนหลายผู้พูดที่ถูกต้อง (No valid multi-speaker dataset rows)")
    manifest = {
        "version": MANIFEST_VERSION,
        "source": "helper",
        "root": os.path.abspath(root) if root else "",
        "rows": valid_rows,
        "speakers": [
            {"id": speaker_id, "name": names_by_id[speaker_id]}
            for speaker_id in sorted(names_by_id)
        ],
        "entries": entries,
    }
    return manifest, invalid_rows


def write_manifest(exp_dir, manifest):
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "multispeaker_manifest.json")
    with open(path, "w", encoding="utf8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def load_manifest(exp_dir):
    path = os.path.join(exp_dir, "multispeaker_manifest.json")
    if not os.path.isfile(path):
        raise ManifestError("ไม่พบรายการ Manifest ของชุดข้อมูลหลายผู้พูด (Multi-speaker manifest not found)")
    with open(path, "r", encoding="utf8") as file:
        manifest = json.load(file)
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ManifestError("รายการ Manifest ไม่มีไฟล์เสียงที่ถูกต้อง (No valid audio entries in multi-speaker manifest)")
    seen = set()
    names_by_id = {}
    for entry in entries:
        try:
            path_value = os.path.abspath(str(entry["path"]))
            speaker_name = str(entry["speaker_name"]).strip()
            speaker_id = int(entry["speaker_id"])
            repeat = int(entry["repeat"])
            output_key = str(entry["output_key"])
        except (KeyError, TypeError, ValueError):
            raise ManifestError("รูปแบบรายการ Manifest ไม่ถูกต้อง (Malformed multi-speaker manifest format)")
        inconsistent = (
            speaker_id in names_by_id and names_by_id[speaker_id] != speaker_name
        )
        if (
            not os.path.isfile(path_value)
            or not speaker_name
            or "|" in speaker_name
            or "\n" in speaker_name
            or "\r" in speaker_name
            or speaker_id < 0
            or speaker_id > 109
            or repeat < 1
            or not output_key
            or inconsistent
        ):
            raise ManifestError("รายการ Manifest มีข้อมูลที่ไม่ถูกต้อง (Invalid entry in multi-speaker manifest): %s", path_value)
        if output_key in seen:
            raise ManifestError("พบ Output Key ซ้ำซ้อนในรายการ Manifest (Duplicate output key in multi-speaker manifest): %s", output_key)
        seen.add(output_key)
        names_by_id[speaker_id] = speaker_name
    manifest["speakers"] = [
        {"id": speaker_id, "name": names_by_id[speaker_id]}
        for speaker_id in sorted(names_by_id)
    ]
    return manifest

# ==============================================================================
# File Name: train_index.py
# Version: 1.2
# Location: / (Root Directory)
# Function: Train Faiss index for Retrieval-based Voice Conversion (RVC), translating all logs to Thai and auto-exporting the final .index to assets\indices using the singer's model name.
#           ( ฝึกสอน Faiss Index สำหรับ RVC แปลงเป็นภาษาไทยทั้งหมด และคัดลอกไฟล์ .index อัตโนมัติไปยัง assets\indices โดยใช้ชื่อโมเดลนักร้อง )
# Credits: Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by RVC Project
# ==============================================================================

import os
import platform
import sys
import traceback
import glob
import json
import shutil

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tools.progress import should_report

# Prevent memory leak warnings on Windows for KMeans
# ป้องกันการแจ้งเตือนหน่วยความจำรั่วไหลใน KMeans บน Windows
os.environ["OMP_NUM_THREADS"] = "1"

# Command line arguments initialization
# การรับค่าตัวแปรจาก Command line
exp_name = sys.argv[1]
version = sys.argv[2]
outside_index_root = sys.argv[3]
n_cpu = int(sys.argv[4])
index_mode = sys.argv[5] if len(sys.argv) > 5 else "auto"

exp_dir = os.path.join("logs", exp_name)
feature_dir = os.path.join(
    exp_dir, "3_feature256" if version == "v1" else "3_feature768"
)
log_path = os.path.join(exp_dir, "train_index.log")
os.makedirs(exp_dir, exist_ok=True)


def log(message):
    """
    Log message to console and file.
    ( บันทึกข้อความลงในคอนโซลและไฟล์ )
    """
    print(message, flush=True)
    with open(log_path, "a", encoding="utf8") as f:
        f.write(str(message) + "\n")


# Clear old log file
# ล้างข้อมูลไฟล์ log เดิม
with open(log_path, "w", encoding="utf8"):
    pass


def newest_index(pattern):
    """
    Find the most recently modified index file matching the pattern.
    ( ค้นหาไฟล์ index ที่ถูกแก้ไขล่าสุดตามรูปแบบที่กำหนด )
    """
    paths = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    return max(paths, key=os.path.getmtime) if paths else ""


def speaker_scope(message, speaker_id):
    """
    Format message with speaker ID if available.
    ( จัดรูปแบบข้อความโดยระบุหมายเลขผู้พูดหากมี )
    """
    if speaker_id is None:
        return message
    return "%s：%s | %s" % ("รหัสผู้พูด (0~109)", speaker_id, message)


def link_added_index(added_path, speaker_id=None):
    """
    Create a hardlink or symlink for the trained index to the outside directory.
    ( สร้างลิงก์สำหรับไฟล์ index ที่ฝึกสอนเสร็จแล้วไปยังไดเรกทอรีภายนอก )
    """
    added_name = os.path.basename(added_path)
    try:
        os.makedirs(outside_index_root, exist_ok=True)
        source = os.path.abspath(added_path)
        outside_root = os.path.abspath(outside_index_root)
        
        if os.path.commonpath([source, outside_root]) == outside_root:
            log(speaker_scope("[การสร้าง Index] มีลิงก์ Index ภายนอกอยู่แล้ว: %s" % source, speaker_id))
            return
            
        target = os.path.abspath(
            os.path.join(outside_index_root, "%s_%s" % (exp_name, added_name))
        )
        
        if os.path.lexists(target):
            try:
                if os.path.samefile(source, target):
                    log(speaker_scope("[การสร้าง Index] มีลิงก์ Index ภายนอกอยู่แล้ว: %s" % target, speaker_id))
                    return
            except (FileNotFoundError, OSError):
                pass
            os.unlink(target)
            
        if platform.system() == "Windows":
            os.link(source, target)
        else:
            os.symlink(source, target)
            
        log(speaker_scope("[การสร้าง Index] ลิงก์ Index ไปยังไดเรกทอรีภายนอกเรียบร้อยแล้ว: %s" % outside_index_root, speaker_id))
    except Exception:
        log(
            speaker_scope(
                "[การสร้าง Index][ล้มเหลว] ไม่สามารถลิงก์ Index ไปยังไดเรกทอรีภายนอกได้: %s\n%s"
                % (outside_index_root, traceback.format_exc()),
                speaker_id,
            )
        )


# Validate feature directory
# ตรวจสอบความถูกต้องของโฟลเดอร์คุณลักษณะ (Feature)
if not os.path.isdir(feature_dir) or not os.listdir(feature_dir):
    log("[การสร้าง Index][ล้มเหลว] กรุณาสกัดคุณลักษณะ (Feature Extraction) ก่อนดำเนินการต่อ")
    raise SystemExit(1)

manifest_path = os.path.join(exp_dir, "multispeaker_manifest.json")
manifest_by_key = {}

# Parse manifest for multispeaker processing
# ประมวลผลไฟล์ manifest สำหรับการฝึกสอนหลายผู้พูด
if index_mode != "single" and os.path.isfile(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf8") as file:
            manifest = json.load(file)
        entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
        manifest_by_key = {
            str(entry["output_key"]): int(entry["speaker_id"])
            for entry in entries
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        manifest_by_key = {}

feature_paths = [
    os.path.join(feature_dir, name)
    for name in sorted(os.listdir(feature_dir))
    if name.lower().endswith(".npy")
]
feature_groups = {}

# Group features by speaker
# จัดกลุ่มคุณลักษณะ (Feature) ตามหมายเลขผู้พูด
if manifest_by_key:
    for path in feature_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        output_key = stem if stem in manifest_by_key else stem.rsplit("_", 1)[0]
        speaker_id = manifest_by_key.get(output_key)
        if speaker_id is not None:
            feature_groups.setdefault(speaker_id, []).append(path)
else:
    feature_groups[None] = feature_paths

if not feature_groups or not any(feature_groups.values()):
    log("[การสร้าง Index][ล้มเหลว] กรุณาสกัดคุณลักษณะ (Feature Extraction) ก่อนดำเนินการต่อ")
    raise SystemExit(1)


def train_one_speaker(speaker_id, paths):
    """
    Train Faiss index for a specific speaker and auto-export to assets/indices.
    ( ฝึกสอน Faiss index สำหรับผู้พูดแต่ละบุคคล และคัดลอกไปที่ assets/indices อัตโนมัติ )
    """
    scope = lambda message: speaker_scope(message, speaker_id)
    features = [np.load(path) for path in paths]
    big_npy = np.concatenate(features, 0)
    big_npy = big_npy[np.random.permutation(big_npy.shape[0])]
    suffix = "" if speaker_id is None else "_spkid%s" % speaker_id
    total_path = os.path.join(exp_dir, "total_fea%s.npy" % suffix)
    np.save(total_path, big_npy)

    trained_pattern = os.path.join(
        exp_dir,
        "trained_IVF*_Flat_nprobe_*_%s_%s%s.index"
        % (exp_name, version, suffix),
    )
    added_pattern = os.path.join(
        exp_dir,
        "added_IVF*_Flat_nprobe_*_%s_%s%s.index"
        % (exp_name, version, suffix),
    )
    
    existing_trained_path = newest_index(trained_pattern)
    existing_added_path = newest_index(added_pattern)
    
    if not existing_added_path:
        existing_added_path = newest_index(
            os.path.join(
                outside_index_root,
                "%s_*added_IVF*_Flat_nprobe_*_%s_%s%s.index"
                % (exp_name, exp_name, version, suffix),
            )
        )
        
    if existing_added_path:
        if existing_trained_path:
            log(
                scope(
                    "[การสร้าง Index][ข้าม] มี trained Index อยู่แล้ว: %s"
                    % os.path.basename(existing_trained_path)
                )
            )
        log(
            scope(
                "[การสร้าง Index][ข้าม] มี added Index อยู่แล้ว: %s"
                % os.path.basename(existing_added_path)
            )
        )
        link_added_index(existing_added_path, speaker_id)
        return

    # Use MiniBatchKMeans for large datasets
    # ใช้ MiniBatchKMeans เพื่อจำกัดขนาดข้อมูลที่ใหญ่เกินไป
    if big_npy.shape[0] > 200000:
        log(
            scope(
                "[การสร้าง Index] กำลังจัดกลุ่มคุณลักษณะ %s รายการ เป็น 10000 ศูนย์กลาง"
                % big_npy.shape[0]
            )
        )
        try:
            big_npy = MiniBatchKMeans(
                n_clusters=10000,
                verbose=False,
                batch_size=256 * n_cpu,
                compute_labels=False,
                init="random",
            ).fit(big_npy).cluster_centers_
        except Exception:
            log(
                scope(
                    "[การสร้าง Index][ล้มเหลว] การจัดกลุ่มล้มเหลว จะดำเนินการต่อด้วยคุณลักษณะดั้งเดิม\n%s"
                    % traceback.format_exc()
                )
            )

    n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
    log(scope("[การสร้าง Index] รูปแบบคุณลักษณะ: %s | จำนวน IVF: %s" % (big_npy.shape, n_ivf)))
    
    if existing_trained_path:
        trained_path = existing_trained_path
        index = faiss.read_index(trained_path)
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        log(
            scope(
                "[การสร้าง Index][ข้าม] มี trained Index อยู่แล้ว: %s"
                % os.path.basename(trained_path)
            )
        )
    else:
        index = faiss.index_factory(
            256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf
        )
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        trained_path = os.path.join(
            exp_dir,
            "trained_IVF%s_Flat_nprobe_%s_%s_%s%s.index"
            % (n_ivf, index_ivf.nprobe, exp_name, version, suffix),
        )
        log(scope("[การสร้าง Index] กำลังเริ่มต้นฝึกสอน Index..."))
        index.train(big_npy)
        faiss.write_index(index, trained_path)

    log(scope("[การสร้าง Index] กำลังเขียนเวกเตอร์คุณลักษณะลงในไฟล์..."))
    starts = list(range(0, big_npy.shape[0], 8192))
    for batch_index, start in enumerate(starts):
        index.add(big_npy[start : start + 8192])
        if should_report(batch_index, len(starts), 10):
            log(
                scope(
                    "[การสร้าง Index] ความคืบหน้าการเขียนเวกเตอร์: %s/%s"
                    % (batch_index + 1, len(starts))
                )
            )

    added_name = "added_IVF%s_Flat_nprobe_%s_%s_%s%s.index" % (
        n_ivf,
        index_ivf.nprobe,
        exp_name,
        version,
        suffix,
    )
    added_path = os.path.join(exp_dir, added_name)
    faiss.write_index(index, added_path)
    log(scope("[การสร้าง Index] สร้างไฟล์ Index สำเร็จ: %s" % added_name))
    link_added_index(added_path, speaker_id)

    # -------------------------------------------------------------------------
    # Auto-Export to assets\indices using the singer's model name
    # คัดลอกไฟล์อัตโนมัติไปยัง assets\indices โดยใช้ชื่อโมเดลของนักร้อง
    # -------------------------------------------------------------------------
    assets_indices_dir = os.path.join("assets", "indices")
    os.makedirs(assets_indices_dir, exist_ok=True)
    
    # กำหนดชื่อไฟล์โดยใช้ชื่อโมเดลนักร้อง (exp_name) และเวอร์ชั่น
    final_export_name = f"{exp_name}_{version}{suffix}.index"
    final_export_path = os.path.join(assets_indices_dir, final_export_name)
    
    try:
        shutil.copy2(added_path, final_export_path)
        log(scope(f"[การส่งออก Index] สำเร็จ! ไฟล์ของนักร้องถูกบันทึกไว้ที่: {final_export_path}"))
    except Exception as e:
        log(scope(f"[การส่งออก Index][ล้มเหลว] เกิดข้อผิดพลาดในการคัดลอกไฟล์ไปยัง {assets_indices_dir}:\n{traceback.format_exc()}"))


# Process training for all organized speakers
# เริ่มกระบวนการฝึกสอนโมเดลสำหรับทุกผู้พูดที่จัดกลุ่มไว้
for speaker_id in sorted(feature_groups, key=lambda value: -1 if value is None else value):
    train_one_speaker(speaker_id, feature_groups[speaker_id])
# -*- coding: utf-8 -*-
# ==============================================================================
# File Name: dataset_auto_cleaner.py
# Version: 2.2
# Location: Root Directory / Tools
# Function: Automatic Non-Audio & System Garbage File Purger for RVC
# Credit: Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#         Original by Patiphat Wongyai (Delta) & RVC Project
# Description: Detects and cleans zero-byte, corrupted, and system garbage files
#              with real-time stdout flushing for WebUI syncing.
# ==============================================================================

import os
import sys
import stat
import logging
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AutoCleaner")

VALID_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".opus"
}

VALID_FEATURE_EXTENSIONS = {
    ".npy", ".index"
}

def print_log(msg: str):
    """Bilingual Real-time Log Flusher"""
    print(msg, flush=True)

def purge_non_audio_files(target_dir: str, allow_features: bool = False, verbose: bool = True) -> list:
    """
    สแกนและลบไฟล์ที่ไม่ใช่ไฟล์เสียงออกจากโฟลเดอร์ทันที
    (Scan and forcefully remove non-audio files from the target directory)
    """
    if not os.path.exists(target_dir):
        return []

    deleted_files = []
    allowed_exts = set(VALID_AUDIO_EXTENSIONS)
    if allow_features:
        allowed_exts.update(VALID_FEATURE_EXTENSIONS)

    for root, dirs, files in os.walk(target_dir):
        for f in files:
            file_path = os.path.join(root, f).replace("\\", "/")
            ext = os.path.splitext(f)[1].lower()
            base_name = f.lower()

            is_system_garbage = base_name in (
                "desktop.ini", "thumbs.db", ".ds_store", "ehthumbs.db", "folder.htt"
            ) or base_name.startswith("._")

            is_not_allowed = ext not in allowed_exts

            is_empty_or_corrupt = False
            try:
                if os.path.getsize(file_path) == 0:
                    is_empty_or_corrupt = True
            except Exception:
                is_empty_or_corrupt = True

            if is_system_garbage or is_not_allowed or is_empty_or_corrupt:
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    if verbose:
                        if is_system_garbage:
                            reason = "ไฟล์ระบบขยะ (System Garbage)"
                        elif is_empty_or_corrupt:
                            reason = "ไฟล์เสียหาย/ขนาด 0 ไบต์ (Corrupted/0-Byte)"
                        else:
                            reason = f"ไม่ใช่ไฟล์เสียง (Unsupported {ext})"
                        print_log(f"🗑️ [AUTO-PURGE] ลบไฟล์ออกจากดาต้าเซ็ต ({reason}): {file_path}")
                except Exception as e:
                    print_log(f"⚠️ [WARNING] ไม่สามารถลบไฟล์ (Cannot delete file) {file_path}: {e}")

    if verbose and deleted_files:
        print_log(f"✨ [CLEAN-COMPLETE] ทำความสะอาดสำเร็จ ลบไฟล์ขยะไปทั้งหมด {len(deleted_files)} ไฟล์ (Purged {len(deleted_files)} items) in {target_dir}")

    # Force Garbage Collection after I/O heavy tasks
    gc.collect()
    return deleted_files

def sanitize_dataset_and_logs(exp_path: str, dataset_path: str = None):
    """
    ทำความสะอาดทั้งโฟลเดอร์ดาต้าเซ็ตต้นทาง และโฟลเดอร์ logs ของการทดลอง
    """
    print_log("=" * 65)
    print_log("  🧹 DELTA SYNTH - ระบบลบไฟล์ขยะและคัดกรองเฉพาะไฟล์เสียงอัตโนมัติ (Auto-Sanitizer)")
    print_log("=" * 65)
    total_deleted = 0
    
    if dataset_path and os.path.exists(dataset_path):
        print_log(f"[SCAN] กำลังตรวจสอบโฟลเดอร์ดาต้าเซ็ตต้นทาง (Scanning source dataset): {dataset_path}")
        d1 = purge_non_audio_files(dataset_path, allow_features=False)
        total_deleted += len(d1)

    if os.path.exists(exp_path):
        gt_dir = os.path.join(exp_path, "0_gt_wavs").replace("\\", "/")
        wav16k_dir = os.path.join(exp_path, "1_16k_wavs").replace("\\", "/")
        
        if os.path.exists(gt_dir):
            print_log(f"[SCAN] กำลังตรวจสอบโฟลเดอร์ (Scanning) 0_gt_wavs...")
            d2 = purge_non_audio_files(gt_dir, allow_features=False)
            total_deleted += len(d2)
            
        if os.path.exists(wav16k_dir):
            print_log(f"[SCAN] กำลังตรวจสอบโฟลเดอร์ (Scanning) 1_16k_wavs...")
            d3 = purge_non_audio_files(wav16k_dir, allow_features=False)
            total_deleted += len(d3)
            
        exp_desktop = os.path.join(exp_path, "desktop.ini").replace("\\", "/")
        if os.path.exists(exp_desktop):
            try:
                os.chmod(exp_desktop, stat.S_IWRITE)
                os.remove(exp_desktop)
                total_deleted += 1
                print_log(f"🗑️ [AUTO-PURGE] ลบ (Deleted) {exp_desktop}")
            except Exception:
                pass

    print_log("-" * 65)
    print_log(f"🎉 พร้อมสำหรับการประมวลผล: ลบไฟล์ที่ไม่เกี่ยวข้องทั้งหมด {total_deleted} ไฟล์ (Ready: {total_deleted} garbage files removed)")
    print_log("=" * 65)
    return total_deleted

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        purge_non_audio_files(path, allow_features=False, verbose=True)
    else:
        print_log("Usage: python dataset_auto_cleaner.py <directory_path>")
# -*- coding: utf-8 -*-
# ==============================================================================
# File: Download_Required_Tools.py
# Path: Download_Required_Tools.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 2.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# Original by Patiphat Wongyai (Delta) & RVC Project Developers
# Date: 2026-09-04
# Standard: DELTA SYNTH Architecture & Optimization Standard (AGENT.md)

import os
import sys
import shutil
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ToolDownloader")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

# Primary and fallback mirrors
HF_MAIN = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
HF_MIRROR = "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/"

# Required tools and model weights
REQUIRED_ASSETS = {
    # Hubert
    "assets/hubert/hubert_base.pt": [
        HF_MAIN + "hubert_base.pt",
        HF_MIRROR + "hubert_base.pt",
    ],
    # RMVPE
    "assets/rmvpe/rmvpe.pt": [
        HF_MAIN + "rmvpe.pt",
        HF_MIRROR + "rmvpe.pt",
    ],
    # UVR5 Weights
    "assets/uvr5_weights/HP2_all_vocals.pth": [
        HF_MAIN + "uvr5_weights/HP2_all_vocals.pth",
        HF_MIRROR + "uvr5_weights/HP2_all_vocals.pth",
    ],
    "assets/uvr5_weights/HP5_only_main_vocal.pth": [
        HF_MAIN + "uvr5_weights/HP5_only_main_vocal.pth",
        HF_MIRROR + "uvr5_weights/HP5_only_main_vocal.pth",
    ],
    "assets/uvr5_weights/VR-DeEchoAggressive.pth": [
        HF_MAIN + "uvr5_weights/VR-DeEchoAggressive.pth",
        HF_MIRROR + "uvr5_weights/VR-DeEchoAggressive.pth",
    ],
    "assets/uvr5_weights/VR-DeEchoNormal.pth": [
        HF_MAIN + "uvr5_weights/VR-DeEchoNormal.pth",
        HF_MIRROR + "uvr5_weights/VR-DeEchoNormal.pth",
    ],
    "assets/uvr5_weights/VR-DeEchoDeReverb.pth": [
        HF_MAIN + "uvr5_weights/VR-DeEchoDeReverb.pth",
        HF_MIRROR + "uvr5_weights/VR-DeEchoDeReverb.pth",
    ],
    "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx": [
        HF_MAIN + "uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
        HF_MIRROR + "uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
    ],
}

# Pretrained v1 and v2 models
PRETRAINED_MODELS = [
    "D32k.pth", "D40k.pth", "D48k.pth",
    "G32k.pth", "G40k.pth", "G48k.pth",
    "f0D32k.pth", "f0D40k.pth", "f0D48k.pth",
    "f0G32k.pth", "f0G40k.pth", "f0G48k.pth",
]

for m in PRETRAINED_MODELS:
    REQUIRED_ASSETS[f"assets/pretrained/{m}"] = [
        HF_MAIN + f"pretrained/{m}",
        HF_MIRROR + f"pretrained/{m}",
    ]
    REQUIRED_ASSETS[f"assets/pretrained_v2/{m}"] = [
        HF_MAIN + f"pretrained_v2/{m}",
        HF_MIRROR + f"pretrained_v2/{m}",
    ]

# DiffSinger Assets & Vocoders
REQUIRED_ASSETS["DiffSinger_Training/checkpoints/nsf_hifigan/model"] = [
    "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-44.1k-hop512-128bin-2024.02/model.ckpt",
]
REQUIRED_ASSETS["DiffSinger_Training/checkpoints/rmvpe/model.pt"] = [
    "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.pt",
    HF_MAIN + "rmvpe.pt",
]
REQUIRED_ASSETS["DiffSinger_Training/checkpoints/0917_continuous256_clean_3spk/model_ckpt_steps_72000_simplified.ckpt"] = [
    "https://github.com/openvpi/DiffSinger/releases/download/v2.1.0/0917_continuous256_clean_3spk.zip",
]

# Common alternative search paths on this workstation
LOCAL_SEARCH_DIRECTORIES = [
    Path(r"A:\Thai_RVC_WebUI_v1"),
    Path(r"A:\Thai_RVC_WebUI_v2"),
    Path(r"A:\Fix the File"),
    Path(r"B:\The Thai RVC and Diffsinger WebUI by DELTA SYNTH"),
    Path(r"B:\weights"),
    Path(r"A:\weights"),
    Path(r"B:\B. All Diffsinger RAR file"),
    Path(r"B:\DiffSinger-2.5.1"),
]

def extract_from_archive_if_exists(archive_name: str, rel_target: str, dest: Path) -> bool:
    """Helper to extract a specific asset from local zip archives if present."""
    import zipfile
    for root in LOCAL_SEARCH_DIRECTORIES:
        zip_path = root / archive_name
        if zip_path.is_file():
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for member in z.namelist():
                        if member.endswith(os.path.basename(rel_target)) or member == rel_target:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            with z.open(member) as src, open(dest, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            logger.info(f"📦 แตกไฟล์สำเร็จจาก {zip_path.name}: {dest.name}")
                            return True
            except Exception as e:
                logger.warning(f"⚠️ ไม่สามารถแตกไฟล์จาก {zip_path.name}: {e}")
    return False

def find_local_copy(rel_path: str) -> Path:
    """Check if the asset already exists in local candidate paths on other directories."""
    filename = os.path.basename(rel_path)
    
    # Check if this asset can be resolved from local sibling folders
    if "checkpoints/rmvpe/model.pt" in rel_path:
        rmvpe_rvc = PROJECT_ROOT / "assets/rmvpe/rmvpe.pt"
        if rmvpe_rvc.is_file() and rmvpe_rvc.stat().st_size > 1024:
            return rmvpe_rvc

    for root in LOCAL_SEARCH_DIRECTORIES:
        # Check identical relative path
        candidate = root / rel_path
        if candidate.is_file() and candidate.stat().st_size > 1024:
            return candidate
        # Check direct filename in same subfolder or weights
        alt_candidate = root / filename
        if alt_candidate.is_file() and alt_candidate.stat().st_size > 1024:
            return alt_candidate
    return None

def check_disk_space(target_path: Path, min_mb: int = 200) -> bool:
    """Verify drive has sufficient free space."""
    try:
        drive = os.path.splitdrive(target_path)[0] or str(target_path.anchor)
        free_bytes = shutil.disk_usage(drive).free
        free_mb = free_bytes / (1024 * 1024)
        if free_mb < min_mb:
            logger.warning(f"⚠️ พื้นที่ว่างบนไดรฟ์ {drive} เหลือน้อยมาก: {free_mb:.1f} MB (ต้องการขั้นต่ำ {min_mb} MB)")
            return False
        return True
    except Exception:
        return True

def link_or_copy(src: Path, dst: Path) -> bool:
    """Link via hardlink (0 disk space on same NTFS volume) or copy safely."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Try hardlink first (takes zero additional disk space)
        if hasattr(os, "link"):
            try:
                if dst.exists():
                    dst.unlink()
                os.link(src, dst)
                logger.info(f"🔗 เชื่อมโยง Hardlink สำเร็จ (ใช้พื้นที่ดิสก์ 0 MB): {dst.name} ⬅️ {src}")
                return True
            except (OSError, NotImplementedError):
                pass
        
        # Fallback to copy if on different drives
        shutil.copy2(src, dst)
        logger.info(f"📋 คัดลอกไฟล์สำเร็จ: {dst.name} ⬅️ {src}")
        return True
    except Exception as e:
        logger.error(f"❌ คัดลอก/เชื่อมโยง {dst.name} ล้มเหลว: {e}")
        return False

def download_with_progress(urls: list, dst: Path) -> bool:
    """Download a file with user-agent, streaming progress, and mirror fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_dst = dst.with_suffix(dst.suffix + ".download")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }
    
    for url in urls:
        try:
            logger.info(f"🌐 กำลังดาวน์โหลด: {dst.name} จาก {url}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_dst, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                block_size = 1024 * 1024 # 1 MB chunks
                downloaded = 0
                
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  [ความคืบหน้า] {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
                print()
                
            # Verify file size
            if temp_dst.exists() and temp_dst.stat().st_size > 1024:
                # Check if it's an error page
                if temp_dst.stat().st_size < 50000:
                    with open(temp_dst, "rb") as f:
                        header = f.read(100)
                        if b"<!DOCTYPE html>" in header or b"<html" in header or b"Entry not found" in header:
                            logger.warning(f"⚠️ URL ส่งคืนหน้าเว็บผิดพลาด (404/Not Found): {url}")
                            temp_dst.unlink(missing_ok=True)
                            continue
                
                temp_dst.replace(dst)
                logger.info(f"✅ ดาวน์โหลดไฟล์สำเร็จ: {dst.name}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ การดาวน์โหลดจาก {url} ขัดข้อง: {e}")
            if temp_dst.exists():
                temp_dst.unlink(missing_ok=True)
                
    return False

def audit_and_prepare_all():
    """Audit all necessary tools and models, resolve locally or download."""
    logger.info("=" * 65)
    logger.info("   DELTA SYNTH — Smart Tool & Model Verifier v2.0")
    logger.info("=" * 65)
    
    total = len(REQUIRED_ASSETS)
    available_count = 0
    resolved_locally = 0
    downloaded_count = 0
    failed_count = 0
    
    for rel_path, urls in REQUIRED_ASSETS.items():
        dest = PROJECT_ROOT / rel_path
        
        # 1. Check if already present and valid
        if dest.is_file() and dest.stat().st_size > 1024:
            available_count += 1
            continue
            
        # 2. Check local discovery from sibling folders
        local_src = find_local_copy(rel_path)
        if local_src and local_src != dest:
            if link_or_copy(local_src, dest):
                resolved_locally += 1
                available_count += 1
                continue
                
        # 3. If missing everywhere, check disk space before downloading
        if not check_disk_space(dest, min_mb=100):
            logger.error(f"❌ ข้ามการดาวน์โหลด {dest.name} เนื่องจากพื้นที่ดิสก์ไม่เพียงพอ")
            failed_count += 1
            continue
            
        # 4. Download from online mirrors
        if download_with_progress(urls, dest):
            downloaded_count += 1
            available_count += 1
        else:
            logger.error(f"❌ ไม่สามารถดาวน์โหลดหรือค้นหา {rel_path} ได้")
            failed_count += 1
            
    logger.info("=" * 65)
    logger.info(f"สรุปผลการตรวจสอบเครื่องมือและโมเดล:")
    logger.info(f"  - พร้อมใช้งานทั้งหมด: {available_count}/{total}")
    logger.info(f"  - ตรวจพบและเชื่อมโยงจากในเครื่อง: {resolved_locally}")
    logger.info(f"  - ดาวน์โหลดใหม่สำเร็จ: {downloaded_count}")
    logger.info(f"  - ขาดหาย/ไม่สามารถดาวน์โหลดได้: {failed_count}")
    logger.info("=" * 65)
    
    return failed_count == 0

if __name__ == "__main__":
    success = audit_and_prepare_all()
    sys.exit(0 if success else 1)


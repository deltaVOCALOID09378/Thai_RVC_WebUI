# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : train_index.py
# เวอร์ชั่นโค้ด (Version)      : 2.8 (MiniBatchKMeans Compression & Chunked FAISS Indexing)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/train_index.py
# หน้าที่ (Function/Role)     : สร้างดัชนีค้นหาคุณลักษณะเสียง (FAISS Index)
#                             รองรับการย่อขนาดด้วย MiniBatchKMeans เมื่อข้อมูลใหญ่เกิน 200,000 เวกเตอร์,
#                             การเพิ่มข้อมูลทีละ Batch (8,192) ป้องกัน RAM เต็ม และการคัดลอกไปที่ assets/indices
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys
import glob
import shutil
import logging
import numpy as np
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TrainIndex")


def build_faiss_index(exp_dir: str, version: str = "v2") -> tuple:
    try:
        import faiss
    except ImportError:
        return False, "❌ ไม่พบโมดูล faiss กรุณาติดตั้ง faiss-cpu ก่อนดำเนินการ (Module faiss not found)", ""

    exp_dir = os.path.abspath(exp_dir).replace("\\", "/")
    model_name = os.path.basename(exp_dir.rstrip("/"))

    feat_dir_768 = os.path.join(exp_dir, "3_feature768").replace("\\", "/")
    feat_dir_256 = os.path.join(exp_dir, "3_feature256").replace("\\", "/")

    if os.path.isdir(feat_dir_768) and len(glob.glob(os.path.join(feat_dir_768, "*.npy"))) > 0:
        resolved_ver = "v2"
        feat_dir = feat_dir_768
        feat_dir_name = "3_feature768"
    elif os.path.isdir(feat_dir_256) and len(glob.glob(os.path.join(feat_dir_256, "*.npy"))) > 0:
        resolved_ver = "v1"
        feat_dir = feat_dir_256
        feat_dir_name = "3_feature256"
    else:
        resolved_ver = version.lower()
        feat_dir_name = "3_feature256" if resolved_ver == "v1" else "3_feature768"
        feat_dir = os.path.join(exp_dir, feat_dir_name).replace("\\", "/")

    if not os.path.exists(feat_dir):
        return False, f"❌ ไม่พบโฟลเดอร์ฟีเจอร์ (Feature folder not found): {feat_dir}", ""

    npy_files = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
    if not npy_files:
        return False, f"❌ ไม่พบไฟล์ .npy ใน (No .npy files found in): {feat_dir}", ""

    logger.info(f"🔄 เริ่มรวบรวมฟีเจอร์จาก {len(npy_files)} ไฟล์ใน {feat_dir_name} (Collecting features)...")

    feature_list = []
    for f in npy_files:
        try:
            if os.path.getsize(f) <= 64:
                continue
            arr = np.load(f, allow_pickle=False)
            if arr.ndim == 2:
                feature_list.append(arr)
            elif arr.ndim == 3:
                feature_list.append(arr.reshape(-1, arr.shape[-1]))
        except Exception as e:
            logger.warning(f"⚠️ ข้ามไฟล์ที่เสียหาย (Skipped corrupted file): {f} ({e})")

    if not feature_list:
        return False, "❌ ไม่สามารถโหลดข้อมูลฟีเจอร์จากไฟล์ใดๆ ได้เลย (Failed to load any feature files)", ""

    big_features = np.concatenate(feature_list, axis=0).astype(np.float32)

    if np.isnan(big_features).any() or np.isinf(big_features).any():
        logger.warning("🧹 ตรวจพบค่า NaN/Inf ในฟีเจอร์ - กำลังทำความสะอาดข้อมูล (Cleaning NaNs)...")
        big_features = np.nan_to_num(big_features, nan=0.0, posinf=0.0, neginf=0.0)

    # สุ่มลำดับเวกเตอร์เพื่อให้การฝึกสอนคลัสเตอร์กระจายตัวอย่างสม่ำเสมอ
    shuffle_idx = np.arange(big_features.shape[0])
    np.random.shuffle(shuffle_idx)
    big_features = big_features[shuffle_idx]

    # หากจำนวนเวกเตอร์มากกว่า 200,000 แถว ให้ย่อด้วย MiniBatchKMeans เหลือ 10,000 Centroids
    if big_features.shape[0] > 200000:
        logger.info(f"⚡ ชุดข้อมูลมีขนาดใหญ่ ({big_features.shape[0]} vectors) กำลังย่อด้วย MiniBatchKMeans เหลือ 10,000 centers...")
        try:
            cpu_cores = os.cpu_count() or 4
            big_features = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=False,
                    batch_size=256 * cpu_cores,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_features)
                .cluster_centers_
                .astype(np.float32)
            )
        except Exception as km_err:
            logger.warning(f"⚠️ MiniBatchKMeans ขัดข้อง ใช้เวกเตอร์เต็มแทน: {km_err}")

    n_samples, d = big_features.shape
    logger.info(f"📊 ขนาดเมทริกซ์ฟีเจอร์สุทธิ: {n_samples} เวกเตอร์ (มิติ: {d} / Dimension: {d})")

    total_fea_path = os.path.join(exp_dir, "total_fea.npy").replace("\\", "/")
    np.save(total_fea_path, big_features, allow_pickle=False)

    n_ivf = max(1, min(int(16 * np.sqrt(n_samples)), max(1, n_samples // 39)))
    index_filename = f"added_IVF{n_ivf}_Flat_nprobe_1_{model_name}_{resolved_ver}.index"
    index_path = os.path.join(exp_dir, index_filename).replace("\\", "/")

    if n_samples < 256 or n_ivf <= 1:
        logger.info(f"⚡ จำนวนเวกเตอร์น้อย ({n_samples}) - สลับใช้ IndexFlatL2 อัตโนมัติ...")
        index = faiss.IndexFlatL2(d)
        index.add(big_features)
    else:
        logger.info(f"⚙️ สร้าง IndexIVFFlat ด้วยจำนวนคลัสเตอร์ (n_ivf): {n_ivf}")
        index = faiss.index_factory(d, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_features)

        trained_filename = f"trained_IVF{n_ivf}_Flat_nprobe_1_{model_name}_{resolved_ver}.index"
        faiss.write_index(index, os.path.join(exp_dir, trained_filename).replace("\\", "/"))

        # เพิ่มข้อมูลทีละ Batch (8,192) เพื่อป้องกันปัญหาหน่วยความจำ RAM เต็ม
        batch_size_add = 8192
        for i in range(0, n_samples, batch_size_add):
            index.add(big_features[i : i + batch_size_add])

    faiss.write_index(index, index_path)

    # คัดลอกไฟล์ไปที่ assets/indices เพื่อให้แสดงใน Dropdown ของ Program.py ทันที
    try:
        project_root = os.path.abspath(os.path.join(exp_dir, "..", ".."))
        assets_indices = os.path.join(project_root, "assets", "indices")
        os.makedirs(assets_indices, exist_ok=True)
        shutil.copy2(index_path, os.path.join(assets_indices, index_filename))
    except Exception as e:
        logger.warning(f"Could not copy index to assets/indices: {e}")

    success_msg = (
        f"-----------------------------------------------------------------\n"
        f"✅ สร้างไฟล์ Index สำเร็จสมบูรณ์! (Index Generation Complete!)\n"
        f"  - 📄 ไฟล์ Index: {index_path}\n"
        f"  - 🔢 จำนวนเวกเตอร์ที่บันทึก: {index.ntotal} เวกเตอร์ (Vectors)\n"
        f"  - 🌐 มิติเวกเตอร์: {d} Channels ({resolved_ver.upper()})\n"
        f"-----------------------------------------------------------------"
    )
    logger.info(success_msg)
    return True, success_msg, index_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        ver = sys.argv[2] if len(sys.argv) > 2 else "v2"
        ok, msg, path = build_faiss_index(target_dir, version=ver)
        print(msg, flush=True)
        sys.exit(0 if ok else 1)
    else:
        print("Usage: python train_index.py <path_to_logs_exp_dir> [v1|v2]")
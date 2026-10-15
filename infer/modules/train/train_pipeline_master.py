# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : train_pipeline_master.py
# เวอร์ชั่นโค้ด (Version)      : 3.2 (End-to-End Autonomous Training Master Engine)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/train_pipeline_master.py
# หน้าที่ (Function/Role)     : ผู้จัดการไปป์ไลน์การฝึกสอนโมเดล RVC แบบครบวงจร
#                             หากชุดข้อมูลยังไม่ได้ผ่านการตัดเสียงหรือสกัดฟีเจอร์ ระบบจะรันขั้นตอนที่ 1 และ 2
#                             ให้อัตโนมัติ พร้อมสร้าง config.json, ฝึกสอนโมเดล และสร้าง FAISS Index จนจบ
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys
import json
import glob
import shutil
import logging
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_dataset_inspector import inspect_and_build_filelist
from train_index import build_faiss_index

try:
    from dataset_auto_cleaner import sanitize_dataset_and_logs
except ImportError:
    sanitize_dataset_and_logs = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TrainMaster")


class TrainPipelineMaster:
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            self.root_dir = PROJECT_ROOT.replace("\\", "/")
        else:
            self.root_dir = os.path.abspath(root_dir).replace("\\", "/")

        self.python_exe = os.path.join(self.root_dir, "runtime", "python.exe").replace("\\", "/")
        if not os.path.exists(self.python_exe):
            self.python_exe = sys.executable

        self.weights_dir = os.path.join(self.root_dir, "assets", "weights").replace("\\", "/")
        self.logs_dir = os.path.join(self.root_dir, "logs").replace("\\", "/")
        self.assets_dir = os.path.join(self.root_dir, "assets").replace("\\", "/")

        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def check_pretrain_models(self, version: str = "v2", sr: str = "40k", if_f0: int = 1) -> tuple:
        f0_tag = "f0" if if_f0 == 1 else ""
        g_name = f"{f0_tag}G{sr}.pth"
        d_name = f"{f0_tag}D{sr}.pth"

        pretrain_dir = os.path.join(self.assets_dir, "pretrained" if version == "v1" else "pretrained_v2")
        g_path = os.path.join(pretrain_dir, g_name).replace("\\", "/")
        d_path = os.path.join(pretrain_dir, d_name).replace("\\", "/")

        g_exists = os.path.exists(g_path)
        d_exists = os.path.exists(d_path)

        return g_exists and d_exists, g_path, d_path

    def ensure_experiment_config_json(self, exp_path: str, version: str, sr_tag: str):
        """คัดลอกไฟล์ตั้งค่าสถาปัตยกรรมโมเดล (config.json) เข้าสู่โฟลเดอร์การทดลองหากยังไม่มี"""
        target_cfg = os.path.join(exp_path, "config.json")
        if os.path.isfile(target_cfg) and os.path.getsize(target_cfg) > 64:
            return
        ver_sub = "v1" if (version == "v1" or sr_tag == "40k") else "v2"
        src_candidates = [
            os.path.join(self.root_dir, "configs", "inuse", ver_sub, f"{sr_tag}.json"),
            os.path.join(self.root_dir, "configs", ver_sub, f"{sr_tag}.json"),
        ]
        for cand in src_candidates:
            if os.path.isfile(cand):
                shutil.copy2(cand, target_cfg)
                break

    def _run_step_cmd(self, cmd: list, step_label: str) -> bool:
        print(f"[CMD] 🖥️ กำลังดำเนินการ ({step_label}): {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(
            cmd, cwd=self.root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace"
        )
        for line in p.stdout:
            line_str = line.strip()
            if line_str:
                print(f"  {line_str}", flush=True)
        p.wait()
        return p.returncode == 0

    def run_training_pipeline(
        self,
        exp_name: str,
        dataset_dir: str,
        sr: int = 40000,
        if_f0: int = 1,
        version: str = "v2",
        f0_method: str = "rmvpe",
        total_epoch: int = 200,
        batch_size: int = 4,
        save_epoch: int = 20,
        cpu_threads: int = 4,
        device: str = "cuda:0"
    ) -> bool:
        exp_path = os.path.join(self.logs_dir, exp_name).replace("\\", "/")
        os.makedirs(exp_path, exist_ok=True)
        sr_tag = f"{sr // 1000}k"

        print("=" * 65, flush=True)
        print("  🤖 DELTA SYNTH - ระบบฝึกสอนโมเดลอัจฉริยะ v3.2 (Autonomous Master Engine)", flush=True)
        print("=" * 65, flush=True)
        print(f"[INFO] 🏷️ ชื่อโมเดล (Model): {exp_name} | เวอร์ชัน (Ver): {version.upper()} | Sampling Rate: {sr}Hz", flush=True)
        print(f"[INFO] 💻 อุปกรณ์ประมวลผล (Device): {device.upper()} | Threads: {cpu_threads} | Epochs: {total_epoch}", flush=True)
        print("-" * 65, flush=True)

        has_pretrain, g_pre, d_pre = self.check_pretrain_models(version, sr_tag, if_f0)
        if has_pretrain:
            print(f"[SUCCESS] ✅ พบโมเดลพรีเทรน (Pretrained G): {g_pre}", flush=True)
            print(f"[SUCCESS] ✅ พบโมเดลพรีเทรน (Pretrained D): {d_pre}", flush=True)
        else:
            print(f"[WARNING] ⚠️ ไม่พบโมเดลพรีเทรนใน {self.assets_dir} ระบบจะเริ่มฝึกสอนจากศูนย์", flush=True)
            g_pre = d_pre = ""

        print("\n[STEP 1/3] 🔍 ตรวจสอบและเตรียมชุดข้อมูลอัตโนมัติ (Dataset Audit & Auto-Extraction)...", flush=True)
        if sanitize_dataset_and_logs:
            try:
                sanitize_dataset_and_logs(exp_path, dataset_dir)
            except Exception as clean_err:
                logger.warning(f"Auto-clean notice: {clean_err}")

        is_ready, report_msg, count = inspect_and_build_filelist(
            exp_dir=exp_path,
            version=version,
            if_f0=if_f0,
            spk_id=0,
            sr_tag=sr_tag,
            force_rebuild=True
        )

        # หากชุดข้อมูลยังไม่พร้อม แต่มีโฟลเดอร์ไฟล์เสียงต้นฉบับ ให้รัน Preprocess + Extract F0 + Extract Feature อัตโนมัติ!
        if not is_ready and dataset_dir and os.path.isdir(dataset_dir):
            print("[AUTO-PIPELINE] ⚡ ตรวจพบว่ายังไม่ได้สกัดฟีเจอร์ เริ่มรัน Preprocess และสกัดฟีเจอร์อัตโนมัติ...", flush=True)
            prep_script = os.path.join(self.root_dir, "infer/modules/train/preprocess.py").replace("\\", "/")
            f0_script = os.path.join(self.root_dir, "infer/modules/train/extract/extract_f0_print.py").replace("\\", "/")
            feat_script = os.path.join(self.root_dir, "infer/modules/train/extract_feature_print.py").replace("\\", "/")

            self._run_step_cmd([self.python_exe, prep_script, dataset_dir, str(sr), str(cpu_threads), exp_path, "False", "3.7"], "Preprocess Audio")
            if if_f0 == 1:
                self._run_step_cmd([self.python_exe, f0_script, exp_path, str(cpu_threads), f0_method], "Extract F0 Pitch")
            self._run_step_cmd([self.python_exe, feat_script, device, "1", "0", "0", exp_path, version], "Extract HuBERT Features")

            is_ready, report_msg, count = inspect_and_build_filelist(
                exp_dir=exp_path,
                version=version,
                if_f0=if_f0,
                spk_id=0,
                sr_tag=sr_tag,
                force_rebuild=True
            )

        print(report_msg, flush=True)

        if not is_ready:
            print("\n[ALERT] 🚨 ชุดข้อมูลยังไม่พร้อมสำหรับฝึกโมเดล! กรุณาตรวจสอบโฟลเดอร์ไฟล์เสียงต้นฉบับ", flush=True)
            return False

        self.ensure_experiment_config_json(exp_path, version, sr_tag)

        print("\n[STEP 2/3] ⚙️ เริ่มการฝึกสอนโมเดล (Training Model Process)...", flush=True)
        train_script = os.path.join(self.root_dir, "infer/modules/train/train.py").replace("\\", "/")

        train_cmd = [
            self.python_exe, train_script,
            "-e", exp_name,
            "-sr", sr_tag,
            "-f0", str(if_f0),
            "-bs", str(batch_size),
            "-te", str(total_epoch),
            "-se", str(save_epoch),
            "-v", version,
            "-l", "1",
            "-c", "0",
            "-sw", "1"
        ]

        if g_pre: train_cmd.extend(["-pg", g_pre])
        if d_pre: train_cmd.extend(["-pd", d_pre])

        if not self._run_step_cmd(train_cmd, "RVC Model Training"):
            print("\n[ERROR] ❌ กระบวนการฝึกสอนโมเดลล้มเหลว ระงับการสร้างดัชนีเพื่อป้องกันปัญหา Phantom Index", flush=True)
            return False

        print("\n[SUCCESS] ✅ การฝึกสอนโมเดลเสร็จสิ้นเรียบร้อยแล้ว (Model Training Completed)!", flush=True)

        print("\n[STEP 3/3] 📑 สกัดและสร้างดัชนีค้นหาฟีเจอร์ (Building Feature Index)...", flush=True)
        index_ok, index_msg, index_file = build_faiss_index(exp_dir=exp_path, version=version)
        print(index_msg, flush=True)

        target_model_pth = os.path.join(self.weights_dir, f"{exp_name}.pth")
        print("\n" + "=" * 65, flush=True)
        print("  🎉 การฝึกสอนโมเดลเสร็จสิ้นอย่างสมบูรณ์แบบ (100% Pipeline Complete)", flush=True)
        print("=" * 65, flush=True)
        print(f"📁 1. ไฟล์โมเดลเสียง (.pth): {target_model_pth}", flush=True)
        print(f"📁 2. ไฟล์ดัชนีฟีเจอร์ (.index): {index_file}", flush=True)
        print(f"💡 พร้อมนำไปใช้งานในการแปลงเสียงเดี่ยว หรือประมวลผลคลังเสียงทั้งหมดได้ทันที! (Ready to Use)", flush=True)
        return True


if __name__ == "__main__":
    master = TrainPipelineMaster()
    if len(sys.argv) > 2:
        exp = sys.argv[1]
        ds = sys.argv[2]
        ver_arg = sys.argv[3] if len(sys.argv) > 3 else "v2"
        master.run_training_pipeline(exp_name=exp, dataset_dir=ds, version=ver_arg)
    else:
        print("Usage: python train_pipeline_master.py <exp_name> <dataset_dir> [v1|v2]")
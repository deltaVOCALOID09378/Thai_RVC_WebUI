# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)  : infer-web.py
# 3. ลำดับเวอร์ชัน (Version)   : 2.0 (Unified Studio-Grade UI & Memory Safe Edition)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              [ Original by Patiphat Wongyai (Delta) & RVC Project ]
# 5. ตำแหน่งที่วางของไฟล์ (Path): A:\Thai_RVC_WebUI_v1\infer-web.py
# 6. รายละเอียดการแก้ไขงาน    : หลอมรวมฟีเจอร์ Native File Dialog (Tkinter), ระบบ Safe Cancel (Zero-Leak), 
#                              และปรับปรุง UI สไตล์ DELTA SYNTH Corporate (Red/Dark Theme)
# ==============================================================================

import os
import sys
from dotenv import load_dotenv

# กำหนดเส้นทางปัจจุบันและโหลดตัวแปรสภาพแวดล้อม
now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)
load_dotenv()

# บังคับใช้ UTF-8 บน Windows Console ป้องกันปัญหาแสดงผล
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import I18nAuto
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
import torch, platform
import numpy as np
import gradio as gr
import faiss
import fairseq
import pathlib
import json
from time import sleep, strftime
import time
from subprocess import Popen
import subprocess
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging
import contextlib
import gc

# ---------------------------------------------------------
# การจัดการระบบและทำความสะอาด (System Management & Clean-up)
# ---------------------------------------------------------
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

ACTIVE_CHILD_PROCESSES = set()
CANCEL_REQUESTED = False

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(os.path.join(now_dir, "runtime", "Lib", "site-packages", "infer_pack"), ignore_errors=True)
shutil.rmtree(os.path.join(now_dir, "runtime", "Lib", "site-packages", "uvr5_pack"), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "weights"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "indices"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

config = Config()
vc = VC(config)

if getattr(config, "dml", False) == True:
    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res
    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

i18n = I18nAuto()

# ---------------------------------------------------------
# การบริหารจัดการฮาร์ดแวร์ (Hardware Management System)
# ---------------------------------------------------------
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if_gpu_ok = True
        gpu_infos.append(f"GPU {i}\t{gpu_name}")
        mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))
        
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "ตรวจพบกราฟิกการ์ด (GPU Detected):\n" + "\n".join(gpu_infos)
    default_batch_size = max(1, min(mem) // 2)
elif getattr(config, "dml", False) == True:
    gpu_info = "ทำงานด้วยระบบ DirectML สำหรับ DirectX (Operating via DirectML for DirectX)"
    default_batch_size = 1
else:
    gpu_info = "ทำงานด้วยระบบหน่วยประมวลผลกลาง / อุปกรณ์ภายนอก (Operating via CPU / External USB Unit)"
    default_batch_size = 1

gpus = "-".join([i[0] for i in gpu_infos]) if if_gpu_ok else ""
logger.info(gpu_info)

# ---------------------------------------------------------
# ระบบเลือกไฟล์ผ่าน Native Dialog (Tkinter Topmost Isolation)
# ---------------------------------------------------------
def browse_folder(current_val=""):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        clean_p = str(current_val or "").strip().strip('"')
        init_dir = clean_p if (clean_p and os.path.isdir(clean_p)) else now_dir
        selected = filedialog.askdirectory(initialdir=init_dir, title="เลือกโฟลเดอร์ (Select Folder)")
        root.destroy()
        if selected: return os.path.normpath(selected).replace("\\", "/")
    except Exception as e:
        logger.warning(f"Browse folder notice: {e}")
    return current_val

def browse_file(current_val="", filetypes=None):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        clean_p = str(current_val or "").strip().strip('"')
        init_dir = os.path.dirname(clean_p) if (clean_p and os.path.isfile(clean_p)) else (clean_p if clean_p and os.path.isdir(clean_p) else now_dir)
        selected = filedialog.askopenfilename(initialdir=init_dir, title="เลือกไฟล์ (Select File)", filetypes=filetypes or [("All files", "*.*")])
        root.destroy()
        if selected: return os.path.normpath(selected).replace("\\", "/")
    except Exception as e:
        logger.warning(f"Browse file notice: {e}")
    return current_val

# ---------------------------------------------------------
# ระบบบริหารจัดการโมเดลและดัชนี (Model & Index Management)
# ---------------------------------------------------------
weight_root = os.getenv("weight_root", "assets/weights")
weight_uvr5_root = os.getenv("weight_uvr5_root", "assets/uvr5_weights")
index_root = os.getenv("index_root", "logs")
SKIP_INDEX_DIRS = {"0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature256", "3_feature768", "mute", "__pycache__", ".git"}

def get_model_names(root=weight_root):
    if not os.path.isdir(root): return []
    return sorted(name for name in os.listdir(root) if name.lower().endswith((".pth", ".pt")))

def get_index_files(root=index_root):
    search_dirs = [root]
    indices_dir = os.path.join(now_dir, "assets", "indices")
    if os.path.isdir(indices_dir) and indices_dir != root: search_dirs.append(indices_dir)
    results = set()
    for s_dir in search_dirs:
        if not os.path.isdir(s_dir): continue
        for cur_root, dirs, files in os.walk(s_dir, topdown=True):
            dirs[:] = [d for d in dirs if d not in SKIP_INDEX_DIRS]
            for file in files:
                if file.lower().endswith(".index") and "trained" not in file.lower():
                    results.add(os.path.normpath(os.path.join(cur_root, file)).replace("\\", "/"))
    return sorted(list(results))

names = get_model_names()
index_paths = get_index_files()
uvr5_names = [name.replace(".pth", "") for name in os.listdir(weight_uvr5_root) if name.endswith(".pth") or "onnx" in name] if os.path.exists(weight_uvr5_root) else []

def change_choices():
    return {"choices": sorted(get_model_names()), "__type__": "update"}, {"choices": sorted(get_index_files()), "__type__": "update"}

# ---------------------------------------------------------
# ระบบยกเลิกงานและจัดการหน่วยความจำ VRAM (Safe Cancel & Zero-Leak)
# ---------------------------------------------------------
def clear_vram_safe():
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with contextlib.suppress(Exception): torch.cuda.ipc_collect()
    except Exception as e:
        logger.warning(f"VRAM clear warning: {e}")

def safe_cancel_process():
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True
    if hasattr(vc, "cancel_render"): vc.cancel_render()
    elif hasattr(vc, "cancel_process"):
        try: vc.cancel_process()
        except Exception: pass
    if hasattr(vc, "cancel_flag"): vc.cancel_flag = True
    clear_vram_safe()
    sep_line = "=" * 52
    return f"{sep_line}\n⚠️ [TH] คำสั่งถูกยกเลิกโดยผู้ใช้งานเรียบร้อยแล้ว / [EN] Render Cancelled by User\n🏁 เวลาสิ้นสุด (End Time): {strftime('%H:%M:%S')}\n{sep_line}"

def clean(): 
    clear_vram_safe()
    return {"value": "", "__type__": "update"}

def export_onnx(ModelPath, ExportedPath):
    from infer.modules.onnx.export import export_onnx as eo
    return eo(ModelPath, ExportedPath)

# ---------------------------------------------------------
# Core Processing Wrappers
# ---------------------------------------------------------
def single_infer_wrapper(spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 52

    clean_input = str(input_audio0 or "").strip().strip('"')
    if not clean_input or not os.path.exists(clean_input):
        err_msg = (
            f"{sep_line}\n"
            f"❌ [TH] ไม่พบไฟล์เสียงต้นฉบับที่ระบุ / [EN] Source Audio Not Found: {clean_input}\n"
            f"⏰ เวลาเริ่มเรนเดอร์: {start_timestamp} | เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')}\n"
            f"{sep_line}"
        )
        return err_msg, None

    info_log, opt_audio = vc.vc_single(
        spk_item, clean_input, vc_transform0, f0_file, f0method0,
        file_index1, file_index2, index_rate1, filter_radius0,
        resample_sr0, rms_mix_rate0, protect0
    )

    elapsed_sec = int(time.time() - start_time_sec)
    mins, secs = divmod(elapsed_sec, 60)
    end_timestamp = strftime("%H:%M:%S")
    fname = os.path.basename(clean_input)

    final_log = (
        f"{sep_line}\n"
        f"🎙️ ลำดับคิว: 1/1 | ชื่อไฟล์: {fname} | เวลาเริ่ม: {start_timestamp} | เวลาเสร็จ: {end_timestamp} (⏱️ {mins:02d}:{secs:02d})\n"
        f"สถานะ: {info_log}\n"
        f"{sep_line}"
    )
    clear_vram_safe()
    return final_log, opt_audio

def batch_render_tracker(sid, dir_input, opt_input, inputs, vc_transform, f0method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1):
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = False
    start_time_sec = time.time()
    start_timestamp = strftime("%H:%M:%S")
    sep_line = "=" * 52
    format1 = str(format1 or "wav").lower().strip()
    dir_input_clean = str(dir_input or "").strip().strip('"')
    opt_input_clean = str(opt_input or "").strip().strip('"')

    if not os.path.exists(dir_input_clean) and not inputs:
        yield f"{sep_line}\n❌ [TH] ไม่พบโฟลเดอร์หรือไฟล์ต้นฉบับ / [EN] Input Not Found\n{sep_line}"
        return

    os.makedirs(opt_input_clean, exist_ok=True)
    if os.path.exists(dir_input_clean):
        oto_src = os.path.join(dir_input_clean, "oto.ini")
        oto_dst = os.path.join(opt_input_clean, "oto.ini")
        if os.path.exists(oto_src) and not os.path.exists(oto_dst):
            shutil.copy(oto_src, oto_dst)

    valid_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    source_files = []
    if inputs:
        source_files = [f.name if hasattr(f, "name") else str(f) for f in inputs]
    elif os.path.exists(dir_input_clean):
        source_files = [os.path.join(dir_input_clean, f) for f in os.listdir(dir_input_clean) if os.path.splitext(f)[1].lower() in valid_exts]

    total_count = len(source_files)
    files_to_render = []
    already_done = 0

    for src_path in source_files:
        fname = os.path.basename(src_path)
        # Custom clean_stem logic
        for ext in [".wav.npy", ".flac.npy", ".npy", ".wav", ".flac", ".mp3", ".ogg", ".m4a"]:
            if fname.lower().endswith(ext):
                fname = fname[:-len(ext)]
                break
        expected_dst = os.path.join(opt_input_clean, f"{fname}.{format1}")
        if os.path.exists(expected_dst) and os.path.getsize(expected_dst) > 1024:
            already_done += 1
        else:
            files_to_render.append(src_path)

    pending_count = len(files_to_render)
    yield (
        f"{sep_line}\n"
        f"⏳ เริ่มกระบวนการตรวจสอบการสานต่อไฟล์ (Smart Batch Resume Initializing)...\n"
        f"📊 ไฟล์ทั้งหมด: {total_count} | สำเร็จแล้วเดิม (Skip): {already_done} | ค้างทำต่อ (Pending): {pending_count}\n"
        f"⏰ เวลาเริ่มต้น: {start_timestamp}\n"
        f"{sep_line}"
    )

    if pending_count == 0:
        mins, secs = divmod(int(time.time() - start_time_sec), 60)
        yield f"{sep_line}\n🎉 [TH] เรนเดอร์เสร็จสมบูรณ์แล้วทั้งหมด 100% / [EN] All Files Already Processed (⏱️ {mins:02d}:{secs:02d})\n{sep_line}"
        clear_vram_safe()
        return

    selected_index = file_index.strip() if file_index else (file_index2 or "")
    generator = vc.vc_multi(
        sid=sid, dir_path=dir_input_clean if os.path.exists(dir_input_clean) else "",
        opt_root=opt_input_clean, paths=files_to_render, f0_up_key=vc_transform,
        f0_method=f0method, file_index=selected_index, file_index2="",
        index_rate=index_rate, filter_radius=filter_radius, resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate, protect=protect, format1=format1
    )

    done_in_loop = 0
    for log in generator:
        if CANCEL_REQUESTED:
            yield f"{sep_line}\n🛑 [TH] คำสั่งถูกยกเลิกโดยผู้ใช้งาน / [EN] Cancelled\n{sep_line}"
            clear_vram_safe()
            return
        done_in_loop += 1
        total_finished = already_done + done_in_loop
        pct = (total_finished / total_count * 100) if total_count > 0 else 100
        yield f"🟢 กำลังประมวลผล (Processing): {total_finished}/{total_count} ({pct:.1f}%)\n{log}"

    mins, secs = divmod(int(time.time() - start_time_sec), 60)
    yield f"{sep_line}\n🟢 [TH] การเรนเดอร์แบบกลุ่มเสร็จสมบูรณ์เรียบร้อยแล้ว {total_count} ไฟล์ / [EN] Batch completed (⏱️ {mins:02d}:{secs:02d})\n{sep_line}"
    clear_vram_safe()

# ---------------------------------------------------------
# Training Wrappers
# ---------------------------------------------------------
sr_dict = {"32k": 32000, "40k": 40000, "48k": 48000}
def if_done(done, p):
    while p.poll() is None: sleep(0.5)
    done[0] = True
def if_done_multi(done, ps):
    while True:
        if all(p.poll() is not None for p in ps): break
        sleep(0.5)
    done[0] = True

def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr_val = sr_dict.get(str(sr).lower().strip(), 40000)
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "preprocess.log")
    with open(log_file, "w", encoding="utf-8") as f: f.write("")
    
    per = 3.0 if config.is_half else 3.7
    cmd = [config.python_cmd, "infer/modules/train/preprocess.py", str(trainset_dir), str(sr_val), str(n_p), str(exp_path), str(config.noparallel), str(per)]
    logger.info(" ".join(cmd))
    p = Popen(cmd, cwd=now_dir)
    ACTIVE_CHILD_PROCESSES.add(p)
    done = [False]
    threading.Thread(target=if_done, args=(done, p,)).start()
    
    while not done[0]:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
        sleep(1)
    ACTIVE_CHILD_PROCESSES.discard(p)
    with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()

def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus_list = str(gpus).split("-") if gpus else ["0"]
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    log_file = os.path.join(exp_path, "extract_f0_feature.log")
    with open(log_file, "w", encoding="utf-8") as f: f.write("")

    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = [config.python_cmd, "infer/modules/train/extract/extract_f0_print.py", exp_path, str(n_p), str(f0method)]
            p = Popen(cmd, cwd=now_dir)
            ACTIVE_CHILD_PROCESSES.add(p)
            done = [False]
            threading.Thread(target=if_done, args=(done, p,)).start()
        else:
            if gpus_rmvpe != "-" and not getattr(config, "dml", False):
                gpus_rmvpe_list = str(gpus_rmvpe).split("-")
                leng = len(gpus_rmvpe_list)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe_list):
                    cmd = [config.python_cmd, "infer/modules/train/extract/extract_f0_rmvpe.py", str(leng), str(idx), str(n_g), exp_path, str(getattr(config, "is_half", False))]
                    p = Popen(cmd, cwd=now_dir)
                    ACTIVE_CHILD_PROCESSES.add(p)
                    ps.append(p)
                done = [False]
                threading.Thread(target=if_done_multi, args=(done, ps,)).start()
            else:
                cmd = [config.python_cmd, "infer/modules/train/extract/extract_f0_rmvpe_dml.py", exp_path]
                p = Popen(cmd, cwd=now_dir)
                ACTIVE_CHILD_PROCESSES.add(p)
                p.wait()
                ACTIVE_CHILD_PROCESSES.discard(p)
                done = [True]
                
        while not done[0]:
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
            sleep(1)

    leng = len(gpus_list)
    ps = []
    for idx, n_g in enumerate(gpus_list):
        cmd = [config.python_cmd, "infer/modules/train/extract_feature_print.py", str(getattr(config, "device", "cpu")), str(leng), str(idx), str(n_g), exp_path, str(version19)]
        p = Popen(cmd, cwd=now_dir)
        ACTIVE_CHILD_PROCESSES.add(p)
        ps.append(p)
    done = [False]
    threading.Thread(target=if_done_multi, args=(done, ps,)).start()
    
    while not done[0]:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()
        sleep(1)
        
    for p in ps: ACTIVE_CHILD_PROCESSES.discard(p)
    with open(log_file, "r", encoding="utf-8", errors="replace") as f: yield f.read()

def click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19):
    exp_dir = os.path.join(now_dir, "logs", str(exp_dir1))
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
    feature_dir = os.path.join(exp_dir, "3_feature256") if version19 == "v1" else os.path.join(exp_dir, "3_feature768")
    
    if if_f0_3:
        f0_dir = os.path.join(exp_dir, "2a_f0")
        f0nsf_dir = os.path.join(exp_dir, "2b-f0nsf")
        names = (set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set([name.split(".")[0] for name in os.listdir(feature_dir)]) & set([name.split(".")[0] for name in os.listdir(f0_dir)]) & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)]))
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set([name.split(".")[0] for name in os.listdir(feature_dir)])
        
    try: safe_spk_id = int(spk_id5)
    except (ValueError, TypeError): safe_spk_id = 0
        
    opt = []
    for name in names:
        if if_f0_3: opt.append("%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s" % (gt_wavs_dir.replace("\\", "/"), name, feature_dir.replace("\\", "/"), name, f0_dir.replace("\\", "/"), name, f0nsf_dir.replace("\\", "/"), name, safe_spk_id))
        else: opt.append("%s/%s.wav|%s/%s.npy|%s" % (gt_wavs_dir.replace("\\", "/"), name, feature_dir.replace("\\", "/"), name, safe_spk_id))
        
    fea_dim = 256 if version19 == "v1" else 768
    mute_dir = os.path.join(now_dir, "logs", "mute")
    if os.path.isdir(mute_dir):
        if if_f0_3:
            for _ in range(2): opt.append("%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|%s/2a_f0/mute.wav.npy|%s/2b-f0nsf/mute.wav.npy|%s" % (mute_dir, sr2, mute_dir, fea_dim, mute_dir, mute_dir, safe_spk_id))
        else:
            for _ in range(2): opt.append("%s/0_gt_wavs/mute%s.wav|%s/3_feature%s/mute.npy|%s" % (mute_dir, sr2, mute_dir, fea_dim, safe_spk_id))
            
    shuffle(opt)
    with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf-8") as f: f.write("\n".join(opt))
    
    config_path = "v1/%s.json" % sr2 if version19 == "v1" or sr2 == "40k" else "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config.json_config[config_path], f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")
            
    save_latest_flag = 1 if str(if_save_latest13).lower() in ["yes", "true", "1"] else 0
    cache_gpu_flag = 1 if str(if_cache_gpu17).lower() in ["yes", "true", "1"] else 0
    save_every_weights_flag = 1 if str(if_save_every_weights18).lower() in ["yes", "true", "1"] else 0
            
    cmd = [
        config.python_cmd, "infer/modules/train/train.py",
        "-e", str(exp_dir1), "-sr", str(sr2), "-f0", "1" if if_f0_3 else "0",
        "-bs", str(batch_size12), "-te", str(total_epoch11), "-se", str(save_epoch10),
        "-l", str(save_latest_flag), "-c", str(cache_gpu_flag), "-sw", str(save_every_weights_flag), "-v", str(version19)
    ]
    if gpus16: cmd.extend(["-g", str(gpus16)])
    if pretrained_G14: cmd.extend(["-pg", str(pretrained_G14)])
    if pretrained_D15: cmd.extend(["-pd", str(pretrained_D15)])
    
    p = Popen(cmd, cwd=now_dir)
    ACTIVE_CHILD_PROCESSES.add(p)
    p.wait()
    ACTIVE_CHILD_PROCESSES.discard(p)
    if p.returncode != 0:
        return f"❌ การฝึกสอนล้มเหลว (Training Failed with Exit Code {p.returncode})"
    return "✅ การฝึกสอนเสร็จสมบูรณ์ (Training finished successfully.)"

def train_index(exp_dir1, version19):
    exp_dir = os.path.join(now_dir, "logs", str(exp_dir1))
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = os.path.join(exp_dir, "3_feature256") if version19 == "v1" else os.path.join(exp_dir, "3_feature768")
    
    if not os.path.exists(feature_dir): return "❌ กรุณาสกัดคุณลักษณะเสียงก่อน (Please extract features first!)"
    listdir_res = [f for f in os.listdir(feature_dir) if f.lower().endswith(".npy")]
    if len(listdir_res) == 0: return "❌ ไม่พบไฟล์ฟีเจอร์ในโฟลเดอร์ (No feature files found)"
    
    infos, npys = [], []
    for name in sorted(listdir_res):
        if name != "total_fea.npy":
            try: npys.append(np.load(os.path.join(feature_dir, name)))
            except: pass
            
    if not npys: return "❌ ไม่พบไฟล์ฟีเจอร์สำหรับการสร้างดัชนี"
            
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    if big_npy.shape[0] > 2e5:
        infos.append("กำลังปรับลดขนาดกลุ่มข้อมูลด้วย K-Means (Optimizing dataset size using K-Means...)")
        yield "\n".join(infos)
        try: big_npy = (MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * config.n_cpu, compute_labels=False, init="random").fit(big_npy).cluster_centers_)
        except Exception as e:
            infos.append(f"⚠️ K-Means Failed: {e}")
            yield "\n".join(infos)
            
    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
    
    n_samples, dimension = big_npy.shape
    if n_samples < 256:
        infos.append(f"ข้อมูลเวกเตอร์น้อยกว่า 256 ({n_samples} ตัวอย่าง) จะใช้ดัชนีแบบ IndexFlatL2")
        yield "\n".join(infos)
        index = faiss.IndexFlatL2(dimension)
        n_ivf_name = "Flat"
        nprobe = 1
    else:
        n_ivf = max(1, min(int(16 * np.sqrt(n_samples)), n_samples // 39))
        infos.append("เริ่มกระบวนการฝึกสอนดัชนี (Commencing index training...)")
        yield "\n".join(infos)
        index = faiss.index_factory(dimension, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        n_ivf_name = str(n_ivf)
        nprobe = index_ivf.nprobe
        index.train(big_npy)
        
    faiss.write_index(index, os.path.join(exp_dir, f"trained_IVF{n_ivf_name}_Flat_nprobe_{nprobe}_{exp_dir1}_{version19}.index"))
    
    infos.append("กำลังบันทึกข้อมูลดัชนี (Saving index data...)")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add): index.add(big_npy[i : i + batch_size_add])
    
    added_index_name = f"added_IVF{n_ivf_name}_Flat_nprobe_{nprobe}_{exp_dir1}_{version19}.index"
    added_index_log = os.path.join(exp_dir, added_index_name)
    faiss.write_index(index, added_index_log)
    
    # Copy to assets/indices
    indices_dir = os.path.join(now_dir, "assets", "indices")
    os.makedirs(indices_dir, exist_ok=True)
    try: shutil.copyfile(added_index_log, os.path.join(indices_dir, added_index_name))
    except Exception: pass

    infos.append("✅ การสร้างดัชนีเสร็จสมบูรณ์ (Index built successfully!)")
    yield "\n".join(infos)
    clear_vram_safe()

def train1key(exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe):
    infos = []
    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)
    yield get_info_str("ขั้นตอนที่ 1: กำลังประมวลผลข้อมูล (Step 1: Processing data...)")
    for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7): yield get_info_str(_)
    yield get_info_str("ขั้นตอนที่ 2: กำลังสกัดระดับเสียงและคุณลักษณะ (Step 2: Extracting pitch & features...)")
    for _ in extract_f0_feature(gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe): yield get_info_str(_)
    yield get_info_str("ขั้นตอนที่ 3: เริ่มการฝึกสอนโมเดล (Step 3: Training model...)")
    res = click_train(exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19)
    yield get_info_str(res)
    
    exp_path = os.path.join(now_dir, "logs", str(exp_dir1))
    model_exists = any(f.endswith(".pth") and f.startswith("G_") for f in os.listdir(exp_path)) if os.path.exists(exp_path) else False
    if not model_exists or "ล้มเหลว" in res:
        yield get_info_str("🛑 ระงับการสร้างดัชนี (Halted Indexing): ไม่พบโมเดลหลัก หรือการเทรนล้มเหลว")
        return
        
    for _ in train_index(exp_dir1, version19): yield get_info_str(_)
    yield get_info_str("🎉 กระบวนการทั้งหมดเสร็จสมบูรณ์! (All processes completed successfully!)")

def change_info_(ckpt_path):
    log_path = ckpt_path.replace(os.path.basename(ckpt_path), "train.log")
    if not os.path.exists(log_path): return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except Exception:
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}

F0GPUVisible = getattr(config, "dml", False) == False
def change_f0_method(f0method8): return {"visible": (F0GPUVisible if f0method8 == "rmvpe_gpu" else False), "__type__": "update"}

# ---------------------------------------------------------
# UI: DELTA SYNTH Corporate Styling
# ---------------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&family=Sarabun:wght@300;400;600&display=swap');
body, .gradio-container {
    font-family: 'Leelawadee UI', 'Prompt', 'Sarabun', 'Segoe UI', Tahoma, sans-serif !important;
    background-color: #1A1A1A !important;
    color: #F0F0F0 !important;
    font-size: 16px !important;
}
.gr-box, div.gradio-container > div {
    background-color: rgba(26, 26, 26, 0.95) !important;
    border: 1px solid #CC2200 !important;
}
button.primary, .gr-button-primary {
    background: linear-gradient(90deg, #991100 0%, #CC2200 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(204, 34, 0, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: linear-gradient(90deg, #CC2200 0%, #FF4422 100%) !important;
    box-shadow: 0 6px 20px rgba(255, 68, 34, 0.7) !important;
    transform: scale(1.01);
}
button.secondary, .gr-button-secondary {
    background: linear-gradient(90deg, #550000 0%, #880000 100%) !important;
    border: 1px solid #CC2200 !important;
    color: #F0F0F0 !important;
}
h1, h2, h3, h4, h5, h6 { font-weight: 600 !important; color: #F0F0F0 !important; }
"""

with gr.Blocks(title="DELTA SYNTH - RVC WebUI", css=custom_css) as app:
    gr.Markdown("## 🎙️ ระบบแปลงเสียงปัญญาประดิษฐ์ (AI Voice Conversion System - RVC WebUI)")
    gr.Markdown(
        value="ซอฟต์แวร์นี้เปิดให้ใช้งานฟรี ภายใต้ใบอนุญาต MIT ผู้ใช้งานต้องรับผิดชอบต่อไฟล์เสียงที่ถูกผลิตและเผยแพร่ทั้งหมดอย่างเป็นทางการ <br>*(This software is open-source under the MIT license. The user holds sole responsibility for all generated and distributed content.)*\n\n**Powered by DELTA SYNTH Engine**"
    )
    with gr.Tabs():
        # --- TAB 1: INFERENCE ---
        with gr.TabItem("🎙️ การอนุมานโมเดล (Model Inference)"):
            with gr.Row():
                sid0 = gr.Dropdown(label="เลือกเสียงเป้าหมาย (Select Target Voice)", choices=sorted(names))
                with gr.Column():
                    refresh_button = gr.Button("🔄 รีเฟรชข้อมูลเสียงและดัชนี (Refresh Voice & Index Data)", variant="primary")
                    clean_button = gr.Button("🧹 คืนค่าหน่วยความจำ VRAM (Clear VRAM)", variant="secondary")
                spk_item = gr.Slider(minimum=0, maximum=2333, step=1, label="โปรดระบุหมายเลขผู้พูด (Please Select Speaker ID)", value=0, visible=False, interactive=True)
                clean_button.click(fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean")
            
            with gr.TabItem("การประมวลผลไฟล์เดี่ยว (Single File Processing)"):
                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            vc_transform0 = gr.Number(label="ปรับระดับเสียง (Pitch Shifting: Semitones, +12 Octave Up, -12 Down)", value=0)
                            with gr.Row():
                                input_audio0 = gr.Textbox(label="เส้นทางไฟล์เสียงต้นฉบับ (Source Audio Path)", placeholder="ตัวอย่าง: H:\\Export to UTAU\\Test.wav", value="", scale=8)
                                btn_browse_audio0 = gr.Button("📄 เลือกไฟล์ (Browse)", scale=2, variant="secondary")
                            with gr.Row():
                                file_index1 = gr.Textbox(label="เส้นทางไฟล์ดัชนีระบุเอง (Custom Feature Index Path, ข้ามได้หากตั้งค่าอัตโนมัติ)", placeholder="logs\\A.index", value="", interactive=True, scale=8)
                                btn_browse_index1 = gr.Button("📑 เลือกดัชนี (Browse)", scale=2, variant="secondary")
                            file_index2 = gr.Dropdown(label="เลือกไฟล์ดัชนีอัตโนมัติ (Auto-Select Index Path)", choices=sorted(index_paths), interactive=True)
                            f0method0 = gr.Radio(label="อัลกอริทึมวิเคราะห์ระดับเสียง (Pitch Extraction Algorithm)", choices=(["pm", "harvest", "crepe", "rmvpe"] if getattr(config, "dml", False) == False else ["pm", "harvest", "rmvpe"]), value="rmvpe", interactive=True)
                            
                            btn_browse_audio0.click(fn=lambda cur: browse_file(cur, [("Audio Files", "*.wav;*.flac;*.mp3;*.m4a;*.ogg"), ("All files", "*.*")]), inputs=[input_audio0], outputs=[input_audio0])
                            btn_browse_index1.click(fn=lambda cur: browse_file(cur, [("Index Files", "*.index"), ("All files", "*.*")]), inputs=[file_index1], outputs=[file_index1])

                        with gr.Column():
                            resample_sr0 = gr.Slider(minimum=0, maximum=48000, step=1, label="ปรับอัตราสุ่มตัวอย่างซ้ำ (Resampling Rate, 0 = Off)", value=0, interactive=True)
                            rms_mix_rate0 = gr.Slider(minimum=0, maximum=1, label="ความเข้มข้นของการผสานระดับเสียง (Volume Envelope Mix Rate)", value=0.25, interactive=True)
                            
                            # ปรับค่าเริ่มต้นเพื่อเพิ่มพื้นที่พยัญชนะให้กว้างขึ้นและชัดเจนขึ้น (Lower Value = More Protection)
                            protect0 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="การปกป้องพยัญชนะและเสียงหายใจ (Consonant & Breath Protection: ค่าน้อย = ปกป้องมาก)", value=0.33, interactive=True)
                            filter_radius0 = gr.Slider(minimum=0, maximum=7, step=1, label="รัศมีตัวกรองเสียงรบกวน (Median Filter Radius)", value=3, interactive=True)
                            index_rate1 = gr.Slider(minimum=0, maximum=1, label="สัดส่วนการใช้อิทธิพลดัชนี (Index Influence Ratio)", value=0.75, interactive=True)
                            f0_file = gr.File(label="ไฟล์เส้นโค้งระดับเสียง (F0 Curve File, หากมี)", visible=False)
                            
                        refresh_button.click(fn=change_choices, inputs=[], outputs=[sid0, file_index2], api_name="infer_refresh")
                
                with gr.Group():
                    with gr.Column():
                        but0 = gr.Button("▶️ เริ่มกระบวนการประมวลผล (Start Rendering)", variant="primary")
                        with gr.Row():
                            vc_output1 = gr.Textbox(label="สถานะการดำเนินการ (Operation Log & Status)")
                            vc_output2 = gr.Audio(label="ผลลัพธ์เสียงที่สมบูรณ์แบบ (Perfect-Length Output Audio)")
                        
                        but0.click(vc.vc_single, [spk_item, input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2, index_rate1, filter_radius0, resample_sr0, rms_mix_rate0, protect0], [vc_output1, vc_output2], api_name="infer_convert")

            with gr.TabItem("การประมวลผลแบบชุด (Batch Processing)"):
                gr.Markdown("นำเข้าโฟลเดอร์ไฟล์เสียง หรือ อัปโหลดไฟล์จำนวนมากเพื่อดำเนินการพร้อมกัน (Import an audio folder or upload multiple files for bulk rendering.)")
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(label="ปรับระดับเสียง (Pitch Shifting: Semitones)", value=0)
                        with gr.Row():
                            opt_input = gr.Textbox(label="ไดเรกทอรีส่งออก (Output Directory)", value="H:/Render file to UTAU/Singer/Language", scale=8)
                            btn_browse_opt = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                        with gr.Row():
                            file_index3 = gr.Textbox(label="เส้นทางไฟล์ดัชนีระบุเอง (Custom Feature Index Path)", value="", interactive=True, scale=8)
                            btn_browse_index3 = gr.Button("📑 เลือกดัชนี (Browse)", scale=2, variant="secondary")
                        file_index4 = gr.Dropdown(label="เลือกไฟล์ดัชนีอัตโนมัติ (Auto-Select Index)", choices=sorted(index_paths), interactive=True)
                        f0method1 = gr.Radio(label="อัลกอริทึมวิเคราะห์ระดับเสียง (Pitch Extraction Algorithm)", choices=(["pm", "harvest", "crepe", "rmvpe"] if getattr(config, "dml", False) == False else ["pm", "harvest", "rmvpe"]), value="rmvpe", interactive=True)
                        format1 = gr.Radio(label="รูปแบบไฟล์ปลายทาง (Export Format)", choices=["wav", "flac", "mp3", "m4a"], value="wav", interactive=True)
                        
                        refresh_button.click(fn=lambda: change_choices()[1], inputs=[], outputs=[file_index4], api_name="infer_refresh_batch")
                        btn_browse_opt.click(fn=browse_folder, inputs=[opt_input], outputs=[opt_input])
                        btn_browse_index3.click(fn=lambda cur: browse_file(cur, [("Index Files", "*.index"), ("All files", "*.*")]), inputs=[file_index3], outputs=[file_index3])

                    with gr.Column():
                        resample_sr1 = gr.Slider(minimum=0, maximum=48000, step=1, label="ปรับอัตราสุ่มตัวอย่างซ้ำ (Resampling Rate)", value=0, interactive=True)
                        rms_mix_rate1 = gr.Slider(minimum=0, maximum=1, label="ความเข้มข้นของการผสานระดับเสียง (Volume Envelope Mix Rate)", value=0.25, interactive=True)
                        protect1 = gr.Slider(minimum=0, maximum=0.5, step=0.01, label="การปกป้องพยัญชนะและเสียงหายใจ (Consonant & Breath Protection: ค่าน้อย = ปกป้องมาก)", value=0.33, interactive=True)
                        filter_radius1 = gr.Slider(minimum=0, maximum=7, step=1, label="รัศมีตัวกรองเสียงรบกวน (Median Filter Radius)", value=3, interactive=True)
                        index_rate2 = gr.Slider(minimum=0, maximum=1, label="สัดส่วนการใช้อิทธิพลดัชนี (Index Influence Ratio)", value=0.75, interactive=True)

                with gr.Row():
                    with gr.Column(scale=8):
                        with gr.Row():
                            dir_input = gr.Textbox(label="เส้นทางโฟลเดอร์ต้นฉบับ (Source Directory Path)", value="", placeholder="H:\\Export to UTAU\\EN", scale=8)
                            btn_browse_dir = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                        btn_browse_dir.click(fn=browse_folder, inputs=[dir_input], outputs=[dir_input])
                    inputs = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียงหลายไฟล์ (Or Upload Multiple Files)", scale=4)
                
                with gr.Row():
                    batch_btn = gr.Button("✅ ตรวจสอบสถานะและเริ่มประมวลผลชุด (Check Status / Start Batch Render)", variant="primary")
                    cancel_btn = gr.Button("🛑 ยกเลิกคำสั่ง (Cancel Render)", variant="secondary")
                    vc_output3 = gr.Textbox(label="ตรวจสอบการทำงานแบบเรียลไทม์ (Real-time Status Tracking)")
                    
                    render_event = batch_btn.click(vc.vc_multi, [spk_item, dir_input, opt_input, inputs, vc_transform1, f0method1, file_index3, file_index4, index_rate2, filter_radius1, resample_sr1, rms_mix_rate1, protect1, format1], [vc_output3], api_name="infer_convert_batch")
                    # เรียกใช้ระบบ Safe Cancel ที่ปกป้องการแครช
                    cancel_btn.click(safe_cancel_process, inputs=[], outputs=[vc_output3], cancels=[render_event])

                sid0.change(fn=vc.get_vc, inputs=[sid0, protect0, protect1], outputs=[spk_item, protect0, protect1, file_index2, file_index4], api_name="infer_change_voice")

        # --- TAB 2: UVR5 ---
        with gr.TabItem("เครื่องมือแยกองค์ประกอบเสียง (UVR5 Separation & De-Echo)"):
            with gr.Group():
                gr.Markdown("ระบบสกัดเสียงร้องและเสียงดนตรี พร้อมเครื่องมือลบเสียงสะท้อนอย่างมีประสิทธิภาพ (Vocal/Instrument separation and Dereverb using UVR5 models).")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            dir_wav_input = gr.Textbox(label="โฟลเดอร์เสียงต้นฉบับ (Input Audio Directory)", placeholder="C:\\Users\\Desktop\\todo-songs", scale=8)
                            btn_browse_uvr_in = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        btn_browse_uvr_in.click(fn=browse_folder, inputs=[dir_wav_input], outputs=[dir_wav_input])
                        wav_inputs = gr.File(file_count="multiple", label="หรือ อัปโหลดไฟล์เสียง (Or Upload Audio Files)")
                    with gr.Column():
                        model_choose = gr.Dropdown(label="โมเดลสกัดเสียง (Select Model)", choices=uvr5_names)
                        agg = gr.Slider(minimum=0, maximum=20, step=1, label="ความแม่นยำในการสกัดเสียง (Aggressiveness)", value=10, interactive=True, visible=False)
                        with gr.Row():
                            opt_vocal_root = gr.Textbox(label="โฟลเดอร์ปลายทางสำหรับเสียงหลัก (Output Vocal Directory)", value="opt", scale=8)
                            btn_browse_vocal = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        with gr.Row():
                            opt_ins_root = gr.Textbox(label="โฟลเดอร์ปลายทางสำหรับเสียงดนตรี (Output Instrument Directory)", value="opt", scale=8)
                            btn_browse_ins = gr.Button("📁 เลือกโฟลเดอร์", scale=2, variant="secondary")
                        format0 = gr.Radio(label="รูปแบบไฟล์ส่งออก (Export Format)", choices=["wav", "flac", "mp3", "m4a"], value="flac", interactive=True)
                        btn_browse_vocal.click(fn=browse_folder, inputs=[opt_vocal_root], outputs=[opt_vocal_root])
                        btn_browse_ins.click(fn=browse_folder, inputs=[opt_ins_root], outputs=[opt_ins_root])
                but2 = gr.Button("▶️ เริ่มกระบวนการแยกเสียง (Start Separation)", variant="primary")
                vc_output4 = gr.Textbox(label="รายงานสถานะการทำงาน (Operation Log)")
                but2.click(uvr, [model_choose, dir_wav_input, opt_vocal_root, wav_inputs, opt_ins_root, agg, format0], [vc_output4], api_name="uvr_convert")

        # --- TAB 3: TRAINING ---
        with gr.TabItem("ระบบฝึกสอนโมเดล (Model Training)"):
            gr.Markdown("ศูนย์กลางการตั้งค่าและดำเนินการฝึกสอนโมเดลเสียงเฉพาะบุคคล (Setup and execute personalized voice model training).")
            with gr.Row():
                exp_dir1 = gr.Textbox(label="ชื่อโปรเจกต์การทดลอง (Experiment Name)", value="NameAI")
                sr2 = gr.Radio(label="อัตราการสุ่มตัวอย่างเป้าหมาย (Target Sample Rate)", choices=["40k", "48k"], value="40k", interactive=True)
                if_f0_3 = gr.Radio(label="ใช้งานการชี้นำระดับเสียงหรือไม่? (Use Pitch Guidance?)", choices=[True, False], value=True, interactive=True)
                version19 = gr.Radio(label="เวอร์ชันของอัลกอริทึมโมเดล (Model Version)", choices=["v1", "v2"], value="v2", interactive=True)
                np7 = gr.Slider(minimum=0, maximum=getattr(config, "n_cpu", 4), step=1, label="เธรด CPU ที่ใช้ประมวลผล (CPU Threads Allocated)", value=int(np.ceil(getattr(config, "n_cpu", 4) / 1.5)), interactive=True)
            
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=8):
                        with gr.Row():
                            trainset_dir4 = gr.Textbox(label="ไดเรกทอรีชุดข้อมูลเสียง (Dataset Directory)", value="H:\\Data\\Singer...", scale=8)
                            btn_browse_train = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
                        btn_browse_train.click(fn=browse_folder, inputs=[trainset_dir4], outputs=[trainset_dir4])
                    spk_id5 = gr.Slider(minimum=0, maximum=4, step=1, label="หมายเลขอ้างอิงผู้พูด (Speaker ID)", value=0, interactive=True)
                with gr.Row():
                    train_preprocess_btn = gr.Button("⚙️ ประมวลผลและเตรียมข้อมูล (Process Data)", variant="primary")
                    info1 = gr.Textbox(label="สถานะของระบบ (System Status)", value="")
                    train_preprocess_btn.click(preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1], api_name="train_preprocess")
            
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(label="หมายเลขหน่วยประมวลผลภาพ (GPU IDs)", value=gpus, interactive=True, visible=F0GPUVisible)
                        gpu_info9 = gr.Textbox(label="ข้อมูลสเปคฮาร์ดแวร์ (Hardware Specifications)", value=gpu_info, visible=F0GPUVisible)
                    with gr.Column():
                        f0method8 = gr.Radio(label="อัลกอริทึมสกัดระดับเสียง (Pitch Extraction Algorithm)", choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"], value="rmvpe_gpu", interactive=True)
                        gpus_rmvpe = gr.Textbox(label="ทรัพยากร GPU สำหรับ RMVPE (GPUs for RMVPE)", value="%s-%s" % (gpus, gpus), interactive=True, visible=F0GPUVisible)
                    but2 = gr.Button("🔍 สกัดคุณลักษณะเสียง (Extract Features)", variant="primary")
                    info2 = gr.Textbox(label="สถานะการสกัดข้อมูล (Extraction Status)", value="", max_lines=8)
                    f0method8.change(fn=change_f0_method, inputs=[f0method8], outputs=[gpus_rmvpe])
                    but2.click(extract_f0_feature, [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe], [info2], api_name="train_extract_f0_feature")
            
            with gr.Group():
                with gr.Row():
                    save_epoch10 = gr.Slider(minimum=1, maximum=50, step=1, label="ความถี่ในการบันทึก / รอบ (Save Frequency per Epoch)", value=40, interactive=True)
                    total_epoch11 = gr.Slider(minimum=2, maximum=1000, step=1, label="จำนวนรอบการฝึกสอนทั้งหมด (Total Epochs)", value=200, interactive=True)
                    batch_size12 = gr.Slider(minimum=1, maximum=40, step=1, label="ขนาดแบทช์ต่อรอบ (Batch Size per GPU)", value=default_batch_size, interactive=True)
                    if_save_latest13 = gr.Radio(label="บันทึกเฉพาะโมเดลล่าสุดเพื่อประหยัดพื้นที่? (Save only latest?)", choices=["Yes", "No"], value="Yes", interactive=True)
                    if_cache_gpu17 = gr.Radio(label="แคชข้อมูลลง VRAM เพื่อความเร็ว? (Cache to VRAM?)", choices=["Yes", "No"], value="Yes", interactive=True)
                    if_save_every_weights18 = gr.Radio(label="บันทึกไฟล์น้ำหนักสำรองเป็นระยะ? (Save weights periodically?)", choices=["Yes", "No"], value="Yes", interactive=True)
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            pretrained_G14 = gr.Textbox(label="โมเดลปฐมภูมิ G (Pretrained G)", value="assets/pretrained_v2/f0G40k.pth", interactive=True, scale=8)
                            btn_browse_pg = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    with gr.Column():
                        with gr.Row():
                            pretrained_D15 = gr.Textbox(label="โมเดลปฐมภูมิ D (Pretrained D)", value="assets/pretrained_v2/f0D40k.pth", interactive=True, scale=8)
                            btn_browse_pd = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    
                    btn_browse_pg.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[pretrained_G14], outputs=[pretrained_G14])
                    btn_browse_pd.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[pretrained_D15], outputs=[pretrained_D15])
                    sr2.change(change_sr2, [sr2, if_f0_3, version19], [pretrained_G14, pretrained_D15])
                    version19.change(change_version19, [sr2, if_f0_3, version19], [pretrained_G14, pretrained_D15, sr2])
                    if_f0_3.change(change_f0, [if_f0_3, sr2, version19], [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15])
                    gpus16 = gr.Textbox(label="การ์ดจอที่ใช้สำหรับการฝึกสอน (GPUs for Training)", value=gpus, interactive=True)
                    
                    but3 = gr.Button("🚀 เริ่มฝึกสอนโมเดล (Train Model)", variant="primary")
                    but4 = gr.Button("📈 ฝึกสอนดัชนีคุณลักษณะ (Train Feature Index)", variant="primary")
                    but5 = gr.Button("⚡ ดำเนินการฝึกสอนแบบครบวงจร (One-Click Training)", variant="primary")
                    info3 = gr.Textbox(label="บันทึกการทำงานของระบบ (System Operation Log)", value="", max_lines=10)
                    
                    but3.click(click_train, [exp_dir1, sr2, if_f0_3, spk_id5, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19], info3, api_name="train_start")
                    but4.click(train_index, [exp_dir1, version19], info3)
                    but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe], info3, api_name="train_start_all")

        # --- TAB 4: CKPT ---
        with gr.TabItem("การจัดการจุดตรวจสอบ (CKPT Processing)"):
            with gr.Group():
                gr.Markdown("ผสานรวมจุดตรวจสอบเพื่อประสิทธิภาพใหม่ (Model Fusion/Merging)")
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            ckpt_a = gr.Textbox(label="เส้นทางโมเดลหลัก A (Model A Path)", value="", interactive=True, scale=8)
                            btn_browse_ca = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    with gr.Column(scale=4):
                        with gr.Row():
                            ckpt_b = gr.Textbox(label="เส้นทางโมเดลรอง B (Model B Path)", value="", interactive=True, scale=8)
                            btn_browse_cb = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    alpha_a = gr.Slider(minimum=0, maximum=1, label="สัดส่วนน้ำหนักโมเดล A (Model A Weight Ratio)", value=0.5, interactive=True, scale=2)
                btn_browse_ca.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_a], outputs=[ckpt_a])
                btn_browse_cb.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_b], outputs=[ckpt_b])
                with gr.Row():
                    sr_ = gr.Radio(label="อัตราสุ่มตัวอย่าง (Target Sample Rate)", choices=["40k", "48k"], value="40k", interactive=True)
                    if_f0_ = gr.Radio(label="อนุรักษ์ระดับเสียงนำทาง? (Maintain Pitch Guidance?)", choices=["Yes", "No"], value="Yes", interactive=True)
                    info__ = gr.Textbox(label="ข้อมูลรายละเอียดโมเดล (Model Info)", value="", max_lines=8, interactive=True)
                    name_to_save0 = gr.Textbox(label="ชื่อไฟล์บันทึกผลลัพธ์ (Save File Name)", value="", max_lines=1, interactive=True)
                    version_2 = gr.Radio(label="เวอร์ชันโครงสร้าง (Model Version)", choices=["v1", "v2"], value="v2", interactive=True)
                with gr.Row():
                    but6 = gr.Button("🔗 ผสานโมเดล (Merge Models)", variant="primary")
                    info4 = gr.Textbox(label="รายงานสถานะการทำงาน (Operation Status Log)", value="", max_lines=8)
                but6.click(merge, [ckpt_a, ckpt_b, alpha_a, sr_, if_f0_, info__, name_to_save0, version_2], info4, api_name="ckpt_merge")

            with gr.Group():
                gr.Markdown("ปรับปรุงและแก้ไขข้อมูลประจำตัวโมเดล (Modify Model Information)")
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            ckpt_path0 = gr.Textbox(label="เส้นทางไฟล์โมเดลเป้าหมาย (Target Model Path)", value="", interactive=True, scale=8)
                            btn_browse_c0 = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    info_ = gr.Textbox(label="ระบุข้อมูลที่ต้องการแก้ไข (Data to modify)", value="", max_lines=8, interactive=True, scale=4)
                    name_to_save1 = gr.Textbox(label="ระบุชื่อไฟล์ใหม่ (New File Name)", value="", max_lines=8, interactive=True, scale=3)
                btn_browse_c0.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_path0], outputs=[ckpt_path0])
                with gr.Row():
                    but7 = gr.Button("💾 บันทึกข้อมูลที่แก้ไข (Modify & Save)", variant="primary")
                    info5 = gr.Textbox(label="สถานะระบบ (System Status)", value="", max_lines=8)
                but7.click(change_info, [ckpt_path0, info_, name_to_save1], info5, api_name="ckpt_modify")

            with gr.Group():
                gr.Markdown("เรียกดูข้อมูลรายละเอียดของโมเดล (View Model Metadata)")
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            ckpt_path1 = gr.Textbox(label="เส้นทางไฟล์โมเดลที่ต้องการตรวจสอบ (Model Path for Inspection)", value="", interactive=True, scale=8)
                            btn_browse_c1 = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    but8 = gr.Button("📄 ตรวจสอบข้อมูล (Check Metadata)", variant="primary", scale=2)
                    info6 = gr.Textbox(label="รายละเอียดเชิงลึก (In-depth Details)", value="", max_lines=8, scale=5)
                btn_browse_c1.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_path1], outputs=[ckpt_path1])
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")

            with gr.Group():
                gr.Markdown("กระบวนการดึงเฉพาะโมเดลย่อย (Extract Lightweight Model)")
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            ckpt_path2 = gr.Textbox(label="เส้นทางโมเดลต้นทางขนาดใหญ่ (Source Large Model Path)", value="", interactive=True, scale=8)
                            btn_browse_c2 = gr.Button("📄 เลือก", scale=2, variant="secondary")
                    save_name = gr.Textbox(label="ตั้งชื่อสำหรับไฟล์บันทึก (Save Name)", value="", interactive=True, scale=3)
                    sr__ = gr.Radio(label="ตั้งค่าอัตราสุ่มตัวอย่าง (Sample Rate)", choices=["32k", "40k", "48k"], value="40k", interactive=True)
                    if_f0__ = gr.Radio(label="ใช้ระดับเสียงนำทาง? (Use Pitch Guidance?)", choices=["1", "0"], value="1", interactive=True)
                    version_1 = gr.Radio(label="เวอร์ชัน (Version)", choices=["v1", "v2"], value="v2", interactive=True)
                    info___ = gr.Textbox(label="ข้อมูลประกอบ (Model Metadata)", value="", max_lines=8, interactive=True)
                    but9 = gr.Button("✂️ ดึงเฉพาะแกนโมเดล (Extract Core)", variant="primary")
                    info7 = gr.Textbox(label="รายงานสถานะ (Status Report)", value="", max_lines=8)
                    
                    btn_browse_c2.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_path2], outputs=[ckpt_path2])
                    ckpt_path2.change(change_info_, [ckpt_path2], [sr__, if_f0__, version_1])
                but9.click(extract_small_model, [ckpt_path2, save_name, sr__, if_f0__, info___, version_1], info7, api_name="ckpt_extract")

        # --- TAB 5: ONNX ---
        with gr.TabItem("การแปลงไฟล์มาตรฐาน ONNX (ONNX Export)"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        ckpt_dir = gr.Textbox(label="เส้นทางระบุโมเดล RVC ต้นฉบับ (Source RVC Model Path)", value="", interactive=True, scale=8)
                        btn_browse_onnx1 = gr.Button("📄 เลือกไฟล์ (Browse)", scale=2, variant="secondary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        onnx_dir = gr.Textbox(label="เส้นทางปลายทางสำหรับการส่งออก (Destination ONNX Output Path)", value="", interactive=True, scale=8)
                        btn_browse_onnx2 = gr.Button("📁 เลือกโฟลเดอร์ (Browse)", scale=2, variant="secondary")
            
            btn_browse_onnx1.click(fn=lambda cur: browse_file(cur, [("Model Weights", "*.pth"), ("All files", "*.*")]), inputs=[ckpt_dir], outputs=[ckpt_dir])
            btn_browse_onnx2.click(fn=browse_folder, inputs=[onnx_dir], outputs=[onnx_dir])
            
            with gr.Row():
                infoOnnx = gr.Label(label="สถานะการประมวลผล (Processing Status)")
            with gr.Row():
                butOnnx = gr.Button("📤 เริ่มกระบวนการส่งออกโมเดล (Export ONNX Model)", variant="primary")
            butOnnx.click(export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx")

        # --- TAB 6: FAQ ---
        tab_faq = "คำถามที่พบบ่อยและเอกสารอ้างอิง (FAQ & Documentation)"
        with gr.TabItem(tab_faq):
            try:
                with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
                    info = f.read()
                gr.Markdown(value=info)
            except Exception:
                gr.Markdown(traceback.format_exc())

# ==========================================
# กระบวนการเริ่มต้นระบบ (System Initialization Launch)
# ==========================================
if __name__ == "__main__":
    try:
        print("\n[System Log] กำลังเริ่มต้นระบบ DELTA SYNTH - Thai RVC WebUI อย่างเป็นทางการ...")
        
        if getattr(config, "iscolab", False):
            app.queue(concurrency_count=511, max_size=1022).launch(share=True)
        else:
            app.queue(concurrency_count=511, max_size=1022).launch(
                server_name="0.0.0.0",
                server_port=getattr(config, "listen_port", 9378),
                share=False,
                inbrowser=not getattr(config, "noautoopen", False),
                quiet=True
            )
            
    except Exception as e:
        print(f"\n[System Log] เกิดข้อผิดพลาดทางเทคนิคในการเริ่มต้นระบบ WebUI: {e}")
        print("[System Log] โปรดตรวจสอบว่ามีโปรแกรมอื่นกำลังใช้งานพอร์ตนี้อยู่หรือไม่\n")
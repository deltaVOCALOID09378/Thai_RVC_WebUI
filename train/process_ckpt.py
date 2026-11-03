# ==============================================================================
# File Name    : process_ckpt.py
# Version      : v2.3.0-Production
# Target Path  : infer/modules/train/process_ckpt.py
# Function     : Checkpoint Extraction, Weight Pruning, Metadata Editing & Model Fusion
# Credits      : Made And Checked By DELTA SYNTH & All Agentic AI
#                Original by RVC-Project / RVC Team
# ==============================================================================

import os
import sys
import traceback
import json
from collections import OrderedDict
import torch

# [TH] เชื่อมโยงโฟลเดอร์หลักเข้าสู่ sys.path เพื่อป้องกันการ Import โมดูลล้มเหลว
# [EN] Ensure project root is present in sys.path for safe package imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Safe Imports with Fallback Mechanisms
try:
    from tools.file_io import read_text
except ImportError:
    def read_text(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

try:
    from i18n.i18n import I18nAuto
    i18n = I18nAuto()
except ImportError:
    i18n = lambda text: text


def ensure_weights_directory(target_dir="assets/weights"):
    """
    [TH] ตรวจสอบและสร้างโฟลเดอร์จัดเก็บโมเดลน้ำหนักเบา
    [EN] Ensure the target export weights directory exists.
    """
    os.makedirs(target_dir, exist_ok=True)


def sanitize_model_name(name):
    """
    [TH] ปรับแต่งชื่อไฟล์โมเดล ป้องกันการซ้ำซ้อนของนามสกุล .pth
    [EN] Sanitize output model filename and strip redundant .pth extension.
    """
    clean_name = os.path.basename(name).strip()
    if clean_name.lower().endswith(".pth"):
        clean_name = clean_name[:-4]
    return clean_name if clean_name else "unnamed_model"


def normalize_sample_rate(sr_input):
    """
    [TH] แปลงค่าอัตราสุ่มสัญญาณให้อยู่ในรูปแบบมาตรฐาน ('32k', '40k', '48k')
    [EN] Normalize various sample rate representations into standardized keys.
    """
    sr_str = str(sr_input).lower().replace("hz", "").strip()
    if sr_str in ("40000", "40k"):
        return "40k", 40000
    elif sr_str in ("48000", "48k"):
        return "48k", 48000
    elif sr_str in ("32000", "32k"):
        return "32k", 32000
    return "40k", 40000


def get_default_config(sr_key, version="v2"):
    """
    [TH] สร้างโครงสร้างคอนฟิกมาตรฐานตามความถี่สุ่มและเวอร์ชันสถาปัตยกรรม (v1/v2)
    [EN] Construct standard architecture config array for specified sample rate and version.
    """
    version = str(version).lower().strip()
    if sr_key == "40k":
        return [
            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
            [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 10, 2, 2],
            512,
            [16, 16, 4, 4],
            109, 256, 40000,
        ]
    elif sr_key == "48k":
        if version == "v1":
            return [
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 6, 2, 2, 2],
                512,
                [16, 16, 4, 4, 4],
                109, 256, 48000,
            ]
        else:
            return [
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [12, 10, 2, 2],
                512,
                [24, 20, 4, 4],
                109, 256, 48000,
            ]
    elif sr_key == "32k":
        if version == "v1":
            return [
                513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 4, 2, 2, 2],
                512,
                [16, 16, 4, 4, 4],
                109, 256, 32000,
            ]
        else:
            return [
                513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 8, 2, 2],
                512,
                [20, 16, 4, 4],
                109, 256, 32000,
            ]
    return get_default_config("40k", version)


def normalize_speaker_info(speaker_info):
    """
    [TH] จัดระเบียบและเรียงลำดับ ID ผู้พูดสำหรับโมเดลประเภท Multi-Speaker (0~109)
    [EN] Clean and normalize speaker metadata registry (ID range: 0~109).
    """
    result = []
    seen = set()
    for item in speaker_info or []:
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if speaker_id < 0 or speaker_id > 109 or not speaker_name or speaker_id in seen:
            continue
        seen.add(speaker_id)
        result.append({"id": speaker_id, "name": speaker_name})
    return sorted(result, key=lambda item: item["id"])


def savee(ckpt, sr, if_f0, name, epoch, version, hps):
    """
    [TH] บันทึกโมเดลน้ำหนักเบาสำหรับการใช้งานจริง (Inference Weights) ระหว่างขั้นตอนฝึกสอน
    [EN] Extract and save pruned inference model weights during ongoing training.
    """
    try:
        ensure_weights_directory()
        clean_name = sanitize_model_name(name)

        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            # บังคับแปลงเป็น Half Precision (FP16) และย้ายลง CPU เพื่อความเข้ากันได้สูงสุด
            opt["weight"][key] = ckpt[key].detach().cpu().half()

        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = f"{epoch}epoch"
        opt["sr"] = sr
        opt["f0"] = 1 if if_f0 in (1, "1", True, "True") else 0
        opt["version"] = version

        speaker_info = normalize_speaker_info(getattr(hps, "speaker_info", []))
        if speaker_info:
            opt["speaker_info"] = speaker_info

        target_file = f"assets/weights/{clean_name}.pth"
        torch.save(opt, target_file)
        return f"[TH] สำเร็จ: ส่งออกโมเดลไปยัง {target_file}\n[EN] Success: Exported model to {target_file}"
    except Exception:
        return f"[ERROR] Failed in savee:\n{traceback.format_exc()}"


def show_info(path):
    """
    [TH] ตรวจสอบและแสดงรายละเอียดโครงสร้างของไฟล์โมเดล .pth
    [EN] Inspect and display internal metadata from an exported .pth weight file.
    """
    try:
        if not os.path.isfile(path):
            return f"[ERROR] File not found: {path}"

        checkpoint = torch.load(path, map_location="cpu")
        f0_flag = checkpoint.get("f0", "None")
        f0_desc = "เปิดใช้งาน (Enabled / 1)" if f0_flag in (1, "1", True) else "ปิดใช้งาน (Disabled / 0)"

        info_text = (
            f"[TH] ------------------------------------------------------------\n"
            f"     รายละเอียดโมเดล (Model Info) : {checkpoint.get('info', 'None')}\n"
            f"     อัตราสุ่มสัญญาณ (Sample Rate) : {checkpoint.get('sr', 'None')}\n"
            f"     ระบบนำทางระดับเสียง (Pitch F0) : {f0_desc}\n"
            f"     เวอร์ชันสถาปัตยกรรม (Version) : {checkpoint.get('version', 'None')}\n"
            f"[EN] ------------------------------------------------------------\n"
            f"     Info: {checkpoint.get('info', 'None')} | SR: {checkpoint.get('sr', 'None')} | "
            f"F0: {f0_flag} | Version: {checkpoint.get('version', 'None')}"
        )
        return info_text
    except Exception:
        return f"[ERROR] Inspection failed:\n{traceback.format_exc()}"


def extract_small_model(path, name, sr, if_f0, info, version):
    """
    [TH] สกัดโมเดลขนาดเล็กสำหรับนำไปใช้งานจริง จากไฟล์ Checkpoint การฝึกสอน (G_*.pth)
    [EN] Extract standalone inference weights from a full training checkpoint file.
    """
    try:
        ensure_weights_directory()
        clean_name = sanitize_model_name(name)
        sr_key, _ = normalize_sample_rate(sr)

        speaker_info = []
        config_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
        if os.path.isfile(config_path):
            try:
                config_data = json.loads(read_text(config_path))
                speaker_info = normalize_speaker_info(config_data.get("speaker_info", []))
            except Exception:
                pass

        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]

        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].detach().cpu().half()

        # สร้าง Config อัตโนมัติ ป้องกันปัญหา Config หาย
        opt["config"] = get_default_config(sr_key, version)

        if not info or not info.strip():
            info = f"Model extracted from training checkpoint ({sr_key}_{version})"

        opt["info"] = info.strip()
        opt["version"] = str(version).strip()
        opt["sr"] = sr_key
        opt["f0"] = 1 if if_f0 in (1, "1", True, "True") else 0

        if speaker_info:
            opt["speaker_info"] = speaker_info

        target_file = f"assets/weights/{clean_name}.pth"
        torch.save(opt, target_file)
        return f"[TH] สกัดโมเดลสำเร็จ: {target_file}\n[EN] Small model extracted successfully: {target_file}"
    except Exception:
        return f"[ERROR] Extraction failed:\n{traceback.format_exc()}"


def change_info(path, info, name):
    """
    [TH] แก้ไขคำอธิบายและข้อความกำกับของโมเดล (Metadata Information)
    [EN] Modify description and metadata within an existing model weight.
    """
    try:
        ensure_weights_directory()
        if not os.path.isfile(path):
            return f"[ERROR] Model file not found: {path}"

        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info.strip()

        target_name = sanitize_model_name(name if name.strip() else os.path.basename(path))
        target_path = f"assets/weights/{target_name}.pth"

        torch.save(ckpt, target_path)
        return f"[TH] อัปเดตข้อมูลโมเดลสำเร็จ: {target_path}\n[EN] Model metadata updated successfully: {target_path}"
    except Exception:
        return f"[ERROR] Failed to update metadata:\n{traceback.format_exc()}"


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    """
    [TH] ผสานน้ำหนักของสองโมเดลเข้าด้วยกันตามอัตราส่วน (Model Interpolation & Fusion)
    [EN] Fuse and interpolate weights between two compatible models by given alpha ratio.
    """
    try:
        ensure_weights_directory()
        clean_name = sanitize_model_name(name)
        sr_key, _ = normalize_sample_rate(sr)

        def extract_weights(raw_ckpt):
            if "model" in raw_ckpt:
                model_dict = raw_ckpt["model"]
            elif "weight" in raw_ckpt:
                model_dict = raw_ckpt["weight"]
            else:
                model_dict = raw_ckpt

            pruned = OrderedDict()
            for k in model_dict.keys():
                if "enc_q" in k:
                    continue
                pruned[k] = model_dict[k]
            return pruned

        if not os.path.isfile(path1) or not os.path.isfile(path2):
            return "[ERROR] One or both checkpoint files could not be found."

        ckpt1 = torch.load(path1, map_location="cpu")
        ckpt2 = torch.load(path2, map_location="cpu")

        speaker_info1 = normalize_speaker_info(ckpt1.get("speaker_info", []))
        speaker_info2 = normalize_speaker_info(ckpt2.get("speaker_info", []))

        # ดึงหรือสร้าง Config สำหรับโมเดลใหม่
        cfg = ckpt1.get("config", ckpt2.get("config", get_default_config(sr_key, version)))

        weights1 = extract_weights(ckpt1)
        weights2 = extract_weights(ckpt2)

        if sorted(list(weights1.keys())) != sorted(list(weights2.keys())):
            return (
                "[TH] ผสานโมเดลล้มเหลว: โครงสร้าง Layer ของทั้งสองโมเดลไม่สอดคล้องกัน\n"
                "[EN] Fusion failed: Layer architecture mismatch between candidate models."
            )

        opt = OrderedDict()
        opt["weight"] = {}
        alpha = float(alpha1)

        for key in weights1.keys():
            t1 = weights1[key].float()
            t2 = weights2[key].float()

            if key == "emb_g.weight" and t1.shape != t2.shape:
                min_dim = min(t1.shape[0], t2.shape[0])
                opt["weight"][key] = (
                    alpha * t1[:min_dim] + (1.0 - alpha) * t2[:min_dim]
                ).cpu().half()
            else:
                opt["weight"][key] = (alpha * t1 + (1.0 - alpha) * t2).cpu().half()

        opt["config"] = cfg
        opt["sr"] = sr_key

        # ตรวจสอบตัวเลือก F0 ให้ครอบคลุมทุกภาษา (TH / EN / CN / Numeric)
        is_f0_active = f0 in (
            1, "1", True, "True", "yes", "Yes", "YES",
            "ใช่", "เปิด", i18n("是"), "是"
        )
        opt["f0"] = 1 if is_f0_active else 0
        opt["version"] = str(version).strip()
        opt["info"] = info.strip() if info else f"Fused Model ({alpha1 * 100:.1f}% A + {(1 - alpha1) * 100:.1f}% B)"

        if speaker_info1 and speaker_info1 == speaker_info2:
            opt["speaker_info"] = speaker_info1

        target_file = f"assets/weights/{clean_name}.pth"
        torch.save(opt, target_file)
        return (
            f"[TH] ผสานโมเดลสำเร็จ: ส่งออกไฟล์ไปยัง {target_file}\n"
            f"[EN] Model fusion completed successfully: {target_file}"
        )
    except Exception:
        return f"[ERROR] Fusion execution failed:\n{traceback.format_exc()}"
# -*- coding: utf-8 -*-
# ==============================================================================
# File Name   : process_ckpt.py
# Version     : 1.5 (PyTorch 2.6+ Safe Load & Parameter Sanitization Edition)
# Location    : infer/lib/train/process_ckpt.py
# Made And Checked By DELTA SYNTH & All Agentic AI
# Original by : Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================

import os
import sys
import traceback
from collections import OrderedDict
import torch
from i18n.i18n import I18nAuto

i18n = I18nAuto()


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _norm_sr(sr):
    s = str(sr).lower().strip()
    if s in ("32k", "32000", "2"):
        return "32k"
    if s in ("48k", "48000", "1"):
        return "48k"
    return "40k"


def _norm_f0(if_f0):
    if isinstance(if_f0, bool):
        return 1 if if_f0 else 0
    s = str(if_f0).lower().strip()
    return 1 if s in ("1", "true", "yes", "y", "t") else 0


def _norm_ver(version):
    s = str(version).lower().strip()
    return "v1" if s in ("v1", "0") else "v2"


def savee(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        sr = _norm_sr(sr)
        if_f0 = _norm_f0(if_f0)
        version = _norm_ver(version)
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
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
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        os.makedirs("assets/weights", exist_ok=True)
        out_path = f"assets/weights/{name}.pth"
        torch.save(opt, out_path)
        return f"✅ สำเร็จ (Success): {out_path}"
    except Exception:
        return traceback.format_exc()


def show_info(path):
    try:
        a = _safe_torch_load(path)
        return "ข้อมูลโมเดล (Model Info): %s\nอัตราสุ่มตัวอย่าง (Sample Rate): %s\nใช้การชี้นำระดับเสียง (Pitch Guidance): %s\nเวอร์ชัน (Version): %s" % (
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
            a.get("version", "None"),
        )
    except Exception:
        return traceback.format_exc()


def extract_small_model(path, name, sr, if_f0, info, version):
    try:
        sr = _norm_sr(sr)
        if_f0 = _norm_f0(if_f0)
        version = _norm_ver(version)
        ckpt = _safe_torch_load(path)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000,
            ]
        elif sr == "48k":
            if version == "v1":
                opt["config"] = [
                    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 48000,
                ]
            else:
                opt["config"] = [
                    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000,
                ]
        elif sr == "32k":
            if version == "v1":
                opt["config"] = [
                    513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 32000,
                ]
            else:
                opt["config"] = [
                    513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2], 512, [20, 16, 4, 4], 109, 256, 32000,
                ]
        opt["info"] = info if info else "Extracted Small Model"
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        os.makedirs("assets/weights", exist_ok=True)
        torch.save(opt, f"assets/weights/{name}.pth")
        return "✅ ดึงโมเดลสำเร็จ (Success): assets/weights/%s.pth" % name
    except Exception:
        return traceback.format_exc()


def change_info(path, info, name):
    try:
        ckpt = _safe_torch_load(path)
        ckpt["info"] = info
        if not name:
            name = os.path.basename(path)
        if not name.endswith(".pth"):
            name += ".pth"
        os.makedirs("assets/weights", exist_ok=True)
        torch.save(ckpt, f"assets/weights/{name}")
        return f"✅ แก้ไขข้อมูลโมเดลสำเร็จ (Success): assets/weights/{name}"
    except Exception:
        return traceback.format_exc()


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:
        sr = _norm_sr(sr)
        f0 = _norm_f0(f0)
        version = _norm_ver(version)

        def extract(ckpt):
            a = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        ckpt1 = _safe_torch_load(path1)
        ckpt2 = _safe_torch_load(path2)
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "❌ โครงสร้างสถาปัตยกรรมของทั้งสองโมเดลไม่ตรงกัน (Fail to merge the models. The model architectures are not the same.)"
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        opt["config"] = cfg
        opt["sr"] = sr
        opt["f0"] = 1 if f0 else 0
        opt["version"] = version
        opt["info"] = info
        os.makedirs("assets/weights", exist_ok=True)
        if not name.endswith(".pth"):
            name += ".pth"
        torch.save(opt, f"assets/weights/{name}")
        return f"✅ ผสานโมเดลสำเร็จ (Success): assets/weights/{name}"
    except Exception:
        return traceback.format_exc()

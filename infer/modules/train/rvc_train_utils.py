# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : rvc_train_utils.py
# เวอร์ชั่นโค้ด (Version)      : 2.4 (Atomic Checkpoint Save, VRAM OOM Guard & Modern AMP)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/train/rvc_train_utils.py
# หน้าที่ (Function/Role)     : โมดูลเสริมสำหรับการฝึกสอน RVC ดูแลอุปกรณ์ประมวลผล, แคชข้อมูลบน GPU
#                             (พร้อมระบบป้องกัน VRAM เต็ม) และการบันทึกไฟล์ Checkpoint แบบ Atomic
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by Patiphat Wongyai (Delta) & RVC-Project
# ==============================================================================
from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
import random
from typing import Iterator, List, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DeviceHelper:
    @staticmethod
    def get_device(rank: int = 0) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", rank)
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            return torch.device("xpu", rank)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def select_ddp_backend() -> str:
        if torch.cuda.is_available() and os.name != "nt":
            return "nccl"
        return "gloo"

    @staticmethod
    def set_device(rank: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

    @staticmethod
    def setup_amp():
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if getattr(torch, "xpu", None) and torch.xpu.is_available():
                from infer.modules.ipex import ipex_init
                ipex_init()
                from torch.xpu.amp import autocast
                from infer.modules.ipex.gradscaler import gradscaler_init
                return gradscaler_init(), autocast
        except Exception:
            pass
        from torch.cuda.amp import GradScaler, autocast
        return GradScaler, autocast


class GpuDataCache:
    def __init__(self, enabled: bool, use_f0: bool, device: Union[torch.device, str], non_blocking: bool = True):
        self.enabled = bool(enabled) and str(device).startswith("cuda")
        self.use_f0 = bool(use_f0)
        self.device = torch.device(device)
        self.non_blocking = bool(non_blocking)
        self._cache: List[Tuple[int, Tuple[torch.Tensor, ...]]] = []

    @property
    def is_ready(self) -> bool:
        return len(self._cache) > 0

    @property
    def size(self) -> int:
        return len(self._cache)

    def build(self, train_loader: DataLoader) -> "GpuDataCache":
        if not self.enabled:
            return self
        if self.is_ready:
            return self
        logger.info(f"💾 กำลังแคชข้อมูลลง GPU (Building GPU cache on): {self.device} ...")
        self._cache.clear()
        try:
            for batch_idx, batch in enumerate(train_loader):
                tensors = self._batch_to_tensors(batch)
                gpu_tensors = tuple(t.to(self.device, non_blocking=self.non_blocking) for t in tensors)
                self._cache.append((batch_idx, gpu_tensors))
            logger.info(f"✅ แคชข้อมูลสำเร็จ (Cached batches on GPU successfully): {self.size} batches")
        except RuntimeError as oom_err:
            logger.warning(f"⚠️ หน่วยความจำ VRAM ไม่เพียงพอสำหรับแคชทั้งหมด สลับกลับไปอ่านจาก RAM ปกติ: {oom_err}")
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.enabled = False
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดขณะแคชข้อมูล (Error building cache): {e}")
            self._cache.clear()
            self.enabled = False
        return self

    def iter_epoch(self, train_loader: DataLoader, epoch: int) -> Iterator:
        if self.enabled:
            if not self.is_ready:
                self.build(train_loader)
            else:
                self.shuffle()
            if self.enabled and self.is_ready:
                return iter(self._cache)
        return enumerate(train_loader)

    def shuffle(self) -> "GpuDataCache":
        if self.enabled and self.is_ready:
            random.shuffle(self._cache)
        return self

    def clear(self) -> "GpuDataCache":
        if not self._cache:
            return self
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def _batch_to_tensors(self, batch) -> Tuple[torch.Tensor, ...]:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2 and isinstance(batch[0], int) and isinstance(batch[1], (list, tuple)):
                batch = batch[1]
            return tuple(batch)
        raise ValueError(f"Unexpected batch format: {type(batch)}")

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return self.enabled and self.is_ready


class CheckpointHelper:
    @staticmethod
    def latest_checkpoint_path(dir_path: Union[str, Path], pattern: str = "G_*.pth") -> str:
        d = Path(dir_path)
        files = [f for f in d.glob(pattern) if f.stat().st_size > 1024]
        if not files:
            raise FileNotFoundError(f"❌ ไม่พบจุดเซฟข้อมูล (No checkpoint found at): {d / pattern}")

        def _step(p: Path) -> int:
            match = re.search(r"_(\d+)\.pth$", p.name)
            if match:
                step_num = int(match.group(1))
                # หากเป็นไฟล์ 2333333 (Save Latest) ให้เรียงตามเวลาแก้ไขล่าสุด
                if step_num == 2333333:
                    return int(p.stat().st_mtime * 1000)
                return step_num
            return int(p.stat().st_mtime * 1000)

        latest = max(files, key=_step)
        return str(latest)

    @staticmethod
    def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, load_opt: bool = True) -> Tuple[torch.nn.Module, object, float, int]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(checkpoint["model"], strict=False)
        epoch = int(checkpoint.get("epoch", 1))
        lr = float(checkpoint.get("learning_rate", 1e-4))
        if optimizer is not None and load_opt and "optimizer" in checkpoint and checkpoint["optimizer"]:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                logger.warning(f"ข้ามการโหลดสถานะ Optimizer เดิมเนื่องจากโครงสร้างต่างกัน: {e}")
        return model, optimizer, lr, epoch

    @staticmethod
    def save_checkpoint(model: torch.nn.Module, optimizer, learning_rate: float, epoch: int, path: Union[str, Path]) -> None:
        """บันทึกไฟล์โมเดลแบบ Atomic Write เพื่อป้องกันไฟล์เสียหายหากโปรแกรมถูกปิดกะทันหัน"""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".pth.tmp")
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save({
            "model": state_dict,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "learning_rate": learning_rate,
            "epoch": epoch,
        }, tmp_path)
        os.replace(tmp_path, target_path)


def load_pretrained(model: torch.nn.Module, path: str, logger: logging.Logger = None) -> None:
    if not path or not os.path.exists(path):
        return
    logger = logger or logging.getLogger(__name__)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else (ckpt["weight"] if isinstance(ckpt, dict) and "weight" in ckpt else ckpt)
    target = model.module if hasattr(model, "module") else model
    model_dict = target.state_dict()
    # คัดกรองเฉพาะ Layer ที่มีขนาด Tensor ตรงกัน ป้องกันข้อผิดพลาด spk_embed_dim ไม่เท่ากัน
    filtered_state = {
        k: v for k, v in state.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    target.load_state_dict(filtered_state, strict=False)
    logger.info(
        f"✅ โหลดโมเดลพื้นฐานสำเร็จ (Loaded pretrained base model): {path} | "
        f"Matched layers: {len(filtered_state)}/{len(model_dict)}"
    )
import math
import random
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.utils import index_put


class HuBERTFeatureExtractor:
    """
    คลาสสำหรับจัดการการดึงข้อมูล Feature จากโมเดล HuBERT
    รองรับการทำ Masking, Padding และการดึงข้อมูลจาก Layer ที่ระบุ
    """

    def __init__(
        self, model_path: str = "assets/hubert/hubert_base.pt", device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self._wrap_methods()

    def _load_model(self, model_path: str) -> nn.Module:
        """โหลดโมเดล HuBERT จากไฟล์ Checkpoint"""
        models, _, _ = load_model_ensemble_and_task([model_path], suffix="")
        model = models[0]
        model.to(self.device)
        model.eval()  # ตั้งค่าเริ่มต้นเป็น Evaluation mode
        return model

    def _wrap_methods(self):
        """ทำการเชื่อมต่อฟังก์ชันปรับแต่งเข้ากับโครงสร้างของ Fairseq"""
        # ผูกฟังก์ชันเข้ากับโมเดลแบบไดนามิกอย่างเป็นระเบียบ
        self.model.encoder.extract_features = self.extract_encoder_features.__get__(
            self.model.encoder
        )
        self.model.apply_mask = self.apply_mask_to_input.__get__(self.model)

    @staticmethod
    def pad_to_multiple(
        x: torch.Tensor, multiple: int, dim: int = -1, value: float = 0
    ) -> Tuple[torch.Tensor, int]:
        """ปรับขนาด Sequence ให้หารด้วยค่า multiple ลงตัว"""
        if x is None:
            return None, 0

        current_size = x.size(dim)
        target_size = math.ceil(current_size / multiple) * multiple
        remainder = target_size - current_size

        if remainder == 0:
            return x, 0

        # สร้าง Padding config สำหรับมิติที่ต้องการ
        pad_offset = (0,) * (abs(dim) - 1) * 2
        return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

    def compute_mask_indices(
        self,
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
    ) -> torch.Tensor:
        """
        คำนวณตำแหน่ง Mask สำหรับข้อมูล Input
        ปรับปรุงให้ใช้ประสิทธิภาพจาก PyTorch Tensor สูงสุด
        """
        bsz, all_sz = shape
        mask = torch.full((bsz, all_sz), False, device=self.device)

        for i in range(bsz):
            actual_sz = all_sz
            if padding_mask is not None:
                actual_sz = all_sz - padding_mask[i].sum().item()

            num_mask = int(
                mask_prob * actual_sz / float(mask_length) + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)

            if mask_type == "static":
                lengths = [mask_length] * num_mask
            elif mask_type == "uniform":
                lengths = np.random.randint(
                    int(mask_other), mask_length * 2 + 1, size=num_mask
                )
            elif mask_type == "normal":
                lengths = np.random.normal(mask_length, mask_other, size=num_mask)
                lengths = [max(1, int(round(x))) for x in lengths]
            else:
                raise ValueError(f"Unknown mask selection type: {mask_type}")

            if sum(lengths) == 0:
                lengths[0] = min(mask_length, actual_sz - 1)

            # การสุ่มตำแหน่ง Mask (Simplification for readability and speed)
            mask_idc = []
            if no_overlap:
                # Logic สำหรับกรณีห้ามซ้อนทับกัน (Recursive-like approach)
                # เพื่อความกระชับ ในที่นี้เน้นการทำงานพื้นฐานที่เสถียร
                available_indices = list(range(actual_sz - max(lengths) - min_space))
                for length in sorted(lengths, reverse=True):
                    if not available_indices:
                        break
                    start = random.choice(available_indices)
                    mask_idc.extend(range(start, start + length))
                    # ลบพื้นที่ใกล้เคียงออกเพื่อกัน overlap
                    available_indices = [
                        idx
                        for idx in available_indices
                        if idx < start - min_space or idx > start + length + min_space
                    ]
            else:
                starts = np.random.choice(
                    actual_sz - min(lengths), num_mask, replace=True
                )
                for start, length in zip(starts, lengths):
                    mask_idc.extend(range(start, min(start + length, actual_sz)))

            mask[i, list(set(mask_idc))] = True

        return mask

    def apply_mask_to_input(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """ดำเนินการใส่ Mask Embedding ลงในข้อมูล"""
        B, T, C = x.shape
        if hasattr(self.model, "mask_prob") and self.model.mask_prob > 0:
            mask_indices = self.compute_mask_indices(
                (B, T), padding_mask, self.model.mask_prob, self.model.mask_length
            )
            x[mask_indices] = self.model.mask_emb
        else:
            mask_indices = None
        return x, mask_indices

    def extract_encoder_features(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        tgt_layer: Optional[int] = None,
        min_layer: int = 0,
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        ฟังก์ชันหลักในการประมวลผลผ่าน Encoder Layers
        ปรับปรุงการจัดการ LayerDrop และการถอด Padding ออกหลังจากประมวลผลเสร็จ
        """
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # Convolutional Relative Positional Encoding
        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # จัดการ Padding ให้เข้ากับเงื่อนไขของ Sequence Length
        x, pad_length = HuBERTFeatureExtractor.pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2
        )

        if pad_length > 0:
            if padding_mask is None:
                padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
                padding_mask[:, -pad_length:] = True
            else:
                padding_mask, _ = HuBERTFeatureExtractor.pad_to_multiple(
                    padding_mask, self.required_seq_len_multiple, dim=-1, value=True
                )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # BxTxC -> TxBxC สำหรับ Transformer

        layer_results = []
        for i, layer in enumerate(self.layers):
            # ตรวจสอบ LayerDrop
            if not self.training or (np.random.random() > self.layerdrop):
                x, (z, lr) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if i >= min_layer:
                    layer_results.append((x, z, lr))

            if i == tgt_layer:
                break

        x = x.transpose(0, 1)  # TxBxC -> BxTxC

        # ย้อนกลับขั้นตอน Padding (Undo Padding)
        if pad_length > 0:
            x = x[:, :-pad_length]
            layer_results = [
                (
                    a[:-pad_length],
                    b[:-pad_length] if b is not None else None,
                    c[:-pad_length],
                )
                for a, b, c in layer_results
            ]

        return x, layer_results

    def infer(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        output_layer: int = 9,
    ) -> torch.Tensor:
        """ฟังก์ชันสำหรับใช้งานจริง (Inference)"""
        with torch.no_grad():
            # ดึงข้อมูลผ่าน forward ของ fairseq โดยตรง
            res = self.model.forward(
                source,
                padding_mask=padding_mask,
                features_only=True,
                output_layer=output_layer,
            )

            feature = res["x"]
            # หากเป็น Layer สุดท้าย (ในที่นี้คือ 9 ตามต้นฉบับ) ให้ผ่าน Final Projection
            if output_layer == 9:
                feature = self.model.final_proj(feature)

            return feature

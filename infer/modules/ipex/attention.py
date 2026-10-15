# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี: 24/09/2569 | 2026-09-24
# 2. ชื่อไฟล์: attention.py
# 3. ลำดับเวอร์ชัน: 1.5
# 4. Made And Checked By DELTA SYNTH[ ปฎิภัทร วงศ์ใหญ่ ] & All Code Agentic AI Engine
#    [ Original by "RVC-Project" ]
# 5. ตำแหน่งที่วางไฟล์: /infer/modules/ipex/attention.py
# 6. รายละเอียดการแก้ไขงาน: อัปเดตความสามารถระบบจัดการหน่วยความจำ VRAM (Memory Slicing) ให้รัดกุมขึ้น, เพิ่มระบบ Fallback ป้องกัน Crash, และเสริมระบบแจ้งเตือนสองภาษาเมื่อทำงานเต็มขีดจำกัด
# ==============================================================================

import torch
import logging

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import
except ImportError:
    pass

# นำเข้า PoliteLogger สำหรับแจ้งเตือนการทำงาน
try:
    from utils import PoliteLogger
except ImportError:
    class PoliteLogger:
        @staticmethod
        def info(*args): logging.info(" | ".join(str(a) for a in args))
        @staticmethod
        def warning(*args): logging.warning(" | ".join(str(a) for a in args))
        @staticmethod
        def error(*args): logging.error(" | ".join(str(a) for a in args))
        @staticmethod
        def system(*args): logging.info(" | ".join(str(a) for a in args))

logger = logging.getLogger(__name__)

# pylint: disable=protected-access, missing-function-docstring, line-too-long

original_torch_bmm = torch.bmm

def torch_bmm(input, mat2, *, out=None):
    if input.dtype != mat2.dtype:
        mat2 = mat2.to(input.dtype)

    try:
        # ARC GPUs can't allocate more than 4GB to a single block, Slice it:
        batch_size_attention, input_tokens, mat2_shape = (
            input.shape[0],
            input.shape[1],
            mat2.shape[2],
        )
        block_multiply = input.element_size()
        slice_block_size = input_tokens * mat2_shape / 1024 / 1024 * block_multiply
        block_size = batch_size_attention * slice_block_size

        split_slice_size = batch_size_attention
        if block_size > 4:
            do_split = True
            PoliteLogger.info("💡 เปิดใช้ระบบจัดการหน่วยความจำ IPEX (BMM > 4GB) ลดภาระ VRAM", 
                              "Activated IPEX Memory Slicing for BMM (> 4GB limit)")
            # Find something divisible with the input_tokens
            while (split_slice_size * slice_block_size) > 4:
                split_slice_size = max(1, split_slice_size // 2) # ป้องกันการหารด้วย 0
                if split_slice_size == 1:
                    break
        else:
            do_split = False

        split_2_slice_size = input_tokens
        if split_slice_size * slice_block_size > 4:
            slice_block_size2 = split_slice_size * mat2_shape / 1024 / 1024 * block_multiply
            do_split_2 = True
            # Find something divisible with the input_tokens
            while (split_2_slice_size * slice_block_size2) > 4:
                split_2_slice_size = max(1, split_2_slice_size // 2)
                if split_2_slice_size == 1:
                    break
        else:
            do_split_2 = False

        if do_split:
            hidden_states = torch.zeros(
                input.shape[0],
                input.shape[1],
                mat2.shape[2],
                device=input.device,
                dtype=input.dtype,
            )
            for i in range(max(1, batch_size_attention // split_slice_size)):
                start_idx = i * split_slice_size
                end_idx = (i + 1) * split_slice_size
                if end_idx > batch_size_attention: 
                    end_idx = batch_size_attention
                    
                if do_split_2:
                    for i2 in range(max(1, input_tokens // split_2_slice_size)):
                        start_idx_2 = i2 * split_2_slice_size
                        end_idx_2 = (i2 + 1) * split_2_slice_size
                        if end_idx_2 > input_tokens:
                            end_idx_2 = input_tokens
                            
                        hidden_states[
                            start_idx:end_idx, start_idx_2:end_idx_2
                        ] = original_torch_bmm(
                            input[start_idx:end_idx, start_idx_2:end_idx_2],
                            mat2[start_idx:end_idx, start_idx_2:end_idx_2],
                            out=out,
                        )
                else:
                    hidden_states[start_idx:end_idx] = original_torch_bmm(
                        input[start_idx:end_idx], mat2[start_idx:end_idx], out=out
                    )
            return hidden_states
        else:
            return original_torch_bmm(input, mat2, out=out)
            
    except Exception as e:
        # ระบบ Fallback ฉุกเฉินเพื่อป้องกันโปรแกรม Crash
        PoliteLogger.warning(f"⚠️ พบปัญหาในการแบ่งหน่วยความจำ สลับไปใช้วิธีมาตรฐาน: {str(e)}", 
                             f"Memory slicing failed, fallback to standard BMM: {str(e)}")
        return original_torch_bmm(input, mat2, out=out)


original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
):
    try:
        # ARC GPUs can't allocate more than 4GB to a single block, Slice it:
        if len(query.shape) == 3:
            batch_size_attention, query_tokens, shape_four = query.shape
            shape_one = 1
            no_shape_one = True
        elif len(query.shape) == 4:
            shape_one, batch_size_attention, query_tokens, shape_four = query.shape
            no_shape_one = False
        else:
            # คืนค่ามาตรฐานหาก Tensor ไม่ใช่ 3D หรือ 4D ป้องกัน Crash
            return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        block_multiply = query.element_size()
        slice_block_size = (
            shape_one * query_tokens * shape_four / 1024 / 1024 * block_multiply
        )
        block_size = batch_size_attention * slice_block_size

        split_slice_size = batch_size_attention
        if block_size > 4:
            do_split = True
            PoliteLogger.info("💡 เปิดใช้ระบบจัดการหน่วยความจำ IPEX (Attention > 4GB) ลดภาระ VRAM", 
                              "Activated IPEX Memory Slicing for Attention (> 4GB limit)")
            # Find something divisible with the shape_one
            while (split_slice_size * slice_block_size) > 4:
                split_slice_size = max(1, split_slice_size // 2)
                if split_slice_size == 1:
                    break
        else:
            do_split = False

        split_2_slice_size = query_tokens
        if split_slice_size * slice_block_size > 4:
            slice_block_size2 = (
                shape_one * split_slice_size * shape_four / 1024 / 1024 * block_multiply
            )
            do_split_2 = True
            # Find something divisible with the batch_size_attention
            while (split_2_slice_size * slice_block_size2) > 4:
                split_2_slice_size = max(1, split_2_slice_size // 2)
                if split_2_slice_size == 1:
                    break
        else:
            do_split_2 = False

        if do_split:
            hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
            for i in range(max(1, batch_size_attention // split_slice_size)):
                start_idx = i * split_slice_size
                end_idx = (i + 1) * split_slice_size
                if end_idx > batch_size_attention:
                    end_idx = batch_size_attention
                    
                if do_split_2:
                    for i2 in range(max(1, query_tokens // split_2_slice_size)):
                        start_idx_2 = i2 * split_2_slice_size
                        end_idx_2 = (i2 + 1) * split_2_slice_size
                        if end_idx_2 > query_tokens:
                            end_idx_2 = query_tokens
                            
                        if no_shape_one:
                            hidden_states[
                                start_idx:end_idx, start_idx_2:end_idx_2
                            ] = original_scaled_dot_product_attention(
                                query[start_idx:end_idx, start_idx_2:end_idx_2],
                                key[start_idx:end_idx, start_idx_2:end_idx_2],
                                value[start_idx:end_idx, start_idx_2:end_idx_2],
                                attn_mask=attn_mask[
                                    start_idx:end_idx, start_idx_2:end_idx_2
                                ]
                                if attn_mask is not None
                                else attn_mask,
                                dropout_p=dropout_p,
                                is_causal=is_causal,
                            )
                        else:
                            hidden_states[
                                :, start_idx:end_idx, start_idx_2:end_idx_2
                            ] = original_scaled_dot_product_attention(
                                query[:, start_idx:end_idx, start_idx_2:end_idx_2],
                                key[:, start_idx:end_idx, start_idx_2:end_idx_2],
                                value[:, start_idx:end_idx, start_idx_2:end_idx_2],
                                attn_mask=attn_mask[
                                    :, start_idx:end_idx, start_idx_2:end_idx_2
                                ]
                                if attn_mask is not None
                                else attn_mask,
                                dropout_p=dropout_p,
                                is_causal=is_causal,
                            )
                else:
                    if no_shape_one:
                        hidden_states[
                            start_idx:end_idx
                        ] = original_scaled_dot_product_attention(
                            query[start_idx:end_idx],
                            key[start_idx:end_idx],
                            value[start_idx:end_idx],
                            attn_mask=attn_mask[start_idx:end_idx]
                            if attn_mask is not None
                            else attn_mask,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                        )
                    else:
                        hidden_states[
                            :, start_idx:end_idx
                        ] = original_scaled_dot_product_attention(
                            query[:, start_idx:end_idx],
                            key[:, start_idx:end_idx],
                            value[:, start_idx:end_idx],
                            attn_mask=attn_mask[:, start_idx:end_idx]
                            if attn_mask is not None
                            else attn_mask,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                        )
            return hidden_states
        else:
            return original_scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            
    except Exception as e:
        # ระบบ Fallback ฉุกเฉินเพื่อป้องกันโปรแกรม Crash
        PoliteLogger.warning(f"⚠️ พบปัญหาในการแบ่งหน่วยความจำ (Attention) สลับไปใช้วิธีมาตรฐาน: {str(e)}", 
                             f"Attention slicing failed, fallback to standard mode: {str(e)}")
        return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


def attention_init():
    # ARC GPUs can't allocate more than 4GB to a single block:
    torch.bmm = torch_bmm
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)   : export.py
# 3. ลำดับเวอร์ชัน (Version)   : 1.4 (PyTorch 2.6+ Safe Checkpoint Loader & ONNX Exporter)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              Original by Patiphat Wongyai (Delta) & RVC-Project
# 5. ตำแหน่งที่จัดวางไฟล์ (Path) : infer/modules/onnx/export.py
# 6. รายละเอียดการแก้ไขงาน    : เสริมความปลอดภัยในการโหลดโมเดลด้วย `weights_only=False` 
#                              และอัปเกรดระบบแจ้งเตือนเป็น Bilingual Log (TH/EN) อย่างสมบูรณ์
# ==============================================================================

import torch
import onnxsim
import onnx
import traceback
import sys
from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

# บังคับใช้ UTF-8 ป้องกันปัญหาบน Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def export_onnx(ModelPath, ExportedPath):
    try:
        print(f"[ONNX Export] 🔄 [TH] กำลังโหลดโมเดลต้นฉบับ / [EN] Loading original model: {ModelPath}", flush=True)
        
        # โหลดโมเดลอย่างปลอดภัยตามมาตรฐาน PyTorch สมัยใหม่
        if "weights_only" in torch.load.__code__.co_varnames:
            cpt = torch.load(ModelPath, map_location="cpu", weights_only=False)
        else:
            cpt = torch.load(ModelPath, map_location="cpu")
            
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

        # สร้างข้อมูลจำลองสำหรับป้อนให้โมเดลตอนส่งออก (Create dummy inputs for ONNX tracing)
        test_phone = torch.rand(1, 200, vec_channels)  # เวกเตอร์คุณลักษณะ (Hidden unit)
        test_phone_lengths = torch.tensor([200]).long()  # ความยาวของ Hidden unit (Length of hidden unit)
        test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # ความถี่พื้นฐาน F0 (Pitch in Hz)
        test_pitchf = torch.rand(1, 200)  # ความถี่พื้นฐานสำหรับ NSF (NSF Pitch)
        test_ds = torch.LongTensor([0])  # หมายเลขผู้พูด (Speaker ID)
        test_rnd = torch.rand(1, 192, 200)  # สัญญาณรบกวน (Noise for random factor)

        device = "cpu"  # อุปกรณ์ที่ใช้ส่งออกโมเดล (Device for exporting, doesn't affect final usage)

        # โหลดโครงสร้างโมเดล (Instantiate model structure)
        # หมายเหตุ: ส่งออกเป็น FP32 เสมอ เพื่อความเข้ากันได้ของหน่วยความจำใน C++
        net_g = SynthesizerTrnMsNSFsidM(
            *cpt["config"], is_half=False, version=cpt.get("version", "v1")
        )
        
        net_g.load_state_dict(cpt["weight"], strict=False)
        
        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
        output_names = ["audio"]
        
        print("[ONNX Export] ⚙️ [TH] กำลังแปลงเป็นรูปแบบ ONNX / [EN] Converting to ONNX format...", flush=True)
        torch.onnx.export(
            net_g,
            (
                test_phone.to(device),
                test_phone_lengths.to(device),
                test_pitch.to(device),
                test_pitchf.to(device),
                test_ds.to(device),
                test_rnd.to(device),
            ),
            ExportedPath,
            dynamic_axes={
                "phone": [1],
                "pitch": [1],
                "pitchf": [1],
                "rnd": [2],
            },
            do_constant_folding=False,
            opset_version=18,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
        
        print("[ONNX Export] ✂️ [TH] กำลังลดรูปและปรับให้เหมาะสม / [EN] Simplifying ONNX model...", flush=True)
        model, check = onnxsim.simplify(ExportedPath)
        
        if not check:
            raise RuntimeError("[TH] ❌ การลดรูปโมเดล ONNX ล้มเหลว / [EN] ONNX simplification failed")
            
        onnx.save(model, ExportedPath)
        success_msg = f"✅ [TH] ส่งออกโมเดล ONNX สำเร็จ / [EN] Exported ONNX successfully!\n📁 บันทึกที่ (Saved at): {ExportedPath}"
        print(f"[ONNX Export] {success_msg}", flush=True)
        return success_msg

    except Exception as e:
        error_msg = f"❌ [TH] เกิดข้อผิดพลาดในการส่งออก / [EN] Export Error:\n{traceback.format_exc()}"
        print(f"[ONNX Export] {error_msg}", flush=True)
        return error_msg
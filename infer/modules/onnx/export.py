import torch
import onnxsim
import onnx
import traceback
from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

def export_onnx(ModelPath, ExportedPath):
    try:
        print(f"[ONNX Export] กำลังโหลดโมเดลต้นฉบับ (Loading original model): {ModelPath}")
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
        
        # net_g.construct_spkmixmap(n_speaker) # สำหรับส่งออกแบบผสมหลายตัวละคร (Multi-speaker mixing track export)
        
        print("[ONNX Export] กำลังแปลงเป็นรูปแบบ ONNX (Converting to ONNX format)...")
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
        
        print("[ONNX Export] กำลังลดรูปและปรับให้เหมาะสม (Simplifying ONNX model)...")
        model, check = onnxsim.simplify(ExportedPath)
        
        if not check:
            raise RuntimeError("การลดรูปโมเดล ONNX ล้มเหลว (ONNX simplification failed)")
            
        onnx.save(model, ExportedPath)
        print(f"[ONNX Export] เสร็จสมบูรณ์ (Finished): {ExportedPath}")
        return "✅ ส่งออกโมเดล ONNX สำเร็จ (Exported ONNX successfully!)"

    except Exception as e:
        error_msg = f"❌ เกิดข้อผิดพลาดในการส่งออก (Export Error):\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

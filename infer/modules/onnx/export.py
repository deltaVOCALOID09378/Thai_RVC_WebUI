# ==============================================================================
# File: export.py
# Path: infer\modules\onnx\export.py
# Description: RVC WebUI Core Module / โมดูลหลัก RVC WebUI
# Version: 2.4
# Made And Checked By DELTA SYNTH & All Agentic AI
# Original by RVC-Project
# ==============================================================================
import os
import torch
import logging
import traceback

from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

# ตั้งค่าระบบแจ้งเตือนแบบครบวงจร / Configure comprehensive logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def export_onnx(ModelPath, ExportedPath):
    try:
        logger.info("======================================================")
        logger.info(f"Starting ONNX export process / เริ่มต้นกระบวนการส่งออกไฟล์ ONNX")
        logger.info(f"Source Model / ไฟล์ต้นทาง: {ModelPath}")
        logger.info(f"Target Path / ไฟล์ปลายทาง: {ExportedPath}")

        # ตรวจสอบการมีอยู่ของไฟล์ต้นทาง / Verify source file existence
        if not os.path.exists(ModelPath):
            error_msg = f"Model file not found / ไม่พบไฟล์โมเดลที่ระบุ: {ModelPath}"
            logger.error(error_msg)
            return error_msg

        logger.info("Loading PyTorch model / กำลังโหลดข้อมูลโมเดล PyTorch...")
        cpt = torch.load(ModelPath, map_location="cpu")
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768
        
        logger.info(f"Model Version / เวอร์ชันโมเดล: {cpt.get('version', 'v1')} | Vector Channels: {vec_channels}")

        logger.info("Generating dummy inputs for tracing / กำลังสร้างข้อมูลจำลองสำหรับการแกะรอย...")
        test_phone = torch.rand(1, 200, vec_channels)  # hidden unit
        test_phone_lengths = torch.tensor([200]).long()  # hidden unit length
        test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # pitch (Hz)
        test_pitchf = torch.rand(1, 200)  # nsf pitch
        test_ds = torch.LongTensor([0])  # ID
        test_rnd = torch.rand(1, 192, 200)  # noise

        device = "cpu"

        logger.info("Initializing Synthesizer network / กำลังเริ่มต้นโครงข่ายการสังเคราะห์เสียง...")
        net_g = SynthesizerTrnMsNSFsidM(
            *cpt["config"], is_half=False, version=cpt.get("version", "v1")
        )
        net_g.load_state_dict(cpt["weight"], strict=False)
        
        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
        output_names = ["audio"]

        logger.info("Exporting to ONNX format / กำลังส่งออกเป็นรูปแบบไฟล์ ONNX...")
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
            opset_version=13,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
        
        success_msg = f"Successfully exported ONNX to / ส่งออกไฟล์ ONNX เสร็จสมบูรณ์ที่: {ExportedPath}"
        logger.info(success_msg)
        logger.info("======================================================")
        return "Finished"
        
    except Exception as e:
        error_msg = f"Export failed / กระบวนการส่งออกล้มเหลว: {str(e)}"
        logger.error("======================================================")
        logger.error(error_msg)
        logger.error(f"Error Details / รายละเอียดข้อผิดพลาด:\n{traceback.format_exc()}")
        logger.error("======================================================")
        return error_msg
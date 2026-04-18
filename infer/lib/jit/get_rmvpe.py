import os
import logging
from typing import Optional, Any
import torch

# กำหนด Logger สำหรับการแจ้งเตือนสถานะการทำงานอย่างเป็นทางการ
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_rmvpe(
    model_path: str = "assets/rmvpe/rmvpe.pt", device: Optional[torch.device] = None
) -> Any:
    """
    ฟังก์ชันสำหรับโหลดโมเดล RMVPE เพื่อใช้ในการสกัดค่า Pitch (F0)

    Args:
        model_path (str): เส้นทางไปยังไฟล์ Checkpoint ของโมเดล (.pt)
        device (torch.device, optional): อุปกรณ์ประมวลผล หากไม่ระบุระบบจะเลือกให้อัตโนมัติ

    Returns:
        model (E2E): ออบเจกต์โมเดล RMVPE ที่พร้อมใช้งานในโหมดประเมินผล (Evaluation)
    """
    # 1. การบริหารจัดการอุปกรณ์ (Device Management)
    # หากไม่ได้ระบุ device เข้ามา ระบบจะทำการตรวจสอบและเลือกใช้ GPU (CUDA) ให้โดยอัตโนมัติหากมี
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. การตรวจสอบความถูกต้องของเส้นทางไฟล์ (File Verification)
    if not os.path.exists(model_path):
        logger.error(
            f"ไม่พบไฟล์โมเดล RMVPE ที่เส้นทาง: {model_path} กรุณาตรวจสอบความถูกต้อง"
        )
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # 3. การนำเข้าโมดูลแบบหน่วงเวลา (Lazy Import)
        # คงการนำเข้าไว้ภายในฟังก์ชันเพื่อประหยัดหน่วยความจำ (RAM) ในกรณีที่ฟังก์ชันนี้ยังไม่ถูกเรียกใช้งาน
        from infer.lib.rmvpe import E2E

        # 4. การสร้างโครงสร้างโมเดล
        model = E2E(4, 1, (2, 2))

        # 5. การโหลดค่าน้ำหนักอย่างปลอดภัย (Secure Weight Loading)
        # ใช้ weights_only=True เพื่อป้องกันความเสี่ยงจากการรันโค้ดไม่พึงประสงค์ที่อาจแฝงมากับไฟล์ Pickle
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)

        # 6. การเตรียมความพร้อมสำหรับการอนุมาน (Inference Preparation)
        model = model.to(device)
        model.eval()

        logger.info(f"โหลดโมเดล RMVPE สำเร็จและได้รับการติดตั้งลงบนอุปกรณ์: {device}")
        return model

    except ImportError as e:
        logger.error(
            "ไม่สามารถนำเข้าโมดูล infer.lib.rmvpe ได้ กรุณาตรวจสอบโครงสร้างโฟลเดอร์ของโปรเจกต์"
        )
        raise e
    except RuntimeError as e:
        logger.error(
            f"เกิดข้อผิดพลาดในการประมวลผลของ PyTorch ขณะโหลดค่าน้ำหนัก: {str(e)}"
        )
        raise e
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิดขณะโหลดโมเดล RMVPE: {str(e)}")
        raise e

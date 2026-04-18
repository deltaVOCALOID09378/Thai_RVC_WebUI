import torch


def get_synthesizer(pth_path, device=torch.device("cpu")):
    """
    ฟังก์ชันสำหรับโหลดโมเดลสังเคราะห์เสียง (Acoustic Model Loader)
    ทำหน้าที่อ่านไฟล์ .pth และสร้างโครงสร้างโมเดลที่ถูกต้อง (v1/v2, มี/ไม่มี F0)
    เพื่อเตรียมพร้อมสำหรับการแปลงเสียง (Inference)
    """
    # นำเข้าโครงสร้างโมเดลรูปแบบต่างๆ (Import available model architectures)
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,  # v1 (256-dim) แบบมี F0
        SynthesizerTrnMs256NSFsid_nono,  # v1 (256-dim) แบบไม่มี F0
        SynthesizerTrnMs768NSFsid,  # v2 (768-dim) แบบมี F0
        SynthesizerTrnMs768NSFsid_nono,  # v2 (768-dim) แบบไม่มี F0
    )

    # โหลดไฟล์จุดตรวจสอบ (Load checkpoint file to CPU first to prevent VRAM spikes)
    cpt = torch.load(pth_path, map_location=torch.device("cpu"))

    # อัปเดตจำนวนผู้พูด (Update speaker count in config based on embedded weights)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    # อ่านการตั้งค่าจากไฟล์ .pth (Read configurations from checkpoint)
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    # ==========================================
    # เลือกโครงสร้างโมเดลให้ตรงกับไฟล์ที่โหลด (Select corresponding model architecture)
    # ==========================================
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    # ลบส่วน Posterior Encoder ทิ้ง เพราะไม่จำเป็นต้องใช้ตอนแปลงเสียง (ช่วยประหยัด VRAM)
    # (Delete training-only component to save memory during inference)
    del net_g.enc_q

    # โหลดค่าน้ำหนักเข้าสู่โครงสร้าง (Load weights into the instantiated model)
    net_g.load_state_dict(cpt["weight"], strict=False)

    # แปลงชนิดข้อมูลเป็น Float32, ตั้งสถานะเป็นโหมดใช้งานจริง, และย้ายไปรันบนอุปกรณ์ที่ระบุ (GPU/CPU)
    # (Convert to Float32, set to evaluation mode, and move to target device)
    net_g = net_g.float()
    net_g.eval().to(device)

    # ลบ Weight Normalization ออกเพื่อให้ประมวลผลตอนแปลงเสียงได้เร็วขึ้น
    # (Remove weight normalization to speed up inference)
    net_g.remove_weight_norm()

    return net_g, cpt

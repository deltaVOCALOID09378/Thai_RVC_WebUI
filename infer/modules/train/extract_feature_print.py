import os
import sys
import traceback
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device_arg = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 6:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]

# ==========================================
# ระบบเขียน Log (รองรับ UTF-8 เพื่อภาษาไทย)
# ==========================================
log_file_path = "%s/extract_f0_feature.log" % exp_dir


def printt(strr):
    print(strr)
    with open(log_file_path, "a+", encoding="utf-8") as f:
        f.write("%s\n" % strr)


printt(f"[ข้อมูล] อาร์กิวเมนต์ที่รับมา (Arguments): {sys.argv}")

# ==========================================
# ระบบบริหารจัดการฮาร์ดแวร์ (Hardware Management System)
# ==========================================
device = device_arg
if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    try:
        import torch_directml

        # ตรวจสอบว่ามองเห็นฮาร์ดแวร์ DirectML หรือไม่ เพื่อป้องกันการแครช
        if torch_directml.device_count() > 0:
            device = torch_directml.device(torch_directml.default_device())

            def forward_dml(ctx, x, scale):
                ctx.scale = scale
                res = x.clone().detach()
                return res

            fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
        else:
            printt(
                "[คำเตือน] ไม่พบอุปกรณ์ DirectML (No DirectML device found). ระบบจะสลับไปใช้ CPU (Falling back to CPU)."
            )
            device = "cpu"
    except Exception as e:
        printt(
            f"[คำเตือน] ไม่สามารถเรียกใช้ DirectML ได้ (DirectML init failed): {e}. ระบบจะสลับไปใช้ CPU (Falling back to CPU)."
        )
        device = "cpu"

model_path = "assets/hubert/hubert_base.pt"
printt(f"[ข้อมูล] ไดเรกทอรีการทดลอง (Experiment Directory): {exp_dir}")

wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# ฟังก์ชันอ่านไฟล์เสียง (wave must be 16k, hop_size=320)
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# โหลด HuBERT model
printt(f"[ข้อมูล] กำลังโหลดโมเดลเสียงหลัก (Loading HuBERT model from): {model_path}")

# ตรวจสอบว่ามีไฟล์โมเดลอยู่หรือไม่
if os.access(model_path, os.F_OK) == False:
    printt(
        f"[ข้อผิดพลาด] ไม่พบไฟล์โมเดล (Model not found): {model_path} \n"
        f"กรุณาดาวน์โหลดโมเดลก่อนดำเนินการ (Please download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main)"
    )
    exit(0)

models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
printt(f"[ข้อมูล] ย้ายโมเดลไปยังหน่วยประมวลผล (Moving model to device): {device}")

if device not in ["mps", "cpu"]:
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # พิมพ์สถานะประมาณ 10 ครั้งต่อรอบ

if len(todo) == 0:
    printt(
        "[ข้อมูล] ไม่มีไฟล์เสียงให้ประมวลผลในรอบนี้ (No features to extract in this batch)."
    )
else:
    printt(
        f"[ข้อมูล] จำนวนไฟล์ที่ต้องประมวลผลทั้งหมด (Total files to process): {len(todo)}"
    )
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                # ข้ามไฟล์ที่เสร็จแล้ว
                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,  # layer 9 or 12
                }

                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt(f"[คำเตือน] พบค่าความผิดปกติ (NaN detected in): {file}")

                # รายงานความคืบหน้า
                if idx % n == 0 or idx == len(todo) - 1:
                    printt(
                        f"📊 สถานะ (Status): [{idx+1}/{len(todo)}] | ไฟล์ (File): {file} | ขนาด (Shape): {feats.shape}"
                    )
        except:
            printt(
                f"[ข้อผิดพลาด] เกิดปัญหาขณะประมวลผล (Error processing) {file}:\n{traceback.format_exc()}"
            )

    printt("🎉 กระบวนการสกัดคุณลักษณะเสียงเสร็จสมบูรณ์ (All feature extraction done!)")

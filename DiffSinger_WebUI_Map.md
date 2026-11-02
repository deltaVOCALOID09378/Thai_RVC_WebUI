<!-- 
==============================================================================
File: DiffSinger_WebUI_Map.md | Version: 1.0.0
Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
Original by Patiphat Wongyai (Delta), and OpenVPI DiffSinger
Description: Detailed Directory Map & Architecture Overview for DiffSinger UI
==============================================================================
-->

# 🎵 THE DIFFSINGER TRAINER UI — DETAILED ARCHITECTURE MAP
## แผนผังและสถาปัตยกรรมระบบ DiffSinger 2.5.1 WebUI (Gradio)

เอกสารฉบับนี้คือ **แผนผังโครงสร้างการทำงาน (Directory Map)** และสรุปการทำงานของโมดูลย่อยในระบบ **The Diff Singer Trainer UI** โดยระบบนี้ถูกออกแบบมาเพื่ออำนวยความสะดวกในการจัดเตรียมข้อมูล (Data Preparation), เทรนโมเดล (Acoustic/Variance Training), ส่งออกโมเดลเป็น ONNX, และแพ็คไฟล์สำหรับ OpenUtau แบบเบ็ดเสร็จในหน้าต่างเดียว (One-Click Pipeline)

---

## 1. 📂 โครงสร้างโฟลเดอร์หลัก (Root Directory Structure)
เส้นทางอ้างอิง: `A:\Thai_RVC_WebUI_v1\DiffSinger-2.5.1\`

```text
DiffSinger-2.5.1/
├── 📄 app.py                  # [MAIN] ระบบหลักของ WebUI (Gradio Interface) - คุมทุกฟังก์ชัน
├── 📄 auto_pipeline.py        # สคริปต์อัตโนมัติ 1-Click (Train + Export) รองรับการรันผ่าน CMD
├── 📄 run_pipeline.bat        # สคริปต์ Batch File เรียกใช้งาน auto_pipeline.py สะดวกขึ้น
├── 📄 requirements.txt        # รายการ Dependencies หลักของโปรเจกต์ (PyTorch, Lightning, etc.)
├── 📄 requirements-onnx.txt   # รายการ Dependencies สำหรับการ Export ONNX
├── 📄 README.md               # คู่มือการใช้งานดั้งเดิมของ OpenVPI
├── 📁 configs/                # [CORE] โฟลเดอร์จัดเก็บเทมเพลตค่าพื้นฐานสำหรับการเทรน
│   ├── 📄 acoustic.yaml       # ค่าเริ่มต้นโมเดล Acoustic (สร้างเสียง)
│   └── 📄 variance.yaml       # ค่าเริ่มต้นโมเดล Variance (คุมระดับเสียง, ระยะเวลา, ฯลฯ)
├── 📁 data/                   # [WORKSPACE] พื้นที่จัดเก็บชุดข้อมูลสำหรับแต่ละโปรเจกต์
│   └── 📁 <project_name>/     # โฟลเดอร์ของโปรเจกต์ (เช่น my_voicebank)
│       ├── 📄 project_meta.json      # ข้อมูล Meta (ชื่อผู้พูด, ภาษา, วันที่สร้าง)
│       ├── 📄 config_acoustic.yaml   # Config เฉพาะสำหรับโปรเจกต์นี้ (Acoustic)
│       ├── 📄 config_variance.yaml   # Config เฉพาะสำหรับโปรเจกต์นี้ (Variance)
│       ├── 📄 custom_dictionary.txt  # พจนานุกรมคำอ่านเฉพาะ (สร้างอัตโนมัติจาก transcriptions)
│       ├── 📁 raw/                   # ข้อมูลดิบที่ผู้ใช้อัปโหลด
│       │   ├── 📄 transcriptions.csv # คำอ่านเสียงแต่ละไฟล์ (name, ph_seq, ph_dur)
│       │   └── 📁 wavs/              # ไฟล์เสียง .wav หรือ .flac
│       ├── 📁 binary/                # ชุดข้อมูลที่ผ่าน Binarization (พร้อมเทรน)
│       └── 📁 export/                # ไฟล์ผลลัพธ์ที่ Export (ONNX, yaml, txt) เพื่อแพ็ค
├── 📁 checkpoints/            # [OUTPUT] พื้นที่จัดเก็บโมเดลที่เทรน (Checkpoints)
│   ├── 📁 rmvpe/              # [PRETRAINED] โมเดลสกัด Pitch (RMVPE)
│   │   └── 📄 model.pt
│   ├── 📁 vr/                 # [PRETRAINED] โมเดลสกัด Harmonic-Noise (ถ้ามี)
│   │   └── 📄 model.pt
│   ├── 📁 nsf_hifigan/        # [PRETRAINED] โมเดล Vocoder เริ่มต้น
│   │   └── 📄 model.ckpt
│   └── 📁 <exp_name>/         # Checkpoints ที่เทรนได้แยกตามชื่อ Experiment
│       ├── 📄 config.yaml
│       └── 📄 model_ckpt_steps_XXXXXX.ckpt 
├── 📁 dictionaries/           # [CORE] พจนานุกรมคำอ่านพื้นฐาน
│   └── 📄 opencpop-extension.txt # Dictionary มาตรฐาน (ภาษาจีนและทั่วไป)
├── 📁 scripts/                # [CORE] สคริปต์ประมวลผลดั้งเดิมของ DiffSinger
│   ├── 📄 binarize.py         # ประมวลผลและแปลงข้อมูลดิบ (Binarization)
│   ├── 📄 train.py            # สคริปต์รันเทรน PyTorch Lightning
│   └── 📄 export.py           # สคริปต์ Export โมเดล ONNX
└── 📁 .venv/                  # Virtual Environment (ถ้าใช้)
    └── 📁 Scripts/
        └── 📄 python.exe      # Python Executable หลัก
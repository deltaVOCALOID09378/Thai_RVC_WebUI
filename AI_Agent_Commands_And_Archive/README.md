# AI Agent Commands & Inactive Files Archive
<!-- Made And Checked By DELTA SYNTH & All Code Agentic AI Engine -->
<!-- Date: 19/10/2026 | Version: 2.1 -->

แฟ้มรวบรวมไฟล์สำหรับรันคำสั่งงานจาก Agent AI และไฟล์ที่ไม่ถูกเรียกใช้งานในโค้ดระบบหลัก เพื่อจัดระเบียบไดเรกทอรีราก (Root Directory) ของโครงการ Thai_RVC_WebUI_v1 ให้กระชับ รวดเร็ว และเป็นระเบียบเรียบร้อย ตามข้อกำหนดใน [AGENTS.md](file:///a:/Thai_RVC_WebUI_v1/AGENTS.md) และ [AGENT.md](file:///a:/Thai_RVC_WebUI_v1/AGENT.md)

---

## 📁 โครงสร้างภายในแฟ้ม (Folder Structure)

```
AI_Agent_Commands_And_Archive/
├── Command_Runners/          # รวมไฟล์สคริปต์และตัวรันคำสั่งเฉพาะกิจของ Agent AI ทั้งหมด
│   ├── build_package_fallback.ps1
│   ├── Check-And-Download-Tools.bat
│   ├── Copy_Thai_RVC_WebUI_v1.bat
│   ├── dataset_auto_cleaner.py
│   ├── diffsinger_suite_optimizer.py
│   ├── DiffSinger_Trainer_v1.0.py
│   ├── fix_diffsinger_config.py
│   ├── Fix_ONNX_Runtime.bat
│   ├── go-realtime-gui-dml.bat
│   ├── go-realtime-gui.bat
│   ├── Install-Env.bat
│   ├── run.sh
│   ├── run_autonomous_trainer.bat
│   ├── run_binarize.bat
│   ├── run_config_binarize.bat
│   ├── Run_DiffSinger_Pipeline_Master.bat
│   ├── Run_DiffSinger_Trainer_v1.0.bat
│   ├── run_training_master.bat
│   ├── run_unified_studio.bat
│   ├── Start-Program-DML.bat
│   ├── Start-ThaiRVC-DesktopStudio.bat
│   ├── Train_Studio_Runner.bat
│   ├── unified_vocal_studio_core.py
│   ├── venv.sh
│   └── verify_and_route_system.bat
│
└── Unused_Files/             # รวมไฟล์เวอร์ชันเก่าและไฟล์ที่ไม่ได้ใช้งานในระบบโค้ดแกนหลัก
    ├── About_Project.md
    ├── All File of DiffSinger.md
    ├── All RVC Original Code.md
    ├── All Thai RVC WebUI.md
    ├── All Thai RVC WebUI.txt
    ├── audit_report.md
    ├── BIGTABLE_SCHEMA_SPEC.md
    ├── config.yaml
    ├── docker-compose.yml
    ├── Dockerfile
    ├── environment_dml.yaml
    ├── fix_onnxruntime.bat (v1.0 CPU-only)
    ├── gui_v1.py
    ├── infer-web.py
    ├── Install_Update_v2.1.0_To_PC.bat
    ├── Location.md
    ├── MIT协议暨相关引用库协议
    ├── Pack-Full-Release-7z.bat (v1.3)
    ├── Pack-Source-Code-Zip.bat (v1.3)
    ├── Program_Backup.py
    ├── Retrieval-based-Voice-Conversion-WebUI-2.2.231006/
    ├── RVC WebUI โค้ดเปล่าๆ ต้นฉบับดั้งเดิม.zip
    ├── SPARK_THAI_RVC_BLUEPRINT.md
    ├── Start-Program.bat
    ├── Start_The_Program.bat (v1.9)
    ├── train-index.py (v1 256-dim)
    └── Version_no_2.1.0/
```

---

## 📝 คำอธิบายหมวดหมู่

1. **`Command_Runners/`**
   - ประกอบด้วยสคริปต์ตรวจสอบ วิเคราะห์ และตัวรันงานเฉพาะกิจที่ยังคงใช้งานร่วมกับระบบหรือเป็นตัวเสริมสำรอง เช่น `build_package_fallback.ps1` (ระบบสำรองบีบอัด .NET)
   - สคริปต์ Batch และ Shell สำหรับรันเทรน, รัน Binarize, รัน GUI แยก และ Export ข้อมูลภายนอก

2. **`Unused_Files/`**
   - ประกอบด้วยไฟล์เวอร์ชันเก่าที่ถูกแทนที่ด้วยเวอร์ชันใหม่ เช่น:
     - `Version_no_2.1.0/` (ถูกแทนที่ด้วย `Thai_RVC_WebUI_v1_Distribution_*.zip`)
     - `Pack-Full-Release-7z.bat` และ `Pack-Source-Code-Zip.bat` (ถูกแทนที่ด้วย `Package_Program.bat` v2.1)
     - `Start-Program.bat` และ `Start_The_Program.bat` (ถูกแทนที่ด้วย `Start-The-Program.bat` v4.7)
     - `train-index.py` (ถูกแทนที่ด้วย `train-index-v2.py`)
     - `fix_onnxruntime.bat` (ถูกแทนที่ด้วย `Fix_ONNX_Runtime.bat`)
     - `Retrieval-based-Voice-Conversion-WebUI-2.2.231006/` (รุ่นเก่าที่ไม่ได้ใช้งาน)
   - ประกอบด้วยเอกสาร Markdown ดัมพ์ขนาดใหญ่ (เช่น `All File of DiffSinger.md`, `All RVC Original Code.md`, `All Thai RVC WebUI.md`)
   - ประกอบด้วยโค้ดสำรองและไฟล์คอนฟิกภายนอกที่ไม่ได้ใช้รันระบบหลัก

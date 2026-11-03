# รายงานสรุปประวัติการพัฒนาและแก้ไขโค้ด: Thai RVC & DiffSinger AI (DELTA SYNTH)
**ผู้รับผิดชอบโครงการ:** Patiphat Wongyai (ท่านเดลต้า) / DELTA SYNTH  
**ทีมวิศวกรรมและผู้ดำเนินการร่วม:** DELTA SYNTH Agent, Manus AI, Gemini AI  
**มาตรฐานทางวิศวกรรม:** DELTA SYNTH Engineering Protocol (`Preserve` → `Strengthen` → `Optimize` → `Verify`)  
**สัญญาอนุญาต:** MIT License (Copyright 2024 RVC Project Developers / Copyright 2026 Patiphat Wongyai - DELTA SYNTH)  
**วันที่จัดทำรายงาน:** 3 พฤศจิกายน 2026  

---

## 1. บทสรุปผู้บริหาร (Executive Summary)

รายงานฉบับนี้รวบรวมประวัติการแก้ไขโค้ด การจัดวางสถาปัตยกรรม และการปรับแต่งระบบทั้งหมดของโครงการ **Thai_RVC_WebUI** และ **DELTA_SYNTH_Studio (RVC & DiffSinger Unified Engine)** ที่บันทึกไว้ในบัญชีนี้

การพัฒนาทั้งหมดมุ่งเน้นไปที่ 3 แกนหลัก:
1. **การปรับระบบให้รองรับภาษาไทยอย่างสมบูรณ์ (Thai Localization & G2P Integration):** ทั้งในส่วน UI, ข้อความแจ้งเตือน, การประมวลผลไฟล์ และระบบพจนานุกรมหน่วยเสียงสำหรับ DiffSinger
2. **การแก้ไขปัญหาสภาพแวดล้อมรันไทม์ (Runtime & Dependency Management):** การแก้ปัญหาคอขวดของ `fairseq`, เส้นทาง Environment ข้ามแพลตฟอร์มบน Windows และระบบตรวจจับ/กู้คืนอัตโนมัติ (Self-Healing / Auto-Repair) ภายใต้หลักการ **Surgical Edits** (ไม่แก้ไขกระทบตรรกะหลักของโมเดลเดิม)
3. **การรวมศูนย์เทคโนโลยีประมวลผลเสียง (Unified Audio Engineering Pipeline):** การบูรณาการกระบวนการสกัดเสียง (PyMSS), การแปลงเสียง (RVC), และการสังเคราะห์เสียงร้อง (DiffSinger สำหรับ OpenUtau) ให้เชื่อมโยงกันอย่างเป็นระบบ

---

## 2. ตารางบันทึกประวัติการแก้ไขโค้ดและไฟล์ระบบ (Code Modification Log)

| หมวดหมู่ | รายการไฟล์ที่ดำเนินการ | สถานะการทำงาน | ผู้ดำเนินการ / รุ่น | การเปลี่ยนแปลงตรรกะและรายละเอียดทางเทคนิค (Logic Altered) |
| :--- | :--- | :--- | :--- | :--- |
| **Core & Engine** | `Program.py`<br>`infer/vc/modules.py` | ปรับปรุงฟีเจอร์ | DELTA SYNTH Agent<br>*(v1.8.5 - 2026-08-31)* | • เพิ่มระบบ **Smart Resume** ข้ามไฟล์ที่ประมวลผลเสร็จแล้วอัตโนมัติในการรัน Batch Inference (`vc_multi`)<br>• เพิ่มระบบ **Global Cancellation** สั่งหยุดชะงักงานได้แบบเรียลไทม์ (`cancel_batch`)<br>• ปรับปรุงระบบแจ้งเตือนแบบ A11Y บน UI ธีม DELTA SYNTH |
| **Localization & DSP** | `infer/lib/jit/get_rmvpe.py`<br>`infer/modules/train/preprocess.py`<br>`infer/lib/train/process_ckpt.py`<br>`infer/modules/train/extract_feature_print.py` | แปลและปรับแต่ง | DELTA SYNTH<br>*(2026-04-18 ถึง 2026-06-28)* | • เพิ่ม Logging ภาษาไทยในโมดูล RMVPE<br>• แปลข้อความสถานะการทำงานของฟังก์ชันบันทึก (`savee`), แก้ไขข้อมูลโมเดล (`change_info`), และรวมโมเดล (`merge`)<br>• ปรับ `preprocess.py` รองรับ Multiprocessing บน Windows อย่างปลอดภัย<br>• ปรับปรุง `extract_feature_print.py` ให้บันทึก Log แบบ UTF-8 และจัดการ Fallback ฮาร์ดแวร์ DirectML / CPU |
| **Installer** | `Install_Environment.bat` | ยกเครื่องใหม่<br>*(v3.0)* | Manus AI | • ลำดับการเลือกใช้ `runtime\python.exe` ก่อน แล้วจึง Fallback ไปใช้ Python จาก System PATH<br>• ติดตั้ง `fairseq==0.12.2` ก่อนเป็นลำดับแรกเพื่อป้องกัน `ModuleNotFoundError`<br>• กำหนดกลไก Retry การดาวน์โหลด 5 ครั้ง (Timeout 120 วินาที)<br>• บันทึก Log ละเอียดลง `logs\Install_Environment.log`<br>• บันทึก Manifest เส้นทางระบบจริงลง `logs\Thai_RVC_WebUI_Paths.txt` |
| **Launcher** | `go-web.bat`<br>`go-web-dml.bat` | ปรับปรุงระบบ | DELTA SYNTH / Manus AI | • บังคับใช้ `chcp 65001` (UTF-8) รองรับการแสดงผลภาษาไทยบน Command Prompt<br>• กำหนดค่าพอร์ตการรัน WebUI: พอร์ต `7897` (และ `9378` สำหรับ DirectML)<br>• เพิ่มตรรกะตรวจสอบ `fairseq` ก่อนเริ่มโปรแกรม หากขาดจะทำการติดตั้งทันที ป้องกันโปรแกรมแครช |
| **Auto-Repair** | `Start_WebUI_AutoRepair.bat` | สร้างไฟล์ใหม่ | Manus AI | • รองรับการลากโฟลเดอร์โปรเจกต์มาวางเพื่อรัน<br>• ตรวจสอบ Health Check ของระบบก่อนเริ่ม: `fairseq`, `Dictionary`, `grad_multiply`, `Torch`, และ `Gradio`<br>• ซ่อมแซมสภาพแวดล้อมอัตโนมัติหากพบไฟล์ขาดหาย พร้อมรัน `pip check`<br>• บันทึกประวัติการกู้คืนลง `logs\WebUI_AutoRepair.log` |
| **Runtime Diagnostics** | `Run_Program.bat`<br>`Repair_Fairseq.bat` | สร้างไฟล์ใหม่<br>*(v1.1)* | Manus AI | • `Run_Program.bat`: สร้างทางเข้ารันตรงสำหรับ `Program.py` พร้อมตรวจ Pre-flight Dependencies<br>• `Repair_Fairseq.bat`: เครื่องมือซ่อมแซมโมดูล Fairseq และแก้ปัญหา Path formatting (`logs\Repair_Fairseq.log`) พร้อมระบบ Fallback ไปยัง `%TEMP%` เมื่อไฟล์ถูกล็อก |
| **System Audit** | `Check_Thai_RVC_Status.bat`<br>`check_thai_rvc_status.py` | สร้างไฟล์ใหม่ & แก้ไขบั๊ก Path | Manus AI | • เครื่องมือตรวจสอบระบบแบบ Read-only ตรวจไฟล์, Weights, CUDA, Port 7897, และ Process บน Windows<br>• จัดกลุ่มผลการตรวจสอบเป็น `PASS`, `INFO`, `WARN`, `FAIL` พร้อมส่งออกรายงาน Markdown และ JSON<br>• **Bug Fix:** แก้ปัญหา Windows Path Syntax ที่ทำให้เส้นทางกลายเป็น `A:\Thai_RVC_WebUI"\logs` จากการมี Quote เกิน |
| **Documentation** | `Thai_RVC_WebUI_User_Guide.md`<br>`README.md`<br>`CONTRIBUTING.md` | สร้างใหม่ & แปลภาษา | DELTA SYNTH / Manus AI | • จัดทำคู่มือผู้ใช้แบบสองภาษา (ไทย-อังกฤษ) อธิบายตั้งแต่ขั้นตอนการเตรียมไฟล์เสียง การใช้งาน Installer จนถึงการรัน WebUI<br>• แปลเอกสารคำแนะนำการเทรน, Faiss Tuning Tips, และกฎระเบียบการร่วมสมทบ (Contributing Rules) |

---

## 3. สถาปัตยกรรมระบบรวม DELTA SYNTH Studio (RVC & DiffSinger Unified Engine)

โครงการได้วางรากฐานโครงสร้างโฟลเดอร์และสายพานการทำงาน (Pipeline) เพื่อรองรับทั้งการแปลงเสียง (Voice Conversion) และการสร้างชุดเสียงร้องเพลง AI (Singing Voice Synthesis) ไว้ดังนี้:

```text
DELTA_SYNTH_Studio/
├── Start_WebUI_v1.0.bat             # ทางเข้าหลักสำหรับเปิดระบบ (รองรับการลากวาง)
├── Install_Environment.bat          # ระบบติดตั้งสภาพแวดล้อมอัตโนมัติ
├── WebUI_v1.0.py                    # ส่วนต่อประสานผู้ใช้ Gradio หลัก
├── requirements_unified.txt         # ไฟล์รวม Dependency ของ RVC และ DiffSinger
├── assets/
│   ├── hubert/                      # Hubert Base Feature Extractor (~350 MB)
│   ├── rmvpe/                       # โมเดล Pitch Extraction ความแม่นยำสูง (RMVPE)
│   ├── uvr5_weights/                # โมเดลสำหรับแยกเสียงร้อง/ดนตรี (UVR5 / PyMSS)
│   ├── pretrained_v1/               # Base Checkpoints RVC v1 (256-dim)
│   └── pretrained_v2/               # Base Checkpoints RVC v2 (768-dim)
├── dictionaries/                    # พจนานุกรมหน่วยเสียงภาษาไทย-สากล (G2P)
│   ├── thai_vccv_dict.txt           # ระบบ Thai VCCV สำหรับ DiffSinger
│   ├── thai_arpasing.txt            # ระบบ Thai Arpasing
│   └── english_dict.txt             # ระบบภาษาอังกฤษมาตรฐาน
├── dataset/                         # สายพานการเตรียมข้อมูลเสียง (Audio Pipeline)
│   ├── 00_raw_audio/                # คลังไฟล์เสียงดิบ
│   ├── 01_cleaned_vocals/           # เสียงร้องสะอาดหลังตัดเสียงรบกวน
│   ├── UTAU_Ready/                  # ชุดข้อมูล UTAU (*.wav + oto.ini)
│   └── DiffSinger_Ready/            # ชุดข้อมูล DiffSinger (*.wav + *.lab หลังทำ Forced Alignment)
├── logs/
│   ├── RVC_Projects/                # เช็คพอยต์การเทรน RVC (G_*.pth, D_*.pth, added_*.index)
│   └── DiffSinger_Projects/         # เช็คพอยต์ DiffSinger (acoustic_ckpt/, variance_ckpt/)
├── weights/
│   ├── RVC_Weights/                 # โมเดล RVC สำเร็จรูป (.pth และ .onnx)
│   └── DiffSinger_Weights/          # โมเดล DiffSinger พร้อมใช้งาน (.onnx)
└── UTAU_Export/                     # โมเดล Voicebank สำเร็จรูปสำหรับ OpenUtau
    └── {Singer_Name}_{Language}_DiffSinger/
        ├── character.txt            # รายละเอียดตัวละครและเงื่อนไขการใช้งาน
        ├── character.yaml           # การตั้งค่า Voicebank และ UI ใน OpenUtau
        ├── dsconfig.yaml            # การเชื่อมโยง Acoustic/Variance Model
        ├── dsdict.txt               # พจนานุกรมเทียบคำอ่าน
        ├── acoustic.onnx            # โมเดลเนื้อเสียง
        ├── variance.onnx            # โมเดลควบคุม Pitch และ Duration
        └── Picture/ (Icon.png, Full.png)
```

---

## 4. มาตรฐานทางวิศวกรรมและการควบคุมความเสี่ยง (Engineering Protocols & Residual Risks)

### 4.1 แนวทางการแก้ไขแบบเฉพาะเจาะจง (Surgical Edits Rule)
* **Preserve Before Replace:** รักษาส่วนประกอบและโครงสร้างเดิมที่ทำงานได้ดีไว้เสมอ ไม่แก้ไขหรือลบลอจิกของโมเดล `infer/`, โค้ดอัลกอริทึมต้นฉบับ หรือไลบรารี Gradio โดยไม่จำเป็น
* **Direct Environment Binding:** สคริปต์ Batch ทั้งหมด (`go-web.bat`, `Install_Environment.bat`, `Start_WebUI_AutoRepair.bat`) ถูกกำหนดให้เรียกใช้ `runtime\python.exe -m pip` หรือคำสั่งเจาะจงที่ Interpreter เพื่อป้องกันปัญหาการติดตั้งแพ็กเกจข้าม Environment

### 4.2 ความเสี่ยงคงค้างและข้อควรระวัง (Residual Risks)
1. **ข้อจำกัดของ Dependency เก่าบน Python เวอร์ชันใหม่:** ไลบรารีอย่าง `fairseq==0.12.2`, `numba==0.56.4` และ `llvmlite==0.39.0` มีข้อกำหนดเรื่องเวอร์ชันของ Compiler สูงมาก **ห้ามลดหรือเพิ่มเวอร์ชันของ `setuptools` แบบสุ่ม** หากเกิดปัญหา ให้ตรวจสอบสาเหตุจาก `logs\Install_Environment.log` หรือ `logs\Repair_Fairseq.log` เป็นหลัก
2. **การแจ้งเตือนจาก Gradio:** ข้อความแจ้งเตือน Deprecation Warning เกี่ยวกับ `pkg_resources` บน Gradio เป็นพฤติกรรมปกติของเวอร์ชันดังกล่าว ไม่ส่งผลต่อการทำงานของระบบ และไม่ควรแก้ด้วยการเปลี่ยนเวอร์ชันไลบรารีเพราะอาจกระทบความเข้ากันได้
3. **สภาพแวดล้อมระดับระบบปฏิบัติการ:** ปัญหาการล็อกไฟล์ Log หรือข้อจำกัดสิทธิ์การเขียนในไดเรกทอรีโปรเจกต์บน Windows จำเป็นต้องได้รับการดูแลเรื่อง Permissions ให้สิทธิ์อ่านเขียนอย่างถูกต้องแก่ผู้ใช้งาน
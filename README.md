<!-- 
==============================================================================
Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
Original by Patiphat Wongyai (Delta), RVC-Project, and OpenVPI DiffSinger
Version: 3.1.0
Description: Comprehensive Research & Development Report with My Mapping
==============================================================================
-->

# DELTA SYNTH — Thai RVC Studio & DiffSinger Unified Engine
## รายงานวิจัยและพัฒนาเชิงระบบ พร้อมแผนผังตำแหน่งไฟล์มาตรฐาน (Research & Development Report & My Mapping)

> **Architectural Standard:** Enterprise Zero-Defect (`Preserve → Strengthen → Optimize → Verify`)  
> **Project Ownership:** Made And Checked By DELTA SYNTH & All Code Agentic AI Engine  
> **Original Creators:** Patiphat Wongyai (Delta), RVC-Project, and OpenVPI DiffSinger  
> **Target Release:** Version no. 3.1.0 (Unified Enterprise Suite)  
> **Execution Root:** `A:\Thai_RVC_WebUI_v1`  
> **Default Port:** `9378` (Dedicated Port)  

---

## 🔬 1. รายงานวิจัยและพัฒนา (Research & Development Report)

### 1.1 บทนำและปัญหาทางวิทยาศาสตร์เสียง (Scientific Background)
ภาษาไทยเป็นภาษาวรรณยุกต์ (Tonal Language) ที่มี 5 ระดับเสียงวรรณยุกต์ ได้แก่ สามัญ (Mid), เอก (Low), โท (Falling), ตรี (High), และจัตวา (Rising) ความถี่มูลฐาน (Fundamental Frequency หรือ F0) มีบทบาททางสัทศาสตร์ในการจำแนกความหมายของคำ หากโมเดลแปลงเสียง (Voice Conversion) สกัดหรือสร้างเส้นเสียง F0 คลาดเคลื่อน จะทำให้เกิด **Semantic Drift** (ความหมายคำเปลี่ยน) ทันที

นอกจากนี้ การออกเสียงสระไทยมีระบบความยาวสระ (Vowel Quantity) ทั้งสระสั้นและสระยาว เช่น "ปะ" vs "ปา" ซึ่งต้องการความต่อเนื่องของ Temporal Acoustic Frames ที่เที่ยงตรง และระบบเสียงภาษาไทยยังมีเสียงพยัญชนะเสียดแทรก (Voiceless Fricatives / Affricates) ได้แก่ /s/ (ส, ซ, ศ, ษ) และ /tɕʰ/ (ฉ, ช) ที่มักเกิดปัญหาแตกพร่า (Acoustic Distortion) เมื่อผ่านกระบวนการ Neural Vocoder ทั่วไป

### 1.2 สถาปัตยกรรมเทคนิคการแก้ปัญหา (Technical Methodology)
ระบบ **Thai RVC Studio & DiffSinger Unified Engine** ได้รับการพัฒนาเพื่อแก้ปัญหาดังกล่าวด้วยกลไกทางวิศวกรรม 5 มิติ:

1. **Parallel Signal Path:** แยกเส้นทางสัญญาณเสียงต้นทางเป็น 2 ทางคู่ขนาน:
   - **Content Stream:** สกัดลักษณะเด่นของเสียงด้วย HuBERT / ContentVec (768 มิติ) ที่ผ่านการจัดระเบียบโครงสร้างตามหน่วยเสียงไทย
   - **Pitch Stream:** สกัด F0 ด้วยอัลกอริทึมความแม่นยำสูง (RMVPE, FCPE, Harvest) พร้อมเสริม STFT Padding Guard เพื่อป้องกันข้อผิดพลาดกรณีไฟล์เสียงสั้นกว่า 1024 ตัวอย่าง
2. **Dynamic Thai De-Esser:** วงจรกรองเสียงเสียดแทรกแบบปรับตัวอัตโนมัติในช่วงความถี่วิกฤต **4.5 kHz – 7.5 kHz** ช่วยลดความกระด้างของพยัญชนะเสียดแทรกภาษาไทยโดยไม่ทำให้เสียงร้องสูญเสียความโปร่งใส
3. **50Hz Butterworth High-Pass Filter:** ขจัดความถี่ต่ำ (Rumble noise) ใต้ 50 Hz ซึ่งเป็นสาเหตุของความขุ่นมัวและกิน Dynamic Range ของตัวแปลงสัญญาณ
4. **EBU R128 Loudness Normalization:** ปรับสมดุลระดับความดังมาตรฐานสากลที่ **-16 LUFS** และคุม True Peak ไว้ที่ **0.98** ป้องกันเสียงแตกพร่า (Clipping) เมื่อนำไปเปิดบนลำโพงหรือแพลตฟอร์มสตรีมมิง
5. **Interactive Native Pickers & Robust Serialization:** ติดตั้ง Tkinter Topmost Native Dialogs เพื่อเลือกไฟล์/โฟลเดอร์บน WebUI โดยตรง และเสริมความปลอดภัยในการแปลงชนิดข้อมูล `sid` (int conversion) ป้องกันข้อผิดพลาด `TypeError` ขณะแปลงเสียง

---

## 🗺️ 2. แผนผังตำแหน่งไฟล์มาตรฐาน (My Mapping)

โครงสร้างโฟลเดอร์และไฟล์ทั้งหมดใน `A:\Thai_RVC_WebUI_v1` ถูกจัดระเบียบตามมาตรฐาน Canonical Zero-Defect ห้ามวางไฟล์ซ้ำซ้อนหรือกระจายออกนอกโครงสร้างที่กำหนด:

```text
A:\Thai_RVC_WebUI_v1\
│
├── 🚀 Core Launchers & Master Interfaces
│   ├── Program.py                                  # ศูนย์กลางควบคุม 12 แท็บหลัก DELTA RVC Studio x DiffSinger (v4.6)
│   ├── Program_Backup.py                           # ไฟล์สำรอง Program.py ป้องกันการสูญหาย
│   ├── infer-web.py                                # หน้าต่าง WebUI มาตรฐาน Gradio (v2.5)
│   ├── realtime_gui.py                             # หน้าต่าง Real-Time Voice Changer GUI (FreeSimpleGUI)
│   └── DiffSinger_Trainer_v1.0.py                  # โปรแกรมฝึกโมเดล DiffSinger Acoustic & Variance GUI
│
├── 📜 Batch Execution Scripts (100% English CLI & Drag-and-Drop)
│   ├── Start-The-Program.bat                       # เปิดโปรแกรมหลักบนพอร์ต 9378 (v1.6)
│   ├── Check-And-Download-Tools.bat                # ตรวจสอบและดาวน์โหลด Checkpoint โมเดลพื้นฐานอัตโนมัติ (v1.0)
│   ├── Pack-Full-Release-7z.bat                    # แพ็กเกจระบบแบบสมบูรณ์ (.7z) สำหรับเผยแพร่ (v1.3)
│   ├── Pack-Source-Code-Zip.bat                    # แพ็กเกจเฉพาะซอร์สโค้ดและคอนฟิก (.zip) (v1.3)
│   ├── Train_Studio_Runner.bat                     # เปิดระบบฝึกโมเดล CLI รองรับ Drag & Drop โฟลเดอร์ชุดข้อมูล
│   ├── run_training_master.bat                     # รัน TrainPipelineMaster แบบ DDP Port 9379
│   ├── Run_DiffSinger_Trainer_v1.0.bat             # รันหน้าต่าง DiffSinger Trainer GUI
│   ├── Install-Env.bat                             # ติดตั้งและตรวจสอบความพร้อมของ Python Environment
│   ├── Install-File.bat                            # ติดตั้งโมเดลและไฟล์ส่วนขยายพร้อมแถบแสดงความคืบหน้า
│   ├── Copy_Thai_RVC_WebUI_v1.bat                  # สำรองข้อมูลทั้งโปรเจกต์ด้วยคำสั่ง Robocopy ที่ปลอดภัย
│   └── Install_Update_v2.1.0_To_PC.bat             # ติดตั้งชุดอัปเดตลงเครื่องและตรวจสอบ py_compile อัตโนมัติ
│
├── 🧠 Central Utilities & Data Cleaning
│   ├── utils.py                                    # คลาส PoliteLogger, Smart Index Matcher, Resource Verifier
│   └── dataset_auto_cleaner.py                     # Universal Auto-Purge Sentinel ลบ desktop.ini / thumbs.db / ขยะ
│
├── 📚 Architectural Blueprint & Documentation
│   ├── About_Project.md                            # แผนงานโครงการฉบับเต็ม 12 หัวข้อสากล และ Multi-Agent Roadmap (v3.1.0)
│   ├── README.md                                   # รายงานวิจัยและพัฒนา พร้อม My Mapping (ไฟล์นี้)
│   ├── SPARK_THAI_RVC_BLUEPRINT.md                 # คัมภีร์กรอบการพัฒนาและกฎเหล็กสำหรับ AI Agent พันธมิตร (v3.0.0)
│   ├── AGENT.md / AGENTS.md                        # ข้อกำหนดทางวิศวกรรมซอฟต์แวร์ DELTA SYNTH
│   ├── audit_report.md                             # รายงานการตรวจสอบระบบและประวัติการแก้บั๊ก Zero-Defect
│   ├── Location.md                                 # ตำแหน่งจัดเก็บไฟล์จริง ไดรฟ์สำรอง และ GitHub Repository
│   ├── All Documents/                              # รวบรวมคู่มือการติดตั้งและการใช้งานอย่างละเอียด
│   └── history/                                    # โฟลเดอร์จัดเก็บประวัติการทำงานทุกรอบ
│
├── ⚙️ Configuration & Architecture Specs
│   └── configs/
│       ├── config.py                               # ตัวตรวจจับฮาร์ดแวร์ CUDA, DirectML, IPEX, MPS, CPU
│       ├── v1/ {32k, 40k, 48k}.json                # คอนฟิกโมเดลสถาปัตยกรรม RVC v1 (256-dim)
│       ├── v2/ {32k, 40k, 48k}.json                # คอนฟิกโมเดลสถาปัตยกรรม RVC v2 (768-dim)
│       └── inuse/                                  # คอนฟิกที่กำลังถูกโหลดใช้งานแบบ Real-time
│
├── 🌐 Internationalization (i18n)
│   └── i18n/
│       ├── i18n.py / scan_i18n.py                  # ตัวแปลภาษาและสแกนคีย์
│       └── locale/                                 # ไฟล์พจนานุกรมแปลภาษา (th_TH, en_US, es_ES, ja_JP)
│
├── 🛠️ Auxiliary Tools & Converters
│   └── tools/
│       ├── export_onnx.py / onnx_inference_demo.py # ระบบแปลงและรันโมเดลด้วย ONNX Runtime
│       ├── calc_rvc_model_similarity.py            # คำนวณความคล้ายคลึงระหว่าง Checkpoint
│       ├── cuda_graph.py                           # เร่งความเร็วการประมวลผลด้วย NVIDIA CUDA Graph
│       ├── infer_batch_rvc.py / infer_cli.py       # ระบบแปลงเสียงผ่าน Command Line แบบเดี่ยวและกลุ่ม
│       └── test_system_smoke.py                    # ชุดทดสอบ Smoke Test ยืนยันความพร้อมของระบบ
│
├── 🎛️ Neural DSP & Inference Pipeline Core
│   └── infer/
│       ├── lib/
│       │   ├── audio.py                            # ระบบโหลดและเขียนไฟล์เสียงความเที่ยงตรงสูง
│       │   ├── slicer2.py                          # ตัดแบ่งท่อนเสียงแบบตรวจจับความเงียบ
│       │   ├── rmvpe.py                            # แกนกลางโมเดล RMVPE Pitch Extraction พร้อม 1024-Pad Guard
│       │   ├── infer_pack/                         # โมเดลโครงข่ายประสาทเทียม VITS, Attentions, Synthesizer
│       │   ├── jit/                                # กลไก PyTorch JIT Tracing เร่งความเร็ว
│       │   └── train/                              # ตัวช่วยประมวลผล Spectrogram, Mel, และ Loss Functions
│       └── modules/
│           ├── vc/
│           │   ├── pipeline.py                     # ไปป์ไลน์แปลงเสียงหลัก (Rumble Cut, De-Esser, EBU R128, Safe SID) (v2.4)
│           │   ├── modules.py                      # โมดูลเชื่อมประสาน VC กับ UI และระบบคืนหน่วยความจำ VRAM (v2.4)
│           │   └── utils.py                        # ตัวช่วยคำนวณและจัดการไฟล์เสียง
│           ├── uvr5/                               # โมดูลแยกเสียงร้อง/ดนตรี (MDX-Net / VR Architecture)
│           └── train/                              # ระบบฝึกสอนโมเดลเสียง
│               ├── preprocess.py                   # ตัวเตรียมข้อมูลและหั่นสไลซ์เสียง
│               ├── extract_feature_print.py        # ตัวสกัดฟีเจอร์ HuBERT พร้อม NaN Guard
│               ├── train_dataset_inspector.py      # ตัวตรวจจับความสมบูรณ์ของ Dataset (Filelist Intersection)
│               ├── train_index.py                  # ตัวสร้าง FAISS Index (IndexFlatL2 / IndexIVFFlat)
│               ├── train_pipeline_master.py        # ระบบควบคุมกระบวนการฝึก DDP พอร์ต 9379
│               └── extract/                        # ตัวสกัด F0 ขนาน (RMVPE CUDA, RMVPE DirectML, Harvest)
│
├── 🎤 DiffSinger Training Suite
│   └── DiffSinger_Training/
│       ├── autoclean.py                            # ตัวทำความสะอาดไฟล์เสียงใน Voicebank
│       ├── diffsinger_suite_optimizer.py           # ปรับสมดุลตามกฎ DELTA SYNTH Rule 10
│       └── Run_DiffSinger_Optimizer.bat            # สคริปต์รันตัวปรับปรุง Voicebank แบบ Drag & Drop
│
└── 🔌 Realtime VST Plugin Project (C++17 x64)
    └── RVCRealtimeVST/
        ├── CMakeLists.txt                          # สคริปต์คอมไพล์ C++ ข้ามแพลตฟอร์ม
        ├── config.h                                # นิยามพารามิเตอร์ VST2 / VST3
        ├── src/                                    # ซอร์สโค้ด C++ ปลั๊กอิน, Ring Buffer, IPC Client
        ├── worker/rvc_worker.py                    # Python Worker รับส่งข้อมูลเสียงผ่าน Windows Shared Memory
        └── scripts/prepare-dependencies.ps1        # สคริปต์ติดตั้งไลบรารี iPlug2 และ VST3 SDK
```

---

## 🚀 3. ขั้นตอนการติดตั้งและเปิดใช้งาน (Deployment & Execution)

### 3.1 การเริ่มต้นระบบครั้งแรก
1. ตรวจสอบว่ามีโฟลเดอร์ `runtime\` ซึ่งบรรจุ **Python 3.12.10 Embedded Runtime** อยู่ในโฟลเดอร์โปรเจกต์
2. ดับเบิลคลิกที่ไฟล์ **`Start-The-Program.bat`**
3. ระบบจะทำการตรวจสอบความพร้อมของระบบ และเปิดหน้าต่าง WebUI อัตโนมัติบนพอร์ต **`9378`** (`http://127.0.0.1:9378`)

### 3.2 การดาวน์โหลดและตรวจสอบโมเดลพื้นฐานอัตโนมัติ (Model Assets Verification)
1. ดับเบิลคลิกที่ไฟล์ **`Check-And-Download-Tools.bat`**
2. ระบบจะตรวจสอบโมเดลพื้นฐานทั้งหมด (HuBERT, RMVPE, UVR5 Weights, Pretrained v1/v2) และดาวน์โหลดส่วนที่ขาดหายโดยอัตโนมัติด้วยระบบสำรอง 4 ระดับ (Aria2 → Python urllib → curl → PowerShell)

### 3.3 การฝึกสอนโมเดลเสียงใหม่ (Model Training)
1. นำไฟล์เสียงคุณภาพสูง (.wav) จัดวางลงในโฟลเดอร์ที่ต้องการ
2. ลากโฟลเดอร์เสียงไปวางทับไฟล์ **`Train_Studio_Runner.bat`**
3. ระบบจะทำความสะอาดไฟล์ขยะ สกัดเสียง และเริ่มการฝึกสอนบน DDP พอร์ต `9379` อย่างปลอดภัย พร้อมระบบ Anti-Phantom Indexing Guard

### 3.4 การใช้งานในโปรแกรมทำเพลง (DAW / VST Integration)
1. คอมไพล์หรือนำไฟล์ VST3 จาก `RVCRealtimeVST` ไปวางในโฟลเดอร์ปลั๊กอินของระบบ (`C:\Program Files\Common Files\VST3`)
2. เปิดปลั๊กอินในโปรแกรมทำเพลง (เช่น Studio One, Cubase, FL Studio, Reaper)
3. ปลั๊กอินจะเชื่อมต่อกับ `worker\rvc_worker.py` ผ่าน Windows Shared Memory โดยไม่ต้องตั้งค่าเครือข่าย

---

## 🤖 4. ข้อกำหนดสำหรับ AI Agent ที่เข้ามาร่วมพัฒนา (Multi-Agent Directives)

เพื่อให้โมเดล AI ทุกรุ่น (Claude, Gemini, ChatGPT, Spark, DeepSeek ฯลฯ) สามารถร่วมมือพัฒนาและสานต่องานได้อย่างราบรื่น **ต้องยึดถือกฎต่อไปนี้เป็นอันขาด** (ดูรายละเอียดเพิ่มเติมใน [SPARK_THAI_RVC_BLUEPRINT.md](file:///A:/Thai_RVC_WebUI_v1/SPARK_THAI_RVC_BLUEPRINT.md)):

1. **ห้ามลบโค้ดเดิมแล้วเขียนใหม่ (Preserve Before Replace):** ให้อ่านและทำความเข้าใจลอจิกเดิมก่อนเสมอ การปรับปรุงต้องเป็นแบบผ่าตัดเฉพาะจุด (Surgical Edits)
2. **การฟอกค่า Speaker ID (`sid`):** ต้องครอบด้วย `try: int(sid) except: 0` ก่อนแปลงเป็น `torch.tensor` เสมอ
3. **ห้ามเขียน `times += elapsed`:** ในไฟล์ประมวลผลเวลา เช่น `infer/modules/vc/pipeline.py` ตัวแปร `times` เป็น `list` ห้ามบวกค่าตรงๆ ให้ระบุตำแหน่ง Index เช่น `times[0] += elapsed` เสมอ
4. **การจัดขอบเขต Tensor (Tensor Clamping):** เมื่อทำการสังเคราะห์เสียง ต้องใช้ `min(feat_len, p_len)` เพื่อป้องกันปัญหาขนาดมิติไม่ตรงกัน (`Shape Mismatch`)
5. **การคืนหน่วยความจำ VRAM (Garbage Collection):** ทุกกระบวนการที่มีการใช้งาน PyTorch ต้องมีการเรียกคืน VRAM ในบล็อก `finally:` เสมอ
6. **การตรวจสอบความถูกต้อง (Verification Gate):** รันคำสั่ง `runtime\python.exe -m py_compile [TargetFile]` ทุกครั้งก่อนส่งมอบงาน โดยต้องได้ผลลัพธ์ผ่าน 100% ไร้ข้อผิดพลาด

---

## 🏆 5. เครดิตและการอ้างอิงลิขสิทธิ์ (Credits & Ownership)
- **Project Architect & Maintainer:** Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
- **Original Authors & Contributors:** Patiphat Wongyai (Delta), RVC Project Community, and OpenVPI DiffSinger Team
- **License:** MIT License (ดูรายละเอียดเพิ่มเติมใน `All Documents/LICENSE_MIT_TH_EN.md`)

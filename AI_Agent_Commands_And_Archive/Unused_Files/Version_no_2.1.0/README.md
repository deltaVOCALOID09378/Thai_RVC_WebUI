# DELTA SYNTH — Thai RVC Studio & DiffSinger Unified Engine
## Release Package: Version no. 2.5.0 (Canonical Directory Structure for `A:\Thai_RVC_WebUI_v1`)

> **Architectural Standard:** Enterprise Zero-Defect (`Preserve → Strengthen → Optimize → Verify`)  
> **Project Ownership:** Made And Checked By DELTA SYNTH & All Agentic AI  
> **Original by:** Patiphat Wongyai (Delta), RVC-Project, and OpenVPI DiffSinger  

### 📌 1. โครงสร้างไดเรกทอรีมาตรฐานตามตำแหน่งจริงใน `A:\Thai_RVC_WebUI_v1` (Canonical Directory Structure)

ทุกไฟล์ได้รับการจัดวางตามตำแหน่งโฟลเดอร์จริงของโปรเจกต์ `A:\Thai_RVC_WebUI_v1` โดยไม่มีการวางไฟล์ซ้ำซ้อนหรือปะปนข้ามโฟลเดอร์:

```text
A:\Thai_RVC_WebUI_v1\
├── Program.py                                  # ศูนย์กลางควบคุม 12 แท็บ DELTA RVC Studio x DiffSinger (v5.0)
├── Program_Backup.py                           # ไฟล์สำรอง Program.py (v5.0)
├── infer-web.py                                # หน้าต่าง WebUI มาตรฐาน (v2.5)
├── realtime_gui.py                             # Real-Time Voice Changer GUI - FreeSimpleGUI (v2.5)
├── utils.py                                    # PoliteLogger, Error Diagnostic & Smart Index Matcher (v1.7)
├── dataset_auto_cleaner.py                     # Universal Auto-Purge Sentinel & Audio Header Verifier (v2.2.0)
├── DiffSinger_Trainer_v1.0.py                  # GUI ฝึกสอนโมเดล DiffSinger Acoustic/Variance (v1.1.0)
├── requirements.txt                            # รายการไลบรารีมาตรฐานสำหรับ Python 3.10 (v2.6)
├── Start-The-Program.bat                       # สคริปต์หลักสำหรับเปิดโปรแกรมบนพอร์ต 9378 (v4.7)
├── Train_Studio_Runner.bat                     # สคริปต์รันระบบฝึกโมเดล CLI แบบ Drag & Drop (v2.1.0)
├── run_training_master.bat                     # สคริปต์รัน TrainPipelineMaster แบบ Drag & Drop (v1.1)
├── Run_DiffSinger_Trainer_v1.0.bat             # สคริปต์เปิด GUI ฝึกสอน DiffSinger (v1.0)
├── Install-Env.bat                             # สคริปต์ตรวจสอบและติดตั้งสภาพแวดล้อม (v1.2)
├── Install-File.bat                            # สคริปต์ติดตั้งแพ็กเกจเสริม (v1.4)
├── Check-And-Download-Tools.bat                # สคริปต์ตรวจสอบและดาวน์โหลดโมเดลพื้นฐานอัตโนมัติ (v1.0)
├── Copy_Thai_RVC_WebUI_v1.bat                  # สคริปต์สำรองข้อมูลโปรเจกต์ด้วย Robocopy (v1.0)
├── Pack-Full-Release-7z.bat                    # สคริปต์บีบอัดแพ็กเกจระบบเต็ม (.7z) (v1.3)
├── Pack-Source-Code-Zip.bat                    # สคริปต์บีบอัดแพ็กเกจซอร์สโค้ด (.zip) (v1.3)
├── Install_Update_v2.1.0_To_PC.bat             # สคริปต์ติดตั้งอัปเดตลงตามพาทจริงใน A:\Thai_RVC_WebUI_v1 (v2.5.0)
├── README.md / About_Project.md / AGENTS.md / AGENT.md / SPARK_THAI_RVC_BLUEPRINT.md
├── All Documents/
│   ├── THAI_RVC_COMPLETE_MASTER_KNOWLEDGE_BASE.md
│   ├── ENTERPRISE_MASTER_PROMPT_SUITE_THAI_RVC.md
│   └── คู่มือการติดตั้งและใช้งานเบื้องต้น Thai_RVC_WebUI.md
├── DiffSinger_Training/
│   ├── autoclean.py
│   ├── diffsinger_suite_optimizer.py
│   └── Run_DiffSinger_Optimizer.bat
├── Published/
│   └── Version no. 2.1.0/                      # ชุดไฟล์แพตช์ย่อยตามโครงสร้างมาตรฐานรุ่น 2.1.0
├── configs/
│   ├── config.py                               # ตัวจัดการฮาร์ดแวร์ CUDA / DirectML / IPEX / MPS / CPU (v1.7)
│   ├── v1/{32k,40k,48k}.json                   # คอนฟิกสถาปัตยกรรม RVC v1 (256-dim)
│   ├── v2/{32k,40k,48k}.json                   # คอนฟิกสถาปัตยกรรม RVC v2 (768-dim)
│   └── inuse/v1,v2/{32k,40k,48k}.json          # คอนฟิกที่ใช้งานจริงขณะรันระบบ
├── i18n/
│   ├── i18n.py / locale_diff.py / scan_i18n.py
│   └── locale/{th_TH,en_US,es_ES,ja_JP}.json   # ไฟล์แปลภาษาในโฟลเดอร์ locale เท่านั้น
├── tools/
│   ├── app.py / calc_rvc_model_similarity.py / cuda_graph.py / dlmodels.bat / download_models.py
│   ├── export_onnx.py / file_io.py / infer_batch_rvc.py / infer_cli.py / multispeaker.py
│   ├── onnx_inference_demo.py / process_utils.py / progress.py / pymss_webui.py / rvc_for_realtime.py
│   └── test_system_smoke.py / test_toast_integration.py
└── infer/
    ├── lib/
    │   ├── audio.py / slicer2.py / rmvpe.py
    │   ├── infer_pack/{attentions,commons,models,models_onnx,modules,onnx_inference,transforms}.py
    │   ├── jit/{__init__,get_hubert,get_rmvpe,get_synthesizer}.py
    │   ├── uvr5_pack/utils.py
    │   └── train/{utils,data_utils,losses,mel_processing,process_ckpt}.py
    └── modules/
        ├── modules.py
        ├── vc/{__init__,pipeline,modules,utils}.py
        ├── onnx/export.py
        ├── ipex/{__init__,attention,gradscaler,hijacks}.py
        ├── uvr5/{mdxnet,modules,preprocess}.py
        └── train/
            ├── preprocess.py
            ├── extract_feature_print.py
            ├── train.py
            ├── rvc_train_utils.py
            ├── train_dataset_inspector.py
            ├── train_index.py
            ├── train_pipeline_master.py
            └── extract/
                ├── f0_extractor.py
                ├── extract_f0_print.py
                ├── extract_f0_rmvpe.py
                └── extract_f0_rmvpe_dml.py
```

### 🛠️ 2. วิธีติดตั้งลง `A:\Thai_RVC_WebUI_v1` (Deployment Instructions)
1. แตกไฟล์ `Thai_RVC_WebUI_v1.zip` แล้วรัน `Install_Update_v2.1.0_To_PC.bat` (หรือแตกไฟล์ทับลงใน `A:\Thai_RVC_WebUI_v1` โดยตรง)
2. สคริปต์ `Install_Update_v2.1.0_To_PC.bat` จะทำความสะอาดไฟล์ที่เคยถูกวางเกินใน Root ออกให้อัตโนมัติ สำรองไฟล์เดิมไว้ที่ `backups\pre_v2.5.0_backup` และคัดลอกทุกไฟล์ลงตามพาทจริงใน `A:\Thai_RVC_WebUI_v1`

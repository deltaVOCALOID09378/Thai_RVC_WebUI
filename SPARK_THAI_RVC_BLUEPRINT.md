<!-- 
==============================================================================
File: SPARK_THAI_RVC_BLUEPRINT.md | Version: 1.0
Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
Original by Patiphat Wongyai (Delta) & RVC Project
Description: Master Development Blueprint for Spark AI & Agentic Collaborators
==============================================================================
-->

# 🇹🇭 THAI RVC STUDIO - SPARK DEVELOPMENT BLUEPRINT
เอกสารนี้คือคัมภีร์และกรอบการทำงาน (Framework) สำหรับการพัฒนาระบบ Thai RVC Studio 
Spark (และ AI Agent ทุกตัว) **ต้อง** ทำความเข้าใจและยึดถือมาตรฐานในเอกสารฉบับนี้อย่างเคร่งครัดก่อนดำเนินการแก้โค้ด

## 1. ปรัชญาการพัฒนา (Core Philosophy)
- **Preserve Before Replace (รักษาของเดิมก่อนสร้างใหม่):** ทำความเข้าใจลอจิกเดิมเสมอ เน้นการแก้ไขแบบผ่าตัด (Surgical Edits) ห้ามลบหรือเขียนระบบใหม่ทั้งหมดหากไม่ได้รับคำสั่ง
- **Enterprise Zero-Defect:** โค้ดที่สร้างต้องปราศจากบัค (Bug-free) และไร้คำเตือน (Zero Warnings) จัดการ Error Handling อย่างรัดกุม 
- **Resource Efficiency:** บริหารทรัพยากร VRAM/RAM ให้เกิดประโยชน์สูงสุด ทุกครั้งที่จบโปรเซสต้องมีการบังคับคืนค่าหน่วยความจำเสมอ

## 2. มาตรฐานระบบและโครงสร้างสถาปัตยกรรม (System Architecture Standards)
การทำงานกับระบบ Thai RVC ต้องปฏิบัติตาม Environment Standards ดังนี้:

### 2.1 Python Environment
- **Path:** ต้องเรียกใช้งานผ่าน `runtime\python.exe` ที่มาพร้อมกับตัวโปรแกรมเสมอ ห้ามพึ่งพา Global Python
- **Execution:** งาน Python ทั้งหมด (Local AI/WebUI) ต้องเริ่มจากไดเรกทอรี `runtime\`

### 2.2 Network & Ports
- **Default Port:** พอร์ตเริ่มต้นสำหรับระบบ WebUI RVC คือ **9378** (ห้ามตั้งค่ากลับไปเป็น 7865 หรือพอร์ตอื่น)

### 2.3 File Naming & UI Standards
- **Main GUI File:** ไฟล์แกนหลักที่ใช้เปิดโปรแกรมและรวมระบบทั้งหมด (Inference, Batch, UVR5, Training) จะต้องใช้ชื่อ **`Program.py`** 
- **Batch Files (.bat):** ไฟล์ .bat สำหรับเปิดระบบหรือเครื่องมือต่างๆ จะต้องเขียนผลลัพธ์ (Terminal Output) และใช้คำสั่งเป็น **ภาษาอังกฤษ 100%** เท่านั้น และรองรับระบบ Drag & Drop
- **User Interface (UI):** 
  - ใช้งานภาษาไทยควบคู่กับภาษาอังกฤษ (Bilingual Communication) ในระดับทางการและเข้าใจง่าย
  - ฟอนต์หลัก: `Leelawadee UI`
  - โทนสีองค์กร (Corporate Styling): สีดำ (`#1A1A1A`), สีขาว (`#F0F0F0`), สีแดงเน้นย้ำ (`#CC2200`) และ สีแดงเมื่อ Hover (`#FF4422`)

## 3. กฎเหล็กในการเขียนและแก้ไขโค้ด (Coding & Debugging Directives)

### 3.1 การจัดการเวลาและตัวแปร (Variables & Data Types)
- **ห้ามบวก Float เข้ากับ List โดยตรง:** หากมีการบันทึกเวลา (Processing Time) ในอาร์เรย์ (เช่น `times` ใน `pipeline.py`) ต้องระบุ Index ให้ชัดเจน เช่น `times[0] += (t1 - t0)` ห้ามใช้ `times += t1 - t0` เด็ดขาดเพื่อป้องกัน `TypeError`

### 3.2 ความปลอดภัยของข้อมูลและ VRAM (Data Safety & VRAM Management)
- **F0 & Feature Synchronization:** ทุกครั้งที่สกัดระดับเสียงหรือฟีเจอร์ ต้องมีการเช็คและหั่นขนาดเฟรม (Frame Cropping) เช่น `min(p_len, len_spec)` ให้เท่ากันทั้งหมด ป้องกันปัญหา `Tensor Shape Mismatch`
- **Garbage Collection:** ต้องมีฟังก์ชัน `clear_vram_safe()` (ประกอบด้วย `gc.collect()` และ `torch.cuda.empty_cache()`) ครอบอยู่ใน `finally:` ของทุกๆ Generator/Try-Except block

### 3.3 การบันทึกและแจ้งเตือน (Logging System)
- ใช้คลาส `PoliteLogger` ที่สร้างไว้เป็นมาตรฐานในการแจ้งเตือนแบบสองภาษา ห้ามใช้ `print()` ธรรมดาหรือปล่อย Error ทิ้งไว้โดยไม่จัดการ (`except: pass` ห้ามใช้เด็ดขาด)
- ระบบ Batch Processing ต้องรายงานความคืบหน้าแบบ Real-time และอัปเดตสถานะใน UI อย่างต่อเนื่อง

### 3.4 ข้อมูลลิขสิทธิ์และเวอร์ชัน (Credits & Versioning)
- โค้ดที่อัปเดตหรือสร้างใหม่ ต้องปรับเลขเวอร์ชันเพิ่มขึ้น `0.1` ภายในไฟล์ (ห้ามเปลี่ยนชื่อไฟล์หลัก)
- ส่วนหัวของไฟล์โค้ด (Header) ต้องคงเครดิตต่อไปนี้ไว้เสมอ:
  ```python
  # ==============================================================================
  # Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
  # Original by Patiphat Wongyai (Delta) & RVC Project
  # ==============================================================================
# บันทึกประวัติการส่งมอบระบบ (HuBERT Feature Dimension Alignment & Zero-Crash Architecture)
**ไฟล์:** `History_no.27_25-09-26_23-55.md`  
**วัน/เดือน/ปี:** 25/09/2569 (2026-09-25) | 23:55 น.  
**Made And Checked By:** DELTA SYNTH & All Code Agentic AI Engine  
**Original by:** RVC-Project & DELTA SYNTH  
**เป้าหมาย:** แก้ไขข้อผิดพลาด `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1298x768 and 256x192)` ในกระบวนการเรนเดอร์เสียง ให้รองรับโมเดล RVC ทั้งสถาปัตยกรรม v1 (256-dim) และ v2 (768-dim) สมบูรณ์ 100%

---

## 1. สาเหตุรากของปัญหา (Root Cause Analysis)
- **ตำแหน่งที่พบข้อผิดพลาด:** `infer/modules/vc/pipeline.py` (ฟังก์ชัน `vc`) เรียก `net_g.infer()` แล้วเกิดข้อผิดพลาดใน `infer/lib/infer_pack/models.py` (บรรทัดที่ 68) 
- **อาการ:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1298x768 and 256x192)`
- **สาเหตุเชิงลึก:**
  1. ในโมเดล RVC สถาปัตยกรรม v1 ตัวเข้ารหัสเสียง (`enc_p.emb_phone`) มีขนาด `Linear(256, 192)` ต้องการฟีเจอร์เสียงขนาด 256 มิติ
  2. โค้ดเดิมใน `pipeline.py` มีการกำหนดสลับชั้นและขาดขั้นตอนการโปรเจกต์มิติข้อมูล:
     - ใน v1 สกัดฟีเจอร์ออกมาโดยไม่ระบุ `output_layer=9` และไม่ได้ผ่าน `model.final_proj` (Linear 768 -> 256) ทำให้ฟีเจอร์ค้างอยู่ที่ขนาด 768 มิติ
     - ส่งผลให้เกิดการคูณเมทริกซ์ระหว่าง `(1298, 768)` กับ `(256, 192)` ซึ่งขัดกับหลักพีชคณิตเชิงเส้นจนโปรแกรมหยุดทำงานกะทันหัน
  3. นอกจากนี้ ใน `infer/modules/vc/modules.py` การระบุเวอร์ชันเดิมอ้างอิงจาก `self.cpt.get("version", "v1")` ซึ่งหาก Checkpoint ไม่มีคีย์ดังกล่าว หรือคีย์ไม่ตรงกับ Weight จริง อาจทำให้ระบบเลือก Generator หรือไปป์ไลน์ผิดพลาด

---

## 2. การแก้ไขที่ต้นเหตุ (Surgical Fixes Applied)

### 2.1 ปรับปรุงตรรกะใน `infer/modules/vc/pipeline.py` (เวอร์ชัน 2.1)
1. **Dynamic Target Dimension Detection:**
   - ตรวจสอบมิติข้อมูลที่ตัวโมเดล `net_g` คาดหวังโดยตรงจาก `net_g.enc_p.emb_phone.in_features`
   - กำหนดสถานะ `use_v1 = (net_dim == 256)` อย่างแม่นยำ
2. **Correct Layer Extraction & Final Projection:**
   - สกัดฟีเจอร์ด้วย `output_layer=9` สำหรับ v1 และ `output_layer=12` สำหรับ v2
   - เมื่อเป็นโมเดล v1 ทำการแปลงเวกเตอร์ผ่าน `model.final_proj(feats)` ลดมิติจาก 768 สู่ 256 มิติอย่างถูกต้อง
3. **Safety Fallback Guard:**
   - หากตรวจพบว่าฟีเจอร์มี 768 มิติแต่ตัวโมเดลต้องการ 256 มิติ จะทำการรัน `model.final_proj` กู้คืนมิติให้อัตโนมัติทันที
4. **Faiss Index Dimension Guard:**
   - เพิ่มการตรวจสอบขนาดมิติของไฟล์ Index (`index.d`) เทียบกับ `feats.shape[-1]`
   - หากผู้ใช้ใส่ Index ข้ามเวอร์ชัน (เช่น นำ Index 768 มิติมาใช้กับโมเดล 256 มิติ) ระบบจะแจ้งเตือนผ่าน `PoliteLogger.warning` และสลับไปเรนเดอร์โดยไม่ใช้ Index แทนการขัดข้องและ Crash

### 2.2 ปรับปรุงการตรวจสอบเวอร์ชันใน `infer/modules/vc/modules.py` (เวอร์ชัน 1.9)
- อ่านมิติของ `enc_p.emb_phone.weight.shape[1]` จาก Checkpoint เพื่อกำหนด `self.version = "v2" if phone_dim == 768 else "v1"` แบบ Deterministic 100%

---

## 3. การเพิ่มชุดทดสอบและผลการทดสอบ (Automated Verification)
- เพิ่ม `test_06_pipeline_feature_dimension_handling` ใน `tools/test_system_smoke.py`
- ตรวจสอบการจำลอง HuBERT Feature Extraction, การโปรเจกต์มิติ v1 (256), v2 (768), และ Faiss Index Dimension Safety
- **ผลการทดสอบ:**
  ```text
  Ran 6 tests in 0.443s
  OK
  [PASS] Config Initialized: Device=cpu, is_half=False
  [PASS] change_rms DSP calculations verified with 0 NaN/Inf
  [PASS] F0 Cache and VRAM/RAM reclamation executed cleanly
  [PASS] Index path discovery utility handles boundary inputs safely
  [PASS] VC Engine lifecycle, unload, and cancel functions verified
  [PASS] Pipeline feature extraction, dimension guards (v1: 256, v2: 768), and Faiss safety verified
  ```

---

## 4. รายการไฟล์ที่ซิงค์ไปยังแพ็กเกจเผยแพร่ (Release Sync)
- `infer/modules/vc/pipeline.py` -> `Published/Version no. 2.0.0/infer/modules/vc/pipeline.py`
- `infer/modules/vc/modules.py` -> `Published/Version no. 2.0.0/infer/modules/vc/modules.py`
- `tools/test_system_smoke.py` -> `Published/Version no. 2.0.0/tools/test_system_smoke.py`
- `history/History_no.27_25-09-26_23-55.md` -> `Published/Version no. 2.0.0/history/History_no.27_25-09-26_23-55.md`

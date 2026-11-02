<!-- 
==============================================================================
Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
Original by Patiphat Wongyai (Delta) & DELTA SYNTH Architecture
Version: 1.0.0
Description: Enterprise Google Cloud Bigtable Schema & Operations Specification
==============================================================================
-->

# DELTA SYNTH — Google Cloud Bigtable Architecture & Schema Specification
## สถาปัตยกรรม Schema ฐานข้อมูล NoSQL และ Row Key Design ความเร็วสูงสำหรับ Thai RVC Studio

> **Standard:** DELTA SYNTH Enterprise Protocol (**Preserve → Strengthen → Optimize → Verify**)  
> **Category:** High-Throughput Speech Data & Vector Feature Store  
> **Target Engine:** Google Cloud Bigtable (Low Latency, Petabyte-Scale NoSQL)  
> **Last Updated:** 2026-10-18  

---

## 1. Executive Summary & Design Principles (บทนำและหลักการออกแบบ)

ในระบบแปลงเสียง Thai RVC และ DiffSinger ระดับสตูดิโอและองค์กร เมื่อต้องรองรับผู้พูดจำนวนมาก (Multi-Speaker 10,000+ Profiles), การจัดเก็บเวกเตอร์เนื้อเสียง HuBERT / ContentVec (768 มิติ), เส้นพิทช์ F0 และชุดข้อมูลคู่เทียบเสียงวรรณยุกต์ (Minimal Tone Pairs) ระดับหลายล้านประโยค การจัดเก็บบนระบบไฟล์ธรรมดาหรือ RDBMS จะเกิดปัญหาคอขวดด้าน I/O, File System Inodes, และ Latency ในการค้นหา

**Google Cloud Bigtable** มอบขีดความสามารถในการอ่านและเขียนข้อมูลเสียงด้วยความเร็วต่ำกว่า 10 มิลลิวินาที (Sub-10ms Latency) พร้อมการขยายตัวแบบไร้ขีดจำกัด เอกสารฉบับนี้กำหนดโครงสร้าง Schema, Row Key Design, Column Families และ Garbage Collection Rules ตามมาตรฐานสูงสุดของ DELTA SYNTH เพื่อป้องกันปัญหา **Hotspotting** และเพิ่มประสิทธิภาพการดึงข้อมูลสูงสุด

---

## 2. Row Key Design & Anti-Hotspotting Rules (กฎการออกแบบ Row Key)

หลักการสำคัญของ Bigtable คือข้อมูลจะถูกจัดเรียงตามลำดับตัวอักษรของ **Row Key (Lexicographical Order)** หากออกแบบ Row Key ไม่ถูกต้อง ข้อมูลที่ถูกเขียนพร้อมกันจะตกไปยังโหนดเดียวกัน (Hotspotting) ทำให้ระบบทำงานช้าลง

### 2.1 กฎเหล็ก 4 ประการในการออกแบบ Row Key ของ DELTA SYNTH:
1. **Field Hashing / Salting Prefix:** นำ Hash 4 ตัวอักษรแรกของ Speaker ID หรือ Entity ID มานำหน้า Row Key เสมอ เพื่อกระจายข้อมูลไปยัง Tablet Servers ทั่วทั้งคลัสเตอร์อย่างสมดุล (ป้องกัน Write Hotspots)
2. **Reverse Timestamp:** สำหรับข้อมูลที่เป็นอนุกรมเวลา (เช่น Feature frames, Logs, Telemetry) ต้องใช้ `(9999999999999 - timestamp_ms)` เพื่อให้ข้อมูลที่ถูกบันทึกล่าสุดอยู่ด้านบนสุดเสมอ เพิ่มความเร็วในการสแกน (Query Latest)
3. **Compound Key Structure:** คั่นข้อมูลใน Row Key ด้วยเครื่องหมาย `#` (Sharp) เสมอ เช่น `<hash>#<speaker_id>#<version>#<target_sr>`
4. **Row Key Byte Limits:** ความยาวของ Row Key ต้องไม่เกิน 4KB (แนะนำให้อยู่ระหว่าง 32 ถึง 128 ไบต์)

---

## 3. Comprehensive Table Schemas (โครงสร้างตารางทั้ง 4 ระบบ)

### 3.1 ตารางที่ 1: `thai_rvc_models` (Model Registry & Checkpoints)
ใช้สำหรับจัดเก็บบันทึกโปรไฟล์โมเดลเสียง, น้ำหนัก Synthesizer, การตั้งค่าความถี่, และดัชนี FAISS ประจำตัวผู้พูด

* **Row Key Pattern:**
  ```text
  <spk_md5_prefix:4>#<speaker_id>#<version:v1|v2>#<sample_rate:32k|40k|48k>
  ตัวอย่าง: a7f2#delta_singer_th#v2#48k
  ```
* **Column Families:**
  | Column Family | หน้าที่และข้อมูลที่จัดเก็บ | GC Policy (การจัดการอายุข้อมูล) |
  | :--- | :--- | :--- |
  | `cf_meta` | `name`, `gender`, `lang`, `f0_type`, `created_at` | MaxVersions = 5 |
  | `cf_config` | `json_config`, `emb_dim`, `hidden_channels` | MaxVersions = 3 |
  | `cf_weights` | `model_bin`, `onnx_bin`, `state_dict_hash` | MaxVersions = 2 |
  | `cf_index` | `faiss_index_bin`, `total_vectors`, `index_type` | MaxVersions = 2 |

---

### 3.2 ตารางที่ 2: `thai_rvc_features` (Acoustic Feature Vector Store)
ใช้สำหรับจัดเก็บเวกเตอร์ 768-dim HuBERT/ContentVec, เส้นเสียง F0, และ Spectrogram ของไฟล์เสียงที่ผ่านการตัดสไลซ์แล้ว

* **Row Key Pattern:**
  ```text
  <spk_md5_prefix:4>#<speaker_id>#<rev_ts:13>#<chunk_id:6>
  ตัวอย่าง: e4b1#delta_singer_th#8458921473210#000142
  ```
* **Column Families:**
  | Column Family | หน้าที่และข้อมูลที่จัดเก็บ | GC Policy (การจัดการอายุข้อมูล) |
  | :--- | :--- | :--- |
  | `cf_content` | `hubert_npy` (เวกเตอร์ 768-dim), `feat_len` | MaxVersions = 1 |
  | `cf_pitch` | `f0_rmvpe_npy`, `f0_harvest_npy`, `p_len` | MaxVersions = 1 |
  | `cf_audio` | `spec_npy`, `rms_energy`, `deess_gain` | MaxVersions = 1 |
  | `cf_quality` | `snr_db`, `clipping_ratio`, `has_nan` | MaxVersions = 1 |

---

### 3.3 ตารางที่ 3: `thai_rvc_minimal_tone_pairs` (Thai Tone Preservation Corpus)
ใช้สำหรับจัดเก็บคลังคำคู่เทียบวรรณยุกต์ภาษาไทย 5 เสียง (สามัญ, เอก, โท, ตรี, จัตวา) สำหรับการทดสอบและวัดความแม่นยำ (Tone Accuracy Benchmark)

* **Row Key Pattern:**
  ```text
  <tone_id:0..4>#<base_consonant>#<vowel_code>#<word_thai>
  ตัวอย่าง: 0#p#aa#ปา (สามัญ)
  ตัวอย่าง: 1#p#aa#ป่า (เอก)
  ตัวอย่าง: 2#p#aa#ป้า (โท)
  ตัวอย่าง: 3#p#aa#ป๊า (ตรี)
  ตัวอย่าง: 4#p#aa#ป๋า (จัตวา)
  ```
* **Column Families:**
  | Column Family | หน้าที่และข้อมูลที่จัดเก็บ | GC Policy |
  | :--- | :--- | :--- |
  | `cf_phonetics` | `ipa`, `tone_name`, `consonant_class` (สูง/กลาง/ต่ำ), `vowel_len` | MaxVersions = 1 |
  | `cf_target_f0` | `f0_contour_shape` (Flat/Fall/Rise), `f0_mean`, `f0_std` | MaxVersions = 1 |
  | `cf_benchmark` | `expected_formants`, `sample_ref_wav` | MaxVersions = 2 |

---

### 3.4 ตารางที่ 4: `thai_rvc_telemetry` (Live Inference QoS & Memory Profiling)
ใช้สำหรับจัดเก็บบันทึกการทำงานสด (Telemetry Logs), ค่า Latency ของอัลกอริทึม, ปริมาณการใช้งาน VRAM/RAM, และตรวจจับ Memory Leak แบบเรียลไทม์

* **Row Key Pattern:**
  ```text
  <date_yyyymmdd>#<spk_md5_prefix:4>#<rev_ts:13>#<request_id>
  ตัวอย่าง: 20261018#a7f2#8458921473210#req_98124b81
  ```
* **Column Families:**
  | Column Family | หน้าที่และข้อมูลที่จัดเก็บ | GC Policy |
  | :--- | :--- | :--- |
  | `cf_timing` | `infer_ms`, `f0_extract_ms`, `hubert_ms`, `rtf` | MaxAge = 30 Days |
  | `cf_memory` | `vram_allocated_mb`, `vram_reserved_mb`, `ram_used_mb`, `leak_detected` | MaxAge = 30 Days |
  | `cf_status` | `status_code`, `error_traceback`, `audio_duration_sec` | MaxAge = 30 Days |

---

## 4. Bigtable Operations & CBT CLI Commands (คำสั่งบริหารจัดการ)

สำหรับการสร้างตารางและกำหนด Column Family ด้วยเครื่องมือ `cbt` CLI ของ Google Cloud:

```bash
# 1. สร้างตาราง thai_rvc_models พร้อม Column Families
cbt createtable thai_rvc_models "families=cf_meta:maxversions=5,cf_config:maxversions=3,cf_weights:maxversions=2,cf_index:maxversions=2"

# 2. สร้างตาราง thai_rvc_features พร้อม Column Families
cbt createtable thai_rvc_features "families=cf_content:maxversions=1,cf_pitch:maxversions=1,cf_audio:maxversions=1,cf_quality:maxversions=1"

# 3. สร้างตาราง thai_rvc_minimal_tone_pairs
cbt createtable thai_rvc_minimal_tone_pairs "families=cf_phonetics:maxversions=1,cf_target_f0:maxversions=1,cf_benchmark:maxversions=2"

# 4. สร้างตาราง thai_rvc_telemetry พร้อมตั้งอายุอัตโนมัติ 30 วัน (Auto-Expire 30d)
cbt createtable thai_rvc_telemetry "families=cf_timing:maxage=30d,cf_memory:maxage=30d,cf_status:maxage=30d"
```

---

## 5. Python Integration Architecture (การเชื่อมต่อกับ Python Pipeline)

```python
# -*- coding: utf-8 -*-
"""
DELTA SYNTH - Google Cloud Bigtable Client Integration Module
"""
import hashlib
import time
from typing import Optional

def generate_model_row_key(speaker_id: str, version: str = "v2", sample_rate: str = "48k") -> str:
    """สร้าง Row Key แบบกระจายข้อมูล (Salting Prefix) ป้องกัน Hotspotting"""
    spk_hash = hashlib.md5(speaker_id.encode("utf-8")).hexdigest()[:4]
    return f"{spk_hash}#{speaker_id}#{version}#{sample_rate}"

def generate_telemetry_row_key(speaker_id: str, request_id: str) -> str:
    """สร้าง Row Key สำหรับ Telemetry โดยใช้ Reverse Timestamp เพื่อดึงข้อมูลล่าสุดได้ทันที"""
    spk_hash = hashlib.md5(speaker_id.encode("utf-8")).hexdigest()[:4]
    date_str = time.strftime("%Y%m%d")
    rev_ts = str(9999999999999 - int(time.time() * 1000))
    return f"{date_str}#{spk_hash}#{rev_ts}#{request_id}"
```

---

## 6. สรุปผลประโยชน์ทางวิศวกรรม (Engineering Impact)
1. **Zero Hotspotting:** ข้อมูลถูกกระจายไปยัง Bigtable Tablets อย่างสม่ำเสมอด้วย Hash Prefix 4 หลัก
2. **Sub-10ms Feature Query:** ดึงข้อมูลเวกเตอร์ 768 มิติสำหรับ Real-time Inference ได้รวดเร็ว รองรับงาน VST Latency ต่ำ
3. **Automated Lifecycle Management:** ข้อมูล Telemetry และ Memory Logs มีการตัดล้างอัตโนมัติ (GC 30 วัน) ไม่เปลืองพื้นที่จัดเก็บ
4. **Phonetic Tone Integrity:** มี Schema สำหรับ Minimal Tone Pairs ที่เป็นระเบียบ รองรับการวัดผลความแม่นยำภาษาไทยครบ 5 เสียง

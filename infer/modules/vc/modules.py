# -*- coding: utf-8 -*-
# ==============================================================================
# ชื่อไฟล์โค้ด (File Name)     : modules.py
# เวอร์ชั่นโค้ด (Version)      : 2.8 (Zero-Defect Balanced Edition)
# ตำแหน่งจัดเก็บ (Location)   : infer/modules/vc/modules.py
# หน้าที่ (Function/Role)     : โมดูลควบคุมการแปลงเสียงหลัก (VC Controller) รองรับงานเดี่ยวและงานกลุ่ม
#                             แก้ไขบั๊กการเขียนทับรายการไฟล์ในโหมด Smart Batch Resume,
#                             เชื่อมต่อ utils.py สำหรับค้นหา Index & โหลด HuBERT และคืนค่าแคช F0 อัตโนมัติ
# เครดิตผู้สร้าง (Credit)      : Made And Checked By DELTA SYNTH & All Agentic AI
#                             Original by RVC-Project & Patiphat Wongyai (Delta)
# ==============================================================================

import traceback
import logging
import os
import shutil
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import soundfile as sf
import torch
import faiss

from utils import PoliteLogger

try:
    import fairseq.data.dictionary
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
except Exception:
    pass

logger = logging.getLogger(__name__)

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline, clear_f0_cache
from infer.modules.vc.utils import get_index_path_from_model, load_hubert


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.hubert_model = None
        self.is_rendering = False
        self.config = config
        PoliteLogger.system("เริ่มต้นโมดูลเปลี่ยนเสียง (VC Engine Initialized - v2.8)")

    def get_vc(self, sid, *to_return_protect):
        PoliteLogger.info(f"กำลังโหลดโมเดลเป้าหมาย: {sid} (Loading target model: {sid})")

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[1] if self.if_f0 != 0 and len(to_return_protect) > 1 else 0.33,
            "__type__": "update",
        }

        if sid == "" or sid == [] or sid is None:
            if self.hubert_model is not None or self.net_g is not None:
                PoliteLogger.system("กำลังล้างแคชโมเดลและ F0 ในหน่วยความจำ (Cleaning model & F0 cache in VRAM/RAM)")
                self.hubert_model = None
                self.net_g = None
                self.n_spk = None
                self.tgt_sr = None
                self.cpt = None
                clear_f0_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {"visible": True, "value": to_return_protect0["value"], "__type__": "update"},
                {"visible": True, "value": to_return_protect1["value"], "__type__": "update"},
                "",
                "",
            )

        weight_root = os.getenv("weight_root", "assets/weights")
        person = os.path.join(weight_root, str(sid))
        if not os.path.exists(person) and os.path.exists(str(sid)):
            person = str(sid)

        if not os.path.exists(person):
            PoliteLogger.error(f"ไม่พบไฟล์โมเดลที่ตำแหน่ง: {person} (Model file not found at: {person})")
            return {"visible": False, "__type__": "update"}, to_return_protect0, to_return_protect1, "", ""

        try:
            self.cpt = torch.load(person, map_location="cpu", weights_only=True)
        except Exception:
            self.cpt = torch.load(person, map_location="cpu", weights_only=False)

        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)

        if self.config.is_half and str(self.config.device) not in ["cpu", "mps"]:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]

        to_return_protect0["visible"] = self.if_f0 != 0
        to_return_protect1["visible"] = self.if_f0 != 0

        idx_path = get_index_path_from_model(sid)
        index = {"value": idx_path, "__type__": "update"}
        PoliteLogger.model(sid, f"| ความถี่: {self.tgt_sr}Hz | เวอร์ชัน: {self.version} | ดัชนี: {os.path.basename(idx_path) if idx_path else 'None'}")

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def cancel_process(self):
        self.is_rendering = False
        PoliteLogger.warning("ยกเลิกกระบวนการเรนเดอร์เสียงตามคำสั่งผู้ใช้งาน (Render process cancelled by user)")
        return "🛑 ยกเลิกคำสั่งเรียบร้อยแล้ว (Process Cancelled)"

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        try:
            sid = int(sid)
        except (ValueError, TypeError):
            sid = 0

        if not input_audio_path:
            PoliteLogger.warning("โปรดระบุเส้นทางไฟล์เสียงต้นฉบับ (Please provide input audio path)")
            return "กรุณาระบุไฟล์เสียงต้นฉบับ (Please provide input audio path)", (None, None)

        input_audio_path = str(input_audio_path).strip().strip('"').strip("'")
        if not os.path.exists(input_audio_path):
            PoliteLogger.error(f"ไม่พบไฟล์เสียงต้นฉบับ: {input_audio_path} (Audio file not found)")
            return f"ไม่พบไฟล์ที่ระบุ: {input_audio_path}", (None, None)

        if self.net_g is None or self.pipeline is None:
            PoliteLogger.error("ยังไม่ได้เลือกหรือโหลดโมเดลเสียง .pth (Voice model is not loaded)")
            return "❌ กรุณาเลือกโมเดลเสียงเป้าหมาย (.pth) ก่อนเริ่มเรนเดอร์ (Please select a target voice model first)", (None, None)

        resolved_index = ""
        if file_index and str(file_index).strip():
            candidate = str(file_index).strip().strip('"').strip("\n").replace("trained", "added")
            if os.path.isfile(candidate):
                resolved_index = candidate
            else:
                PoliteLogger.warning(f"ไม่พบไฟล์ Index ที่ระบุเอง: {candidate} ระบบจะสลับไปใช้ค่าสำรองหรือข้ามดัชนี")

        if not resolved_index and file_index2 and str(file_index2).strip():
            candidate2 = str(file_index2).strip().strip('"').replace("trained", "added")
            if os.path.isfile(candidate2):
                resolved_index = candidate2

        if resolved_index:
            try:
                faiss.read_index(resolved_index)
            except Exception:
                PoliteLogger.error(f"ไฟล์ Index เสียหาย ข้ามการใช้ดัชนี: {resolved_index} (Corrupted Index file, skipping)")
                resolved_index = ""

        f0_up_key = int(f0_up_key) if f0_up_key is not None else 0
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                PoliteLogger.system("กำลังโหลด HuBERT Base Model... (Loading HuBERT Base Model...)")
                self.hubert_model = load_hubert(self.config)

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                resolved_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            tgt_sr = resample_sr if self.tgt_sr != resample_sr >= 16000 else self.tgt_sr

            index_info = f"Index Path: {resolved_index}" if resolved_index else "Index: Not Used"
            PoliteLogger.success(f"เรนเดอร์ไฟล์เดี่ยวสำเร็จ: {os.path.basename(input_audio_path)} (Single file render complete)")

            return (
                f"🟢 เรนเดอร์สำเร็จ (Render Complete)\n{index_info}\nเวลาประมวลผล (Processing Time): npy: {times[0]:.2f}s, f0: {times[1]:.2f}s, infer: {times[2]:.2f}s",
                (tgt_sr, audio_opt),
            )
        except Exception as e:
            err_ui = PoliteLogger.error(
                msg=f"การเรนเดอร์ไฟล์เดี่ยวล้มเหลว: {os.path.basename(input_audio_path)}",
                exc=e,
                context="Inference - Single Audio Render"
            )
            return err_ui, (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        if self.is_rendering:
            warning_msg = "มีงานเรนเดอร์แบบกลุ่มกำลังทำงานอยู่แล้ว (A batch render process is already active)"
            PoliteLogger.warning(warning_msg)
            yield f"⚠️ {warning_msg}"
            return

        self.is_rendering = True
        batch_start_time = time.time()
        success_count = 0
        fail_count = 0
        try:
            # ถ้ารับค่า sid เป็นชื่อไฟล์โมเดล .pth และยังไม่ได้โหลดโมเดล ให้ทำการโหลดอัตโนมัติ
            if isinstance(sid, str) and sid.lower().endswith(".pth"):
                if self.net_g is None:
                    self.get_vc(sid)
                sid_num = 0
            else:
                try:
                    sid_num = int(sid)
                except (ValueError, TypeError):
                    sid_num = 0

            dir_path = dir_path.strip(" ").strip('"').strip("\n") if dir_path else ""
            opt_root = opt_root.strip(" ").strip('"').strip("\n") if opt_root else ""

            if dir_path != "" and not os.path.exists(dir_path) and not paths:
                err_msg = f"ไม่พบโฟลเดอร์ต้นฉบับ: {dir_path} (Source directory not found)"
                PoliteLogger.error(err_msg)
                yield f"❌ {err_msg}"
                return

            os.makedirs(opt_root, exist_ok=True)
            infos = []

            # ====================================================
            # OpenUtau Configuration Processing (oto.ini / character files)
            # ====================================================
            if dir_path != "" and os.path.isdir(dir_path):
                oto_path_src = os.path.join(dir_path, "oto.ini")
                oto_path_dst = os.path.join(opt_root, "oto.ini")
                if os.path.exists(oto_path_src) and not os.path.exists(oto_path_dst):
                    try:
                        shutil.copy2(oto_path_src, oto_path_dst)
                        PoliteLogger.system("คัดลอกไฟล์ oto.ini สำหรับคลังเสียงสำเร็จ (Copied oto.ini for Voicebank successfully)")
                        infos.append("🗂️ นำไฟล์ oto.ini มารวมด้วยสำเร็จ (Included oto.ini successfully)")
                    except Exception as e:
                        PoliteLogger.error(f"ไม่สามารถคัดลอกไฟล์ oto.ini ได้ (Failed to copy oto.ini): {e}")

                for config_file in ["character.txt", "character.yaml", "readme.txt"]:
                    conf_src = os.path.join(dir_path, config_file)
                    conf_dst = os.path.join(opt_root, config_file)
                    if os.path.exists(conf_src) and not os.path.exists(conf_dst):
                        try:
                            shutil.copy2(conf_src, conf_dst)
                        except Exception:
                            pass

            # ====================================================
            # การคัดกรองคิวไฟล์: ให้ความสำคัญกับ paths ที่ส่งมาจาก Smart Batch Resume ก่อนเสมอ
            # ====================================================
            resolved_paths = []
            if paths and len(paths) > 0:
                for p_item in paths:
                    p_str = p_item.name if hasattr(p_item, "name") else str(p_item)
                    if os.path.isfile(p_str):
                        resolved_paths.append(p_str)
                    elif dir_path and os.path.isfile(os.path.join(dir_path, p_str)):
                        resolved_paths.append(os.path.join(dir_path, p_str))
            elif dir_path != "" and os.path.isdir(dir_path):
                resolved_paths = [
                    os.path.join(dir_path, name)
                    for name in sorted(os.listdir(dir_path))
                    if name.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg"))
                ]

            paths = resolved_paths
            total_files = len(paths)
            if total_files == 0:
                PoliteLogger.warning("ไม่พบไฟล์เสียงในคิวการประมวลผล (No valid audio files found in queue)")
                yield "⚠️ ไม่พบไฟล์เสียงที่รองรับในคิวการทำงาน"
                return

            start_banner = (
                "🚀 [เริ่มต้นกระบวนการเรนเดอร์ (Batch Render Initialized)]\n"
                f"📁 โฟลเดอร์ต้นทาง: {dir_path}\n"
                f"🎯 โฟลเดอร์ปลายทาง: {opt_root}\n"
                f"📊 จำนวนไฟล์เสียงที่ตรวจพบ: {total_files} ไฟล์\n"
                f"⚙️ โมเดล: {sid} | Format: {format1.upper()} | F0 Method: {f0_method} | Key Shift: {f0_up_key}\n"
                + "=" * 60 + "\n"
                + "⏳ กำลังเริ่มประมวลผลไฟล์ลำดับแรก..."
            )
            PoliteLogger.system(f"เริ่มต้นประมวลผลคลังเสียงแบบกลุ่ม ทั้งหมด {total_files} ไฟล์ (Starting batch render for {total_files} files)")
            yield start_banner

            for i, path in enumerate(paths):
                if not self.is_rendering:
                    cancel_msg = f"⚠️ งานถูกยกเลิกโดยผู้ใช้งาน (Cancelled by user at {i}/{total_files})"
                    PoliteLogger.warning(cancel_msg)
                    infos.insert(0, f"🛑 {cancel_msg}")
                    yield "⚠️ [ระงับการทำงาน] กระบวนการเรนเดอร์ถูกหยุดโดยผู้ใช้งาน\n" + "=" * 60 + "\n" + "\n".join(infos)
                    return

                file_name = os.path.basename(path)
                file_ext = os.path.splitext(file_name)[1].lstrip(".").upper() or "AUDIO"
                file_start_time = time.time()

                try:
                    f_size = os.path.getsize(path)
                    if f_size < 1024:
                        size_str = f"{f_size} B"
                    elif f_size < 1024 * 1024:
                        size_str = f"{f_size / 1024:.1f} KB"
                    else:
                        size_str = f"{f_size / (1024 * 1024):.2f} MB"
                except Exception:
                    size_str = "N/A"

                try:
                    a_info = sf.info(path)
                    dur_val = a_info.duration
                    d_mm, d_ss = divmod(int(dur_val), 60)
                    dur_str = f"{d_mm:02d}:{d_ss:02d} ({dur_val:.1f}s)"
                except Exception:
                    dur_str = "N/A"

                pct = ((i + 1) / total_files) * 100

                current_active_status = (
                    f"🎙️ [กำลังเรนเดอร์ไฟล์ที่ {i+1}/{total_files} | ความคืบหน้า {pct:.1f}%]\n"
                    f"   • ชื่อไฟล์ (Name): {file_name}\n"
                    f"   • ระยะเวลาเสียง (Duration): {dur_str}\n"
                    f"   • ประเภท (Type): {file_ext} -> {format1.upper()}\n"
                    f"   • ขนาดไฟล์ (Size): {size_str}\n"
                    + "=" * 60 + "\n"
                    + ("\n".join(infos[:12]) if infos else "⏳ กำลังแปลงสัญญาณเสียงด้วยโมเดล RVC...")
                )
                yield current_active_status

                try:
                    PoliteLogger.progress(i + 1, total_files, f"{file_name} ({dur_str}, {size_str})")
                except Exception:
                    pass

                info, opt = self.vc_single(
                    sid_num, path, f0_up_key, None, f0_method, file_index, file_index2,
                    index_rate, filter_radius, resample_sr, rms_mix_rate, protect
                )

                status_msg = "Completed" if "สำเร็จ" in info and "ไม่สำเร็จ" not in info else "Failed"

                if status_msg == "Completed":
                    try:
                        tgt_sr, audio_opt = opt
                        base_name = os.path.splitext(file_name)[0]
                        out_path = os.path.join(opt_root, f"{base_name}.{format1}")

                        if format1 in ["wav", "flac"]:
                            sf.write(out_path, audio_opt, tgt_sr)
                        else:
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(out_path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except Exception as e:
                        info += f"\nเกิดข้อผิดพลาดในการบันทึกไฟล์: {str(e)}"
                        status_msg = "Failed"

                file_elapsed_sec = int(time.time() - file_start_time)
                mm, ss = divmod(file_elapsed_sec, 60)
                time_str = f"{mm:02d}:{ss:02d}"

                if status_msg == "Completed":
                    success_count += 1
                    log_entry = (
                        f"✅ [{i+1}/{total_files}] สำเร็จ: {file_name} "
                        f"| ⏱️ ความยาว: {dur_str} | ชนิด: {file_ext} | ขนาด: {size_str} "
                        f"| เรนเดอร์ใน: {time_str}"
                    )
                    PoliteLogger.success(f"Rendered [{i+1}/{total_files}]: {file_name} ({dur_str}) in {time_str}")
                else:
                    fail_count += 1
                    clean_err = info.strip().replace("\n", " ")
                    log_entry = (
                        f"❌ [{i+1}/{total_files}] ขัดข้อง: {file_name} "
                        f"| ความยาว: {dur_str} | ชนิด: {file_ext} | ขนาด: {size_str}\n"
                        f"   ⚠️ สาเหตุ: {clean_err}"
                    )
                    PoliteLogger.error(f"[Render Failed] File: {file_name} | Error: {clean_err}")

                infos.insert(0, log_entry)

                display_output = (
                    f"📊 [ไฟล์ที่ {i+1}/{total_files} ({pct:.1f}%)] | สำเร็จ: {success_count} 🟢 | ขัดข้อง: {fail_count} 🔴\n"
                    + "=" * 60 + "\n"
                    + "\n".join(infos[:15])
                )
                yield display_output

            batch_elapsed_sec = int(time.time() - batch_start_time)
            b_mm, b_ss = divmod(batch_elapsed_sec, 60)
            batch_time_str = f"{b_mm:02d}:{b_ss:02d}"

            final_summary = (
                "🎉 [การเรนเดอร์คลังเสียงเสร็จสมบูรณ์ 100% (Batch Render Completed)]\n"
                + "=" * 60 + "\n"
                + f"• จำนวนไฟล์ทั้งหมด: {total_files} ไฟล์\n"
                + f"• สำเร็จเรียบร้อย (Success): {success_count} ไฟล์ 🟢\n"
                + f"• เกิดข้อผิดพลาด (Failed): {fail_count} ไฟล์ 🔴\n"
                + f"• เวลารวมทั้งสิ้น (Total Elapsed): {batch_time_str}\n"
                + "=" * 60 + "\n"
                + "\n".join(infos)
            )
            PoliteLogger.success(f"ประมวลผลแบทช์เสร็จสมบูรณ์ {total_files} ไฟล์ (สำเร็จ: {success_count}, พลาด: {fail_count}) ในเวลา {batch_time_str}")
            yield final_summary
        except Exception as e:
            err_ui = PoliteLogger.error(
                msg="เกิดข้อผิดพลาดในการประมวลผลกลุ่ม (Batch render error)",
                exc=e,
                context="Inference - Batch Processing"
            )
            critical_alert = (
                f"{err_ui}\n"
                + "=" * 60 + "\n"
                + "คำแนะนำ: ตรวจสอบไฟล์เสียงต้นทาง และหน่วยความจำ VRAM ของ GPU"
            )
            yield critical_alert
        finally:
            self.is_rendering = False
            clear_f0_cache()
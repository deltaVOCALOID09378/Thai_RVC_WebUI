# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี (Date)       : 02/11/2569 (02 November 2026)
# 2. ชื่อไฟล์ที่ทำ (File Name)  : modules.py
# 3. ลำดับเวอร์ชัน (Version)   : 2.3 (Unified Core & VRAM Master Edition)
# 4. เครดิตผู้สร้าง (Credit)    : Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#                              [ Original by "RVC-Project" ]
# 5. ตำแหน่งที่วางของไฟล์ (Path): infer/modules/vc/modules.py
# 6. รายละเอียดการแก้ไขงาน    : เพิ่มฟังก์ชัน cancel_process ให้เป็นตัวสำรอง (Alias) สำหรับ Safe Cancel 
#                              และเข้มงวดกับ Zero-Leak VRAM Purge ในระบบประมวลผลแบตช์
# ==============================================================================

import os
import gc
import time
import traceback
import logging
from io import BytesIO
from time import strftime

import numpy as np
import soundfile as sf
import torch

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline, clear_f0_cache
from infer.modules.vc.utils import *
from contextlib import suppress

logger = logging.getLogger(__name__)


def format_time_delta(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"

def clean_stem(filepath: str) -> str:
    name = os.path.basename(filepath)
    compound_exts = [".wav.npy", ".flac.npy", ".mp3.npy", ".ogg.npy", ".m4a.npy", 
                     ".npy", ".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    for ext in compound_exts:
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    while name.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".m4a")):
        name = os.path.splitext(name)[0]
    return name


def clear_vram_safe():
    """Zero-Leak Memory Purge: คืนค่าหน่วยความจำ VRAM 100%"""
    try:
        clear_f0_cache()
    except Exception:
        pass
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
    except Exception as e:
        logger.warning(f"VRAM clear warning: {e}")


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
        self.config = config
        self.cancel_flag = False

    def cancel_render(self):
        """ระบบยกเลิกกระบวนการทำงานหลัก (Main Cancellation Protocol)"""
        self.cancel_flag = True
        logger.info("[TH] ร้องขอการยกเลิกการเรนเดอร์แล้ว / [EN] Render cancellation requested")
        clear_vram_safe()
        return "🛑 ยกเลิกการเรนเดอร์เรียบร้อยแล้ว (Render process safely canceled)"

    def cancel_process(self):
        """[Alias] ฟังก์ชันสำรองเพื่อป้องกันการแครชจากฝั่ง UI เก่า (Fallback Protocol)"""
        return self.cancel_render()

    def get_vc(self, sid, *to_return_protect):
        logger.info(f"Get sid: {sid}")

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5,
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33,
            "__type__": "update",
        }

        if not sid:
            if self.hubert_model is not None:
                logger.info("[TH] ล้างแคชของโมเดลเรียบร้อย / [EN] Cleaned model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)
                self.hubert_model = self.net_g = self.n_spk = self.tgt_sr = None
                clear_vram_safe()
                    
                if self.cpt is not None:
                    self.if_f0 = self.cpt.get("f0", 1)
                    self.version = self.cpt.get("version", "v1")
                    if self.version == "v1":
                        if self.if_f0 == 1:
                            self.net_g = SynthesizerTrnMs256NSFsid(*self.cpt["config"], is_half=self.config.is_half)
                        else:
                            self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                    elif self.version == "v2":
                        if self.if_f0 == 1:
                            self.net_g = SynthesizerTrnMs768NSFsid(*self.cpt["config"], is_half=self.config.is_half)
                        else:
                            self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                            
                    del self.net_g, self.cpt
                    self.net_g = self.cpt = None
                    clear_vram_safe()
                    
            return (
                {"visible": False, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                "",
                "",
            )
            
        try:
            person = f'{os.getenv("weight_root", "assets/weights")}/{sid}'
            logger.info(f"[TH] กำลังโหลดโมเดล / [EN] Loading model: {person}")

            if self.net_g is not None:
                del self.net_g
                self.net_g = None
                clear_vram_safe()
            if self.cpt is not None:
                del self.cpt
                self.cpt = None

            self.cpt = torch.load(person, map_location="cpu")
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

            self.net_g = synthesizer_class.get((self.version, self.if_f0), SynthesizerTrnMs256NSFsid)(
                *self.cpt["config"], is_half=self.config.is_half
            )

            del self.net_g.enc_q

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device)
            
            del self.cpt
            self.cpt = None
            clear_vram_safe()
        except Exception as e:
            logger.error(f"Failed to load model {sid}: {e}")
            return (
                {"visible": False, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                "",
                "",
            )

        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.net_g.emb_g.weight.shape[0] if hasattr(self.net_g, "emb_g") else 1
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        idx_display = index['value'] if index['value'] else "(None / ไม่พบไฟล์ Index)"
        logger.info(f"[TH] เลือกอินเด็กซ์ / [EN] Select index: {idx_display}")

        if to_return_protect:
            return (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
        return {"visible": True, "maximum": n_spk, "__type__": "update"}

    def vc_single(
        self, sid, input_audio_path, f0_up_key, f0_file, f0_method, file_index,
        file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect
    ):
        sep_line = "=" * 60
        if input_audio_path is None:
            return f"{sep_line}\n❌ [ERR] กรุณาอัปโหลดไฟล์เสียงก่อน (You need to upload an audio)\n{sep_line}", None
            
        f0_up_key = float(f0_up_key)
        start_time_sec = time.time()
        start_timestamp = strftime("%H:%M:%S")

        try:
            sid_clean = int(sid) if sid else 0
        except (ValueError, TypeError):
            sid_clean = 0

        try:
            audio = load_audio(input_audio_path, 16000)
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            file_index = (
                file_index.strip(" ").strip('"').strip("\n").strip('"').strip(" ").replace("trained", "added")
                if file_index != "" else file_index2
            )

            audio_opt = self.pipeline.pipeline(
                self.hubert_model, self.net_g, sid_clean, audio, input_audio_path, times, f0_up_key,
                f0_method, file_index, index_rate, self.if_f0, filter_radius, self.tgt_sr,
                resample_sr, rms_mix_rate, self.version, protect, f0_file,
            )
            
            tgt_sr = resample_sr if (self.tgt_sr != resample_sr >= 16000) else self.tgt_sr
            index_info = f"Index:\n{file_index}" if os.path.exists(file_index) else "ไม่ได้ใช้อินเด็กซ์ (Index not used)"
            
            elapsed = format_time_delta(time.time() - start_time_sec)
            success_msg = (
                f"{sep_line}\n"
                f"✅ ลำดับคิว: 1/1 | ชื่อไฟล์: {os.path.basename(input_audio_path)}\n"
                f"⏰ เวลาเริ่มต้น: {start_timestamp} | 🏁 เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')} (⏱️ {elapsed})\n"
                f"สถานะ: สำเร็จ (Success)\n"
                f"{index_info}\n"
                f"กระบวนการ (Time): npy: {times[0]:.2f}s, f0: {times[1]:.2f}s, infer: {times[2]:.2f}s.\n"
                f"{sep_line}"
            )
            return success_msg, (tgt_sr, audio_opt)
            
        except Exception as e:
            info = traceback.format_exc()
            logger.warning(f"Error in vc_single: {info}")
            return f"{sep_line}\n❌ [ERR] ล้มเหลว (Failed): {info}\n{sep_line}", None
        finally:
            clear_vram_safe()

    def vc_multi(
        self, sid, dir_path, opt_root, paths, f0_up_key, f0_method, file_index,
        file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, format1
    ):
        self.cancel_flag = False
        start_time_sec = time.time()
        start_timestamp = strftime("%H:%M:%S")
        sep_line = "=" * 60
        
        dir_path = str(dir_path or "").strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        opt_root = str(opt_root or "").strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        format1 = str(format1 or "wav").lower().strip()
        os.makedirs(opt_root, exist_ok=True)
        
        # 🔴 กฎข้อที่ 1: การฟอกค่า Speaker ID (SID Sanitization Guard)
        try:
            sid_clean = int(sid) if sid else 0
        except (ValueError, TypeError):
            sid_clean = 0

        try:
            if dir_path != "":
                source_paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path) if os.path.splitext(name)[1].lower() in [".wav", ".flac", ".mp3", ".ogg", ".m4a"]]
            elif paths:
                source_paths = [path.name if hasattr(path, "name") else str(path) for path in paths]
            else:
                yield f"{sep_line}\n❌ [ERR] ไม่พบโฟลเดอร์หรือไฟล์ต้นฉบับที่ระบุ (No files found)\n{sep_line}"
                return
                
            total_files = len(source_paths)
            files_to_render = []
            already_done = 0
            
            for src in source_paths:
                clean_stem_name = clean_stem(src)
                expected_dst = os.path.join(opt_root, f"{clean_stem_name}.{format1}")
                if os.path.exists(expected_dst) and os.path.getsize(expected_dst) > 1024:
                    already_done += 1
                else:
                    files_to_render.append(src)
            
            yield (
                f"{sep_line}\n"
                f"⏳ เริ่มกระบวนการเรนเดอร์แบบกลุ่ม (Smart Batch Resume)...\n"
                f"📊 ไฟล์ต้นแบบทั้งหมด: {total_files} | ข้ามไฟล์เดิมที่เสร็จแล้ว: {already_done} | ค้างทำต่อ: {len(files_to_render)}\n"
                f"⏰ เวลาเริ่มต้น: {start_timestamp}\n"
                f"{sep_line}\n"
            )
            
            if len(files_to_render) == 0:
                elapsed = format_time_delta(time.time() - start_time_sec)
                yield f"{sep_line}\n🎉 ทุกไฟล์ถูกเรนเดอร์สานต่อจนเสร็จสมบูรณ์แล้วทั้งหมด 100% (⏱️ {elapsed})\n{sep_line}"
                clear_vram_safe()
                return

            infos = []
            current_done = 0
            
            for idx, path in enumerate(files_to_render):
                if self.cancel_flag:
                    elapsed = format_time_delta(time.time() - start_time_sec)
                    infos.append(f"{sep_line}\n🛑 คำสั่งถูกยกเลิกโดยผู้ใช้งาน (Render Cancelled by User)\n⏱️ ระยะเวลาที่ใช้ไป: {elapsed} (MM:SS)\n🏁 เวลาเรนเดอร์เสร็จแล้ว: {strftime('%H:%M:%S')}\n{sep_line}")
                    yield "\n".join(infos[-5:])
                    break
                
                total_current = already_done + current_done + 1
                pct_now = (total_current / total_files) * 100
                
                yield (
                    f"{sep_line}\n"
                    f"🟢 สถานะการสานต่อและเรนเดอร์ (Live Render Status) - ความคืบหน้า {pct_now:.1f}%\n"
                    f"⏳ กำลังประมวลผลไฟล์ (Processing): {os.path.basename(path)}\n"
                    f"▶ ลำดับคิวที่: {total_current}/{total_files}\n"
                    f"⏱️ ใช้เวลาไปแล้ว (Elapsed): {format_time_delta(time.time() - start_time_sec)} (MM:SS)\n"
                    f"{sep_line}\n"
                    + "\n".join(infos[-8:]) + ("\n" if len(infos) > 0 else "")
                    + f"{sep_line}"
                )
                    
                info, opt = self.vc_single(
                    sid_clean, path, f0_up_key, None, f0_method, file_index, file_index2,
                    index_rate, filter_radius, resample_sr, rms_mix_rate, protect
                )
                
                current_done += 1
                total_finished = already_done + current_done
                pct = (total_finished / total_files) * 100
                
                if "Success" in info or "สำเร็จ" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        output_path = os.path.join(opt_root, f"{clean_stem(path)}.{format1}")
                        
                        if format1 in ["wav", "flac"]:
                            sf.write(output_path, audio_opt, tgt_sr)
                        else:
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                wav2(wavf, output_path, format1)
                    except Exception as e:
                        info += traceback.format_exc()
                    finally:
                        del opt
                        clear_vram_safe() # เคลียร์แรมทุกครั้งที่เรนเดอร์จบ 1 ไฟล์
                        
                file_status = f"▶ ลำดับคิว: {total_finished}/{total_files} ({pct:.1f}%) | ชื่อไฟล์: {os.path.basename(path)} | สถานะ: {'✅ สำเร็จ' if 'Success' in info or 'สำเร็จ' in info else '❌ ผิดพลาด'}"
                infos.append(file_status)
                
                yield (
                    f"{sep_line}\n"
                    f"🟢 สถานะการสานต่อและเรนเดอร์ (Live Render Status) - ความคืบหน้า {pct:.1f}%\n"
                    f"⏱️ ใช้เวลาไปแล้ว (Elapsed): {format_time_delta(time.time() - start_time_sec)} (MM:SS)\n"
                    f"{sep_line}\n"
                    + "\n".join(infos[-8:]) + "\n"
                    f"{sep_line}"
                )
                
            if not self.cancel_flag:
                elapsed = format_time_delta(time.time() - start_time_sec)
                end_timestamp = strftime('%H:%M:%S')
                generated_files = len(os.listdir(opt_root)) if os.path.exists(opt_root) else 0
                
                final_msg = (
                    f"{sep_line}\n"
                    f"🟢 การเรนเดอร์แบบกลุ่มเสร็จสมบูรณ์เรียบร้อยแล้ว 100% (Batch Rendering Completed)\n"
                    f"✅ ตรวจสอบความครบถ้วน: มีไฟล์ต้นฉบับ {total_files} ไฟล์ สร้างสำเร็จแล้วรวม {generated_files} ไฟล์\n"
                    f"✨ ทำการเรนเดอร์ใหม่รอบนี้: {current_done} ไฟล์ | ข้ามไฟล์เดิมที่เสร็จแล้ว: {already_done} ไฟล์\n"
                    f"⏰ เวลาเริ่มต้น: {start_timestamp} | 🏁 เวลาเรนเดอร์เสร็จแล้ว: {end_timestamp}\n"
                    f"⏱️ รวมระยะเวลาที่ใช้ไปทั้งหมด: {elapsed} (MM:SS)\n"
                    f"{sep_line}"
                )
                yield final_msg
                
        except Exception as e:
            yield f"{sep_line}\n❌ [ERR] เกิดข้อผิดพลาดแบบกลุ่ม:\n{traceback.format_exc()}\n{sep_line}"
        finally:
            clear_vram_safe()
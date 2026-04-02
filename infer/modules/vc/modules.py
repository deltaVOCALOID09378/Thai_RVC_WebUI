import traceback
import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *


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
        self.cancel_batch = False  # ระบบเก็บสถานะการสั่งยกเลิกงาน

    def cancel_process(self):
        """รับคำสั่งยกเลิกจาก UI และหยุดการทำงานทันที"""
        self.cancel_batch = True
        return "🛑 ได้รับคำสั่งยกเลิกแล้ว... ระบบกำลังเคลียร์คิวงาน (Cancellation requested... stopping pending tasks)"

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if self.hubert_model is not None:
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)
                self.hubert_model = self.net_g = self.n_spk = self.tgt_sr = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {"visible": True, "value": to_return_protect0, "__type__": "update"},
                {"visible": True, "value": to_return_protect1, "__type__": "update"},
                "",
                "",
            )

        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

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

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)
        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])

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
        if input_audio_path is None:
            return "You need to upload an audio", None
        f0_up_key = int(f0_up_key)
        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
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

            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr

            expected_length = int((len(audio) / 16000) * tgt_sr)
            if len(audio_opt) > expected_length:
                audio_opt = audio_opt[:expected_length]
            elif len(audio_opt) < expected_length:
                pad_length = expected_length - len(audio_opt)
                audio_opt = np.pad(audio_opt, (0, pad_length), mode="constant")

            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)

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
        self.cancel_batch = False  # รีเซ็ตสถานะก่อนเริ่มงานใหม่ทุกครั้ง
        try:
            dir_path = dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            os.makedirs(opt_root, exist_ok=True)

            try:
                if dir_path != "":
                    input_paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    input_paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                input_paths = [path.name for path in paths]

            audio_paths = [
                p
                for p in input_paths
                if p.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
            ]
            if not audio_paths:
                yield "❌ ไม่พบไฟล์เสียงที่รองรับในโฟลเดอร์นำเข้า (No valid audio files found)."
                return

            file_statuses = {
                os.path.basename(p): "⏳ รอดำเนินการ (Pending)" for p in audio_paths
            }

            def generate_report():
                report = (
                    "📊 รายงานสถานะจากโฟลเดอร์ Import โดยตรง (Live Status Report):\n"
                )
                report += "=" * 65 + "\n"
                for f_name, status in file_statuses.items():
                    report += f"🎵 {f_name} : {status}\n"
                report += "=" * 65
                return report

            yield generate_report()

            for path in audio_paths:
                # ----------------- เช็คเบรกฉุกเฉิน -----------------
                if self.cancel_batch:
                    yield generate_report() + "\n🛑 กระบวนการถูกยกเลิกกลางคันโดยผู้ใช้! (Process stopped by user!)"
                    break
                # --------------------------------------------------

                filename = os.path.basename(path)
                name_without_ext = os.path.splitext(filename)[0]
                expected_out_path = os.path.join(
                    opt_root, f"{name_without_ext}.{format1}"
                )

                if os.path.exists(expected_out_path):
                    file_statuses[filename] = "⏭️ ข้าม (Already Exists)"
                    yield generate_report()
                    continue

                file_statuses[filename] = "🔄 กำลังเรนเดอร์ (Processing...)"
                yield generate_report()

                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )

                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(expected_out_path, audio_opt, tgt_sr)
                        else:
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(expected_out_path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                        file_statuses[filename] = "✅ สำเร็จ (Done)"
                    except:
                        file_statuses[filename] = "❌ ล้มเหลวตอนบันทึก (Save Error)"
                        traceback.print_exc()
                else:
                    file_statuses[filename] = "❌ ล้มเหลว (Inference Error)"

                yield generate_report()

            if not self.cancel_batch:
                yield generate_report() + f"\n🎉 ดำเนินการเสร็จสมบูรณ์ 100%! (All Processed)\n📁 ไฟล์ถูกบันทึกที่: {opt_root}"

        except:
            yield f"❌ ข้อผิดพลาดระบบ (System Error):\n{traceback.format_exc()}"

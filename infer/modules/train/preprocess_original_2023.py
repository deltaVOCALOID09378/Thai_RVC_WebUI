# -*- coding: utf-8 -*-
# ==============================================================================
# 1. วัน/เดือน/ปี: 11/10/2569 | 11/10/2026
# 2. ชื่อไฟล์: preprocess.py
# 3. ลำดับเวอร์ชัน: 3.4
# 4. Made And Checked By DELTA SYNTH [ Patiphat Wongyai ] & All Code Agentic AI Engine
#    [ Original by "RVC-Project" ]
# 5. ตำแหน่งที่วางไฟล์: infer/modules/train/preprocess.py
# 6. รายละเอียดการแก้ไขงาน: อัปเกรดระบบแจ้งเตือนสองภาษา (Thai/English) พร้อมอีโมจิ, ป้องกันการเขียนชิ้นส่วนสุดท้ายซ้ำ และรักษาโครงสร้างเดิมให้เสถียรที่สุด
# ==============================================================================

import multiprocessing
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import traceback

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa

now_dir = os.getcwd()
sys.path.append(now_dir)

inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

mutex = multiprocessing.Lock()
log_file_path = os.path.join(exp_dir, "preprocess.log")

def println(strr: str) -> None:
    mutex.acquire()
    try:
        try:
            print(strr, flush=True)
        except Exception:
            try:
                print(str(strr).encode(getattr(sys.stdout, "encoding", None) or "utf-8", errors="replace").decode(getattr(sys.stdout, "encoding", None) or "utf-8"), flush=True)
            except Exception:
                pass
        try:
            with open(log_file_path, "a+", encoding="utf-8", errors="replace") as f:
                f.write(f"{strr}\n")
        except Exception:
            pass
    finally:
        mutex.release()

def apply_edge_fade(audio_chunk: np.ndarray, sample_rate: int, fade_ms: float = 5.0) -> np.ndarray:
    fade_len = int(sample_rate * fade_ms / 1000.0)
    if len(audio_chunk) <= fade_len * 2:
        return audio_chunk
    fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
    fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
    chunk = audio_chunk.copy()
    chunk[:fade_len] *= fade_in
    chunk[-fade_len:] *= fade_out
    return chunk

class PreProcess:
    def __init__(self, sr: int, exp_dir: str, per: float = 3.7):
        self.slicer = Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.95
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs").replace("\\", "/")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs").replace("\\", "/")
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio: np.ndarray, idx0: int, idx1: int) -> None:
        if len(tmp_audio) == 0:
            return
        tmp_max = np.abs(tmp_audio).max()
        if np.isnan(tmp_max) or np.isinf(tmp_max) or tmp_max > 2.5:
            println(f"⚠️ [FILTER] กรองเสียงที่มีความผิดปกติของแอมพลิจูดออก (Filtered abnormal amplitude): {idx0}_{idx1} (Max: {tmp_max:.2f})")
            return
        rms = np.sqrt(np.mean(tmp_audio ** 2) + 1e-9)
        if rms < 0.002:
            return
        if tmp_max > 1e-5:
            tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio
        tmp_audio = apply_edge_fade(tmp_audio, self.sr, fade_ms=5.0)
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sr, tmp_audio.astype(np.float32))
        try:
            tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000, res_type="soxr_vhq")
        except Exception:
            tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", 16000, tmp_audio_16k.astype(np.float32))

    def pipeline(self, path: str, idx0: int) -> None:
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)
            audio = signal.medfilt(audio, kernel_size=3)
            gate_threshold = 0.0005
            audio = np.where(np.abs(audio) < gate_threshold, audio * 0.1, audio)
            idx1 = 0
            for audio_segment in self.slicer.slice(audio):
                if len(audio_segment) == 0: continue
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio_segment[start:]) > self.tail * self.sr:
                        tmp_audio = audio_segment[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio_segment[start:]
                        if len(tmp_audio) > 0:
                            self.norm_write(tmp_audio, idx0, idx1)
                            idx1 += 1
                        break
            println(f"✅ สำเร็จ (Processed successfully) -> {os.path.basename(path)}")
        except Exception:
            println(f"❌ เกิดข้อผิดพลาด (Error) -> {os.path.basename(path)}:\n{traceback.format_exc()}")

    def pipeline_mp(self, infos: list) -> None:
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root: str, n_p: int) -> None:
        try:
            valid_audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".aiff"}
            raw_entries = sorted(list(os.listdir(inp_root)))
            audio_entries = []
            ignored_entries = []
            for name in raw_entries:
                full_p = os.path.join(inp_root, name)
                ext = os.path.splitext(name)[1].lower()
                if os.path.isfile(full_p) and ext in valid_audio_exts:
                    if os.path.getsize(full_p) > 512:
                        audio_entries.append(name)
                    else:
                        ignored_entries.append(f"{name} (ไฟล์ขนาด 0 ไบต์หรือเสียหาย / 0-byte or corrupted file)")
                else:
                    ignored_entries.append(f"{name} (ไม่ใช่ไฟล์เสียงหรือเป็นโฟลเดอร์ / Not an audio file or is a directory)")

            if ignored_entries:
                println("-" * 65)
                println(f"🔍 [DATASET FILTER] คัดกรองไฟล์ที่ไม่เกี่ยวข้องออก (Ignored) {len(ignored_entries)} รายการ (Files):")
                for ign in ignored_entries[:10]:
                    println(f"   - ข้าม (Skip): {ign}")
                if len(ignored_entries) > 10:
                    println(f"   ... และอีก (and) {len(ignored_entries)-10} รายการ (more)")
                println("-" * 65)

            infos = [(os.path.join(inp_root, name), idx) for idx, name in enumerate(audio_entries)]
            println(f"🎙️ ตรวจพบไฟล์เสียงที่พร้อมสำหรับฝึกสอน (Valid audio files detected): {len(infos)} ไฟล์ (Files) | Thread(s): {n_p}")

            if noparallel or n_p <= 1:
                self.pipeline_mp(infos)
            else:
                ps = []
                for i in range(n_p):
                    sub_infos = infos[i::n_p]
                    if sub_infos:
                        p = multiprocessing.Process(target=self.pipeline_mp, args=(sub_infos,))
                        ps.append(p)
                        p.start()
                for p in ps:
                    p.join()
        except Exception:
            println(f"❌ ล้มเหลว (Failed): {traceback.format_exc()}")

def preprocess_trainset(inp_root: str, sr: int, n_p: int, exp_dir: str, per: float) -> None:
    start_time = time.time()
    pp = PreProcess(sr, exp_dir, per)
    println("=" * 65)
    println("🚀 เริ่มต้นกระบวนการเตรียมและปรับปรุงคุณภาพชุดข้อมูล (Starting Preprocess)...")
    println("=" * 65)
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    println("=" * 65)
    println("🎉 สิ้นสุดกระบวนการเตรียมชุดข้อมูลเรียบร้อยแล้ว (Preprocess Ended)")
    println(f"⏱️ ระยะเวลาที่ใช้ไป (Time Elapsed): {mins:02d}:{secs:02d} (MM:SS)")
    println("=" * 65)

if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
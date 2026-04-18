import os
import sys
import multiprocessing
import traceback
import logging
import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

# ตั้งค่าให้โฟลเดอร์ปัจจุบันอยู่ใน Path เพื่อให้เรียกใช้โมดูลของ RVC ได้อย่างถูกต้อง
now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

# ระบบป้องกันการเขียน Log ชนกันเมื่อทำงานแบบหลายเธรด (Thread-safe Logging)
mutex = multiprocessing.Lock()


def println(strr, log_path):
    """ฟังก์ชันสำหรับพิมพ์ข้อความลงคอนโซลและเขียนลงไฟล์ Log อย่างปลอดภัย"""
    with mutex:
        print(strr)
        with open(log_path, "a+", encoding="utf-8") as f:
            f.write("%s\n" % strr)


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.0):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75

        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        self.log_path = "%s/preprocess.log" % exp_dir

        # สร้างโฟลเดอร์เป้าหมายหากยังไม่มี
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        """ทำการปรับระดับเสียง (Normalize) และบันทึกไฟล์"""
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            println(
                "[ข้าม/Skipped] ไฟล์ %s_%s ถูกกรองออกเนื่องจากระดับเสียงสูงเกินมาตรฐาน (Filtered due to high volume: %s)"
                % (idx0, idx1, tmp_max),
                self.log_path,
            )
            return

        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio

        # บันทึกไฟล์เสียงสำหรับฝึกสอน
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )

        # ปรับ Sample Rate เป็น 16000Hz สำหรับโมเดล Hubert
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        """กระบวนการสกัดและหั่นไฟล์เสียง (Audio Slicing Pipeline)"""
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio_slice in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio_slice[start:]) > self.tail * self.sr:
                        tmp_audio = audio_slice[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio_slice[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            println("[สำเร็จ/Success] ประมวลผลไฟล์สมบูรณ์: %s" % path, self.log_path)
        except Exception as e:
            println(
                "[ล้มเหลว/Error] เกิดข้อผิดพลาดกับไฟล์ %s:\n%s"
                % (path, traceback.format_exc()),
                self.log_path,
            )

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p, noparallel):
        """กระจายงานไปยังซีพียูหลายคอร์ (Multiprocessing Handler)"""
        try:
            infos = [
                ("%s/%s" % (inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except Exception as e:
            println(
                "[ล้มเหลว/Error] ระบบประมวลผลแบบกลุ่มขัดข้อง (Batch processing failed):\n%s"
                % traceback.format_exc(),
                self.log_path,
            )


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, noparallel):
    """ฟังก์ชันหลักสำหรับเริ่มเตรียมข้อมูลฝึกสอน"""
    pp = PreProcess(sr, exp_dir, per)
    log_path = "%s/preprocess.log" % exp_dir

    # สร้างไฟล์ Log ใหม่หรือเขียนทับเพื่อความสะอาด
    open(log_path, "w", encoding="utf-8").close()

    println("=" * 60, log_path)
    println(
        "[ข้อมูล] เริ่มกระบวนการเตรียมข้อมูลฝึกสอน (Starting dataset preprocessing)...",
        log_path,
    )
    println(f"Command Arguments: {sys.argv}", log_path)
    pp.pipeline_mp_inp_dir(inp_root, n_p, noparallel)
    println("[ข้อมูล] สิ้นสุดกระบวนการเตรียมข้อมูล (End of preprocessing)", log_path)
    println("=" * 60, log_path)


if __name__ == "__main__":
    # การย้ายมารับค่าตัวแปรภายในบล็อกนี้ จะช่วยป้องกันบัค Multiprocessing ในระบบ Windows ได้ 100%
    inp_root = sys.argv[1]
    sr = int(sys.argv[2])
    n_p = int(sys.argv[3])
    exp_dir = sys.argv[4]
    noparallel = sys.argv[5] == "True"
    per = float(sys.argv[6])

    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, noparallel)

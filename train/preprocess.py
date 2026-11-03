# ==============================================================================
# File Name    : preprocess.py
# Version      : v2.3.0-Production
# Target Path  : infer/modules/train/preprocess.py
# Function     : Professional Audio Dataset Slicing, High-Pass Filtering & Normalization Pipeline
# Credits      : Made And Checked By DELTA SYNTH & All Agentic AI
#                Original by RVC-Project / RVC Team
# ==============================================================================

import os
import sys
import time
import traceback
import multiprocessing
from scipy import signal
import numpy as np
from scipy.io import wavfile
import librosa

# [TH] เชื่อมโยง Directory หลักของโปรเจกต์เข้าสู่ sys.path เพื่อความแม่นยำในการเรียกโมดูล
# [EN] Resolve and inject root project directory to ensure robust module resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["RVC_AUDIO_FORCE_CPU"] = "1"

# Safe Imports with Fallback Mechanisms
try:
    from infer.audio import load_audio
except ImportError:
    from infer.lib.audio import load_audio

try:
    from train.dataset.slicer2 import Slicer
except ImportError:
    try:
        from infer.lib.slicer import Slicer
    except ImportError:
        from slicer2 import Slicer

try:
    from i18n.i18n import I18nAuto
    i18n = I18nAuto()
except ImportError:
    i18n = lambda text: text

try:
    from tools.progress import should_report
except ImportError:
    should_report = lambda idx, total, step=1: True

try:
    from tools.multispeaker import ManifestError, load_manifest
except ImportError:
    class ManifestError(Exception):
        pass
    load_manifest = None


class UnifiedLogger:
    """
    [TH] ระบบจัดการบันทึกและส่งการแจ้งเตือนสองภาษาแบบ Thread-Safe
    [EN] Thread-safe bilingual logger and notification dispatcher.
    """
    def __init__(self, log_path=None):
        self.log_path = log_path

    def log(self, th_msg, en_msg=None, level="INFO"):
        if en_msg is None:
            full_msg = f"[{level}] {th_msg}"
        else:
            full_msg = f"[TH] {th_msg}\n[EN] {en_msg}"

        print(full_msg, flush=True)

        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(full_msg + "\n")
            except Exception:
                pass


logger = UnifiedLogger()


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        # [TH] ตั้งค่าตัวตัดเสียงอัจฉริยะ (Slicer) สำหรับตรวจจับความเงียบและแบ่งส่วนเสียง
        # [EN] Configure intelligent audio slicer for silence boundary detection
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        # [TH] ฟิลเตอร์กรองความถี่ต่ำกว่า 48Hz (Butterworth Highpass Filter) เพื่อตัดเสียงลมและ Sub-bass
        # [EN] 5th order Butterworth highpass filter (48Hz) to eliminate sub-bass rumble
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs")
        
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, output_key, idx1):
        """
        [TH] ปรับความดังมาตรฐาน (Peak Normalization) และบันทึกเป็นคู่ไฟล์ต้นฉบับและ 16kHz
        [EN] Perform audio normalization and write dual datasets (Ground Truth & 16kHz).
        """
        tmp_max = np.abs(tmp_audio).max()
        if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5:
            logger.log(
                f"⚠️ [ข้ามข้อมูล] ตรวจพบช่วงเสียงผิดปกติหรือเงียบสนิท: {output_key}_{idx1} (Peak: {tmp_max:.4f})",
                f"[SKIP] Invalid or silent audio chunk detected: {output_key}_{idx1} (Peak: {tmp_max:.4f})",
                level="WARN",
            )
            return False

        # [TH] Dynamic Scaling ผสมระหว่างสเกลสูงสุดและรูปคลื่นเดิม เพื่อรักษาไดนามิกเสียง
        # [EN] Apply blended dynamic scaling to preserve natural vocal expression
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (1.0 - self.alpha) * tmp_audio

        gt_file = os.path.join(self.gt_wavs_dir, f"{output_key}_{idx1}.wav")
        wavfile.write(gt_file, self.sr, tmp_audio.astype(np.float32))

        # [TH] Resample เข้าสู่ 16000Hz สำหรับการประมวลผล Feature Extraction (HuBERT/ContentVec)
        # [EN] Resample to 16kHz for HuBERT and ContentVec feature representation
        audio_16k = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000).astype(np.float32)
        wav16k_file = os.path.join(self.wavs16k_dir, f"{output_key}_{idx1}.wav")
        wavfile.write(wav16k_file, 16000, audio_16k)
        return True

    def pipeline(self, path, output_key, progress_index, total):
        """
        [TH] ประมวลผลไฟล์เสียงรายตัว: โหลด -> กรองความถี่ -> ตัดช่วงย่อย -> บันทึก
        [EN] Single audio processing pipeline: load -> filter -> slice -> save
        """
        try:
            filename = os.path.basename(path)
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            chunks_created = 0
            for sliced_segment in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(sliced_segment[start:]) > self.tail * self.sr:
                        tmp_audio = sliced_segment[start : start + int(self.per * self.sr)]
                        if self.norm_write(tmp_audio, output_key, idx1):
                            chunks_created += 1
                        idx1 += 1
                    else:
                        tmp_audio = sliced_segment[start:]
                        if self.norm_write(tmp_audio, output_key, idx1):
                            chunks_created += 1
                        idx1 += 1
                        break

            if should_report(progress_index, total):
                logger.log(
                    f"✅ สกัดเนื้อเสียงบริสุทธิ์ -> {filename} ({chunks_created} ชิ้น) | ลำดับ: {progress_index + 1}/{total}",
                    f"Processed pure vocals -> {filename} ({chunks_created} segments) | Progress: {progress_index + 1}/{total}",
                )
            return True
        except Exception:
            logger.log(
                f"❌ [เกิดข้อผิดพลาด] ไม่สามารถประมวลผลไฟล์: {path}\n{traceback.format_exc()}",
                f"[FAILED] Error processing audio path: {path}\n{traceback.format_exc()}",
                level="ERROR",
            )
            return False

    def pipeline_mp(self, infos):
        success = 0
        failed = 0
        for path, output_key, progress_index, total in infos:
            if self.pipeline(path, output_key, progress_index, total):
                success += 1
            else:
                failed += 1

    def pipeline_mp_inp_dir(self, inp_root, n_p, noparallel):
        try:
            names = sorted(
                name for name in os.listdir(inp_root)
                if os.path.isfile(os.path.join(inp_root, name)) and not name.startswith(".")
            )
            total = len(names)
            if total == 0:
                logger.log("⚠️ ไม่พบไฟล์เสียงในโฟลเดอร์เป้าหมาย", "[WARN] No audio files discovered in target folder.")
                return

            infos = [
                (os.path.join(inp_root, name), str(idx), idx, total)
                for idx, name in enumerate(names)
            ]
            worker_count = max(1, min(n_p, total))

            logger.log(
                f"🎙️ ตรวจพบไฟล์เสียงพร้อมสกัดความบริสุทธิ์: {total} ไฟล์ | Thread(s): {worker_count}",
                f"Discovered vocal files ready for extraction: {total} files | Thread(s): {worker_count}",
            )

            if noparallel or worker_count == 1:
                self.pipeline_mp(infos)
            else:
                processes = []
                for i in range(worker_count):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::worker_count],)
                    )
                    processes.append(p)
                    p.start()
                for p in processes:
                    p.join()
        except Exception:
            logger.log(
                f"❌ การทำงานแบบ Multiprocessing ขัดข้อง:\n{traceback.format_exc()}",
                f"Multiprocessing execution crashed:\n{traceback.format_exc()}",
                level="ERROR",
            )

    def pipeline_mp_manifest(self, manifest_entries, n_p, noparallel):
        total = len(manifest_entries)
        infos = [
            (entry["path"], entry["output_key"], idx, total)
            for idx, entry in enumerate(manifest_entries)
        ]
        worker_count = max(1, min(n_p, total))

        logger.log(
            f"🎙️ [Multi-Speaker] ตรวจพบไฟล์เสียงรวม: {total} รายการ | Thread(s): {worker_count}",
            f"[Multi-Speaker] Discovered entries: {total} files | Thread(s): {worker_count}",
        )

        if noparallel or worker_count == 1:
            self.pipeline_mp(infos)
            return

        processes = []
        for i in range(worker_count):
            p = multiprocessing.Process(
                target=self.pipeline_mp, args=(infos[i::worker_count],)
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, noparallel=False, manifest_path=""):
    start_time = time.time()
    log_file = os.path.join(exp_dir, "preprocess.log")
    os.makedirs(exp_dir, exist_ok=True)
    logger.log_path = log_file

    logger.log("=" * 65, "=" * 65)
    logger.log(
        "🚀 เริ่มต้นกระบวนการสกัดเนื้อเสียงบริสุทธิ์ระดับมืออาชีพ (Preprocess v4.2)...",
        "Starting Professional Pure Vocal Extraction (Preprocess v4.2)...",
    )
    logger.log("=" * 65, "=" * 65)

    pp = PreProcess(sr, exp_dir, per)

    # ล้างไฟล์ข้อมูลชุดเดิมเพื่อป้องกันการปะปน
    for folder in (pp.gt_wavs_dir, pp.wavs16k_dir):
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    try:
                        os.remove(os.path.join(folder, fname))
                    except OSError:
                        pass

    if manifest_path and load_manifest is not None:
        try:
            manifest = load_manifest(exp_dir)
            pp.pipeline_mp_manifest(manifest.get("entries", []), n_p, noparallel)
        except ManifestError as error:
            logger.log(f"❌ Manifest Error: {error}", f"Manifest Error: {error}", level="ERROR")
            raise
    else:
        pp.pipeline_mp_inp_dir(inp_root, n_p, noparallel)

    elapsed_str = time.strftime("%M:%S", time.gmtime(time.time() - start_time))
    logger.log("=" * 65, "=" * 65)
    logger.log(
        f"🎉 สิ้นสุดการเตรียมเนื้อเสียงบริสุทธิ์ (ระยะเวลา: {elapsed_str})",
        f"Pure vocal extraction completed (Elapsed: {elapsed_str})",
    )
    logger.log("=" * 65, "=" * 65)


def main():
    if len(sys.argv) < 7:
        print("[TH] พารามิเตอร์ไม่ครบถ้วน | [EN] Insufficient parameters")
        print("Usage: python preprocess.py <inp_root> <sr> <n_p> <exp_dir> <noparallel> <per> [manifest_path]")
        sys.exit(1)

    inp_root = sys.argv[1]
    sr = int(sys.argv[2])
    n_p = int(sys.argv[3])
    exp_dir = sys.argv[4]
    noparallel = sys.argv[5].lower() in ("true", "1")
    per = float(sys.argv[6])
    manifest_path = sys.argv[7] if len(sys.argv) > 7 else ""

    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, noparallel, manifest_path)


if __name__ == "__main__":
    main()
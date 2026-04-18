import os
import numpy as np
import librosa
import soundfile
from scipy import signal
from argparse import ArgumentParser

# ==============================================================================
# ฟังก์ชันคำนวณค่า RMS (Root Mean Square)
# ==============================================================================
def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    
    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    
    target_axis = axis - 1 if axis < 0 else axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)

# ==============================================================================
# ระบบลดเสียงรบกวนและปรับปรุงคุณภาพเสียง (Noise Reduction & Enhancement)
# ==============================================================================
def apply_noise_reduction(audio, sr):
    """
    ใช้ Butterworth High-pass filter เพื่อตัดเสียงรบกวนความถี่ต่ำ 
    และใช้ Noise Gate แบบนุ่มนวล
    """
    # 1. ตัดเสียง Rumble/Hum ความถี่ต่ำ (ต่ำกว่า 50Hz)
    sos = signal.butter(10, 50, 'hp', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    
    # 2. ทำ Normalization เพื่อให้ระดับเสียงเต็มอิ่มและสม่ำเสมอ
    max_peak = np.max(np.abs(audio))
    if max_peak > 0:
        audio = audio / max_peak * 0.95 # ปรับไปที่ -0.5 dB
        
    return audio

# ==============================================================================
# คลาส Slicer สำหรับตัดแบ่งไฟล์เสียงอย่างอัจฉริยะ
# ==============================================================================
class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError("เงื่อนไขที่ต้องปฏิบัติตาม: min_length >= min_interval >= hop_size")
        
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(sr * min_interval / 1000), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(sr * min_interval / 1000 / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
            
        if samples.shape[0] <= self.min_length:
            return [waveform]
            
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
                
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (i - silence_start >= self.min_interval and i - clip_start >= self.min_length)
            
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            
            # คำนวณจุดตัดที่เหมาะสมที่สุด (จุดที่เสียงเงียบที่สุด)
            pos = rms_list[silence_start : i + 1].argmin() + silence_start
            if silence_start == 0:
                sil_tags.append((0, pos))
            else:
                sil_tags.append((pos, pos))
            clip_start = pos
            silence_start = None

        # จัดการเสียงเงียบที่ท้ายไฟล์
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        if len(sil_tags) == 0:
            return [waveform]
        
        chunks = []
        if sil_tags[0][0] > 0:
            chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
        for i in range(len(sil_tags) - 1):
            chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
        if sil_tags[-1][1] < total_frames:
            chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            
        return chunks

# ==============================================================================
# ฟังก์ชันหลัก (Main Execution)
# ==============================================================================
def main():
    parser = ArgumentParser(description="เครื่องมือตัดแบ่งไฟล์เสียงพร้อมระบบลดเสียงรบกวน")
    parser.add_argument("audio", type=str, help="ไฟล์เสียงต้นฉบับ")
    parser.add_argument("--out", type=str, help="ไดเรกทอรีสำหรับบันทึกผลลัพธ์")
    parser.add_argument("--sr", type=int, default=44100, help="บังคับ Sample Rate (เช่น 40000, 44100, 48000)")
    parser.add_argument("--db_thresh", type=float, default=-40, help="ค่า Threshold สำหรับตรวจจับความเงียบ (dB)")
    
    args = parser.parse_args()
    
    # กำหนดไดเรกทอรีบันทึกไฟล์
    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.audio))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. โหลดไฟล์เสียงพร้อมบังคับ Sample Rate ให้ตรงกันทันที
    print(f"กำลังโหลดและปรับ Resampling เป็น {args.sr}Hz...")
    audio, sr = librosa.load(args.audio, sr=args.sr, mono=True)

    # 2. ลดเสียงรบกวนก่อนทำการตัดแบ่ง
    print("กำลังดำเนินการ Noise Reduction และ Normalization...")
    audio = apply_noise_reduction(audio, sr)

    # 3. เริ่มกระบวนการตัดแบ่ง (Slicing)
    slicer = Slicer(
        sr=sr,
        threshold=args.db_thresh,
        min_length=5000,
        min_interval=300,
        hop_size=10,
        max_sil_kept=500
    )
    
    chunks = slicer.slice(audio)
    
    # 4. บันทึกไฟล์ที่ถูกตัดแบ่ง
    base_name = os.path.basename(args.audio).rsplit(".", maxsplit=1)[0]
    for i, chunk in enumerate(chunks):
        save_path = os.path.join(out_dir, f"{base_name}_{i}.wav")
        soundfile.write(save_path, chunk, sr)
        
    print(f"เสร็จสมบูรณ์! บันทึกไฟล์ทั้งหมด {len(chunks)} ชิ้น ไปยัง: {out_dir}")

if __name__ == "__main__":
    main()

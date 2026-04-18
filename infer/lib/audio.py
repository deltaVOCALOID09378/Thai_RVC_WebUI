import platform
import os
import re
import traceback
import ffmpeg
import numpy as np
import av
from typing import Union

def wav2(i: str, o: str, output_format: str):
    """
    แปลงไฟล์เสียงโดยใช้ PyAV พร้อมการจัดการ Codec ที่ถูกต้อง
    """
    inp = av.open(i, "rb")
    
    # ปรับปรุงการเลือก Format และ Codec ให้เหมาะสม
    if output_format == "m4a":
        output_format = "mp4"
        codec_name = "aac"
    elif output_format == "ogg":
        codec_name = "libvorbis"
    elif output_format == "mp4":
        codec_name = "aac"
    else:
        codec_name = output_format

    out = av.open(o, "wb", format=output_format)
    ostream = out.add_stream(codec_name)

    # วนลูปเพื่อ Encode ข้อมูลเสียง
    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    # Flush encoder
    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()

def load_audio(file: str, sr: int) -> np.ndarray:
    """
    โหลดไฟล์เสียงและแปลงเป็น Floating Point 32-bit 
    พร้อมปรับปรุงคุณภาพเพื่อลดอาการเสียงแหบและเสียงแตก
    """
    try:
        file = clean_path(file)
        if not os.path.exists(file):
            raise RuntimeError(f"ไม่พบไฟล์เสียงในเส้นทางที่ระบุ: {file}")

        # ปรับปรุง FFmpeg Pipeline:
        # 1. 'loudnorm' หรือ 'peak normalization' เพื่อป้องกันเสียงแตก
        # 2. 'resampler' คุณภาพสูง (soxr)
        out, _ = (
            ffmpeg.input(file, threads=0)
            .filter("aresample", sr, resampler="soxr") # ใช้ soxr เพื่อความใสของเสียง
            .filter("volume", "0.95") # ลดระดับลงเล็กน้อยเพื่อป้องกัน Clipping (Headroom)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"เกิดข้อผิดพลาดในการโหลดไฟล์เสียง: {e}")

    # แปลงเป็น Numpy Array
    audio_data = np.frombuffer(out, np.float32).flatten()
    
    # ตรวจสอบและแก้ไขกรณีระดับเสียงเบาเกินไป (Automatic Gain Control)
    # หากค่าสูงสุดยังต่ำมาก ให้ขยายขึ้นมาในระดับที่เหมาะสม
    max_val = np.abs(audio_data).max()
    if max_val > 0:
        audio_data = audio_data / max_val * 0.9  # Normalize ไปที่ -1dB โดยประมาณ

    return audio_data

def clean_path(path_str: str) -> str:
    """
    ทำความสะอาดเส้นทางไฟล์ ป้องกันข้อผิดพลาดจากการคัดลอกเส้นทางที่มีอักขระพิเศษ
    """
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    
    # กำจัด Unicode Control Characters ที่มักติดมากับการคัดลอกใน Windows
    path_str = re.sub(r'[\u202a\u202b\u202c\u202d\u202e]', '', path_str)
    
    # ลบช่องว่างและเครื่องหมายคำพูดที่เกินมา
    return path_str.strip().strip('"').strip("'").strip()

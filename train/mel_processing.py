# ==============================================================================
# File Name    : mel_processing.py
# Version      : v2.2.0-Production
# Target Path  : infer/modules/train/mel_processing.py (หรือ infer/lib/audio/mel_processing.py)
# Function     : Professional Spectral Analysis, Windowing & Mel-Spectrogram Extraction Engine
# Credits      : Made And Checked By DELTA SYNTH & All Agentic AI
#                Original by RVC-Project / VITS Team
# ==============================================================================

import logging
import torch
import torch.utils.data
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

logger = logging.getLogger(__name__)

# ระดับสัญญาณเสียงสูงสุดมาตรฐานสำหรับฟอร์แมต 16-bit PCM
# Maximum standard audio value for 16-bit PCM waveform representation
MAX_WAV_VALUE = 32768.0

# แคชคลังข้อมูลตัวกรอง Mel และหน้าต่าง Hann เพื่อลดการจัดสรรหน่วยความจำซ้ำซ้อน
# Reusable global cache banks for Hann windows and Mel filterbanks
mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1.0, clip_val=1e-5):
    """
    [TH] บีบอัดช่วงไดนามิกของสัญญาณ Amplitude ด้วยฟังก์ชัน Logarithmic
    [EN] Compress the dynamic range of amplitude spectrograms using logarithmic scaling.

    Parameters:
        x (torch.Tensor): อินพุตสเปกโตรแกรม / Input spectrogram tensor
        C (float): ตัวคูณสเกลการบีบอัด / Compression scaling factor
        clip_val (float): ค่าต่ำสุดเพื่อป้องกันภาวะลบหรือศูนย์ / Floor threshold to prevent log(0)
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1.0):
    """
    [TH] คลายการบีบอัดช่วงไดนามิกกลับคืนสู่สเกลเชิงเส้นดั้งเดิม
    [EN] Decompress logarithmic spectrograms back to linear amplitude scale.

    Parameters:
        x (torch.Tensor): สเปกโตรแกรมที่ผ่านการบีบอัด / Compressed spectrogram
        C (float): ตัวคูณสเกลคืนสภาพ / Decompression scaling factor
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    """
    [TH] ปรับขนาดสเปกตรัมให้อยู่ในมาตรฐาน Log Scale สำหรับโมเดลสังเคราะห์เสียง
    [EN] Apply spectral normalization to magnitude spectrograms.
    """
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    """
    [TH] แปลงค่าสเปกตรัมจาก Log Scale กลับสู่ขนาดเชิงเส้น
    [EN] Reverse spectral normalization back to linear magnitudes.
    """
    return dynamic_range_decompression_torch(magnitudes)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    [TH] แปลงสัญญาณคลื่นเสียง (Waveform) สู่ขนาดสเปกโตรแกรมความถี่เชิงเส้น (Linear Spectrogram)
    [EN] Convert time-domain audio waveforms into linear-frequency magnitude spectrograms.

    Args:
        y (torch.Tensor): รูปคลื่นสัญญาณเสียง มิติ (B, T) หรือ (T)
        n_fft (int): ขนาดการแปลงฟูเรียร์ (FFT size)
        sampling_rate (int): อัตราสุ่มสัญญาณเสียง (Sampling rate)
        hop_size (int): ระยะการก้าวของหน้าต่างวิเคราะห์ (Hop length)
        win_size (int): ความกว้างของหน้าต่างวิเคราะห์ (Window length)
        center (bool): กำหนดการจัดวางจุดกึ่งกลางของหน้าต่าง

    Returns:
        torch.Tensor: ขนาดสเปกโตรแกรม มิติ (B, Freq, Frame)
    """
    # ตรวจสอบและแปลงมิติข้อมูลให้เป็น 2D Tensor (B, T) เสมอ
    # Standardize input dimension to ensure shape is (B, T)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # แคชหน้าต่าง Hann Window โดยระบุขนาด, ชนิดข้อมูล และอุปกรณ์คำนวณอย่างรัดกุม
    # Unique cache key for Hann window accounting for size, dtype, and device
    window_key = f"{win_size}_{y.device}"
    if window_key not in hann_window:
        hann_window[window_key] = torch.hann_window(win_size).to(
            dtype=torch.float32, device=y.device
        )

    # คำนวณระยะ Padding เพื่อให้สอดคล้องกับขนาด STFT
    # Compute symmetric reflection padding length
    pad_amount = int((n_fft - hop_size) / 2)
    
    # เสริมขอบเขตความปลอดภัยกรณีสัญญาณสั้นกว่าระยะ Padding
    # Apply reflection padding along the temporal dimension safely
    if y.size(-1) < pad_amount * 2:
        y = F.pad(y, (pad_amount, pad_amount), mode="constant", value=0.0)
    else:
        y = F.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)

    # รักษาระดับความแม่นยำตัวเลขด้วย Float32 ระหว่างการคำนวณ STFT เพื่อป้องกัน NaN
    # Enforce float32 precision during STFT to guarantee numerical stability
    orig_dtype = y.dtype
    y_float = y.to(dtype=
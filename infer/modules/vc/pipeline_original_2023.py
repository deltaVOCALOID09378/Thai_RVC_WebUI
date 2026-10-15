# -*- coding: utf-8 -*-
# ==============================================================================
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# Original by RVC-Project
# File: pipeline.py | Version: 2.3 (High-Performance Enhanced Edition)
# Date: 2026-10-09
# Description: Conversion Pipeline with De-Esser, EBU R128 Normalization & Tensor Alignment
# ==============================================================================

import os
import sys
import traceback
import logging
from functools import lru_cache
from time import time as ttime

import faiss
import librosa
import numpy as np
import parselmouth
import pyworld
import torch
import torch.nn.functional as F
import torchcrepe
from scipy import signal

from utils import PoliteLogger

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(audio, fs=fs, f0_ceil=f0max, f0_floor=f0min, frame_period=frame_period)
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def clean_and_deess(audio, sr=16000):
    """
    โมดูลกรองเสียงล่วงหน้า (Pre-processing Filter):
    1. กำจัด DC Offset และ Sub-bass rumble ความถี่ต่ำกว่า 50Hz
    2. Dynamic De-esser กดเสียงเสียดแทรกพยัญชนะไทยที่แหลมกระแทก (4.5kHz - 7.5kHz)
    """
    if len(audio) == 0:
        return audio
    # High-pass filter กำจัด DC offset และ Sub-bass rumble
    sos_hp = signal.butter(4, 50, btype="highpass", fs=sr, output="sos")
    audio = signal.sosfiltfilt(sos_hp, audio)
    
    # Dynamic De-esser ตรวจจับย่านความถี่เสียดแทรก (Sibilance)
    high_cut = min(7500, sr // 2 - 100)
    if high_cut > 4500:
        sos_sibilance = signal.butter(4, [4500, high_cut], btype="bandpass", fs=sr, output="sos")
        sibilant_band = signal.sosfiltfilt(sos_sibilance, audio)
        env = np.abs(sibilant_band)
        threshold = np.percentile(env, 95) * 0.8
        if threshold > 1e-4:
            excess = np.maximum(0, env - threshold)
            attenuation = excess / (excess + threshold + 1e-6) * 0.45
            audio = audio - sibilant_band * attenuation
    return audio


def normalize_loudness_ebu_r128(audio, sr, target_lufs=-16.0, peak_limit=0.98):
    """
    โมดูลเกลี่ยความดังอัตโนมัติมาตรฐาน EBU R128 / ITU-R BS.1770
    - คำนวณ K-Weighting Loudness (LUFS)
    - ปรับเกลี่ยความดังเฉลี่ยทุกไฟล์ในคลังเสียงให้ได้ระดับเดียวกัน (-16 LUFS)
    - ป้องกันเสียงแตกพร่า (True Peak Limiting) ไม่เกิน 0.98 (-0.17 dBFS)
    """
    if len(audio) == 0:
        return audio
        
    f0 = 1681.9744509555319
    G = 3.99984385397
    Q = 0.7071752369554193
    K = np.tan(np.pi * f0 / sr)
    Vh = 10.0 ** (G / 20.0)
    Vb = Vh ** 0.499666774155
    a0 = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0
    b1 = 2.0 * (K * K - Vh) / a0
    b2 = (Vh - Vb * K / Q + K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    
    f0_hp = 38.13547087613982
    Q_hp = 0.5003270373253953
    K_hp = np.tan(np.pi * f0_hp / sr)
    a0_hp = 1.0 + K_hp / Q_hp + K_hp * K_hp
    b0_hp = 1.0 / a0_hp
    b1_hp = -2.0 / a0_hp
    b2_hp = 1.0 / a0_hp
    a1_hp = 2.0 * (K_hp * K_hp - 1.0) / a0_hp
    a2_hp = (1.0 - K_hp / Q_hp + K_hp * K_hp) / a0_hp
    
    y = signal.lfilter([b0, b1, b2], [1.0, a1, a2], audio)
    y = signal.lfilter([b0_hp, b1_hp, b2_hp], [1.0, a1_hp, a2_hp], y)
    
    ms = np.mean(y ** 2)
    if ms < 1e-12:
        max_p = np.max(np.abs(audio))
        if max_p > 0:
            return audio * (peak_limit / max_p)
        return audio
        
    current_lufs = -0.691 + 10.0 * np.log10(ms)
    gain_db = target_lufs - current_lufs
    gain_linear = 10.0 ** (gain_db / 20.0)
    
    norm_audio = audio * gain_linear
    max_peak = np.max(np.abs(norm_audio))
    if max_peak > peak_limit:
        norm_audio = norm_audio * (peak_limit / max_peak)
        
    return norm_audio


def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad, config.x_query, config.x_center, config.x_max, config.is_half,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device

    def get_f0(self, input_audio_path, x, p_len, f0_up_key, f0_method, filter_radius, inp_f0=None):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        PoliteLogger.info(f"สกัดค่าระดับเสียง (Extracting F0 curve via): {f0_method.upper()}")

        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(time_step=time_step / 1000, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max)
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio, self.sr, self.window, f0_min, f0_max, "full", batch_size=512, device=self.device, return_periodicity=True
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from infer.lib.rmvpe import RMVPE
                rmvpe_path = f"{os.environ.get('rmvpe_root', 'assets/rmvpe')}/rmvpe.pt"
                PoliteLogger.system(f"กำลังโหลดโมเดล RMVPE (Loading RMVPE model from): {rmvpe_path}")
                self.model_rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device)
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        f0 *= pow(2, f0_up_key / 12)
        tf0 = self.sr // self.window
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0)
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
            
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak

    def pipeline(
        self, model, net_g, sid, audio, input_audio_path, times, f0_up_key,
        f0_method, file_index, index_rate, if_f0, filter_radius, tgt_sr,
        resample_sr, rms_mix_rate, version, protect, f0_file=None
    ):
        start_timer = ttime()
        filename = os.path.basename(input_audio_path) if input_audio_path else 'Audio_Stream'
        
        PoliteLogger.info(f"เริ่มกระบวนการสกัดฟีเจอร์และแปลงเสียง (Starting pipeline processing): {filename}")

        if file_index and os.path.exists(file_index) and index_rate > 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
                PoliteLogger.info(f"ประยุกต์ใช้ Faiss Index (Applying Faiss Index Ratio: {index_rate})")
            except Exception as e:
                PoliteLogger.warning(f"ไม่สามารถโหลดไฟล์ Index ได้ (Failed to load Faiss Index): {e}")
                index = big_npy = None
        else:
            index = big_npy = None

        # 1. ระบบกรองเสียงล่วงหน้า (Pre-filter & Dynamic De-esser)
        audio = clean_and_deess(audio, sr=16000)
        
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t - self.t_query + np.where(
                        audio_sum[t - self.t_query : t + self.t_query] == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = np.array([[float(i) for i in line.split(",")] for line in lines], dtype="float32")
            except Exception:
                pass

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(input_audio_path, audio_pad, p_len, f0_up_key, f0_method, filter_radius, inp_f0)
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) and "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
            
        t2 = ttime()
        
        # Safe time aggregation for F0 Pitch
        if isinstance(times, list) and len(times) >= 3:
            times[1] += (t2 - t1)
        else:
            times += (t2 - t1)

        # Iterative VC Slices
        for t in opt_ts:
            t = t // self.window * self.window
            hasp = if_f0 == 1
            p_slice = pitch[:, s // self.window : (t + self.t_pad2) // self.window] if hasp else None
            pf_slice = pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if hasp else None
            
            audio_opt.append(
                self.vc(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window],
                        p_slice, pf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t

        hasp = if_f0 == 1
        p_slice = pitch[:, t // self.window :] if (hasp and t is not None) else pitch
        pf_slice = pitchf[:, t // self.window :] if (hasp and t is not None) else pitchf
        
        audio_opt.append(
            self.vc(model, net_g, sid, audio_pad[t:], p_slice, pf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt]
        )
        
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
            
        output_sr = tgt_sr
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)
            output_sr = resample_sr

        # 2. ระบบเกลี่ยความดังอัตโนมัติ EBU R128 (-16 LUFS) & True Peak Limiter
        audio_opt = normalize_loudness_ebu_r128(audio_opt, output_sr, target_lufs=-16.0, peak_limit=0.98)
        
        audio_opt = (audio_opt * 32768.0).astype(np.int16)
        
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        total_time_taken = ttime() - start_timer
        PoliteLogger.success(f"ประมวลผลไฟล์สำเร็จ (Render finished): {filename} (Time: {total_time_taken:.2f}s)")
        return audio_opt

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

        # 1. Clone raw acoustic features (feats0) before retrieval for consonant protection
        feats0 = feats.clone()

        # 2. FAISS Index Retrieval
        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        # 3. Upsample features from 50Hz (HuBERT) to 100Hz (VITS / F0)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        t1 = ttime()
        
        # Safe time aggregation for Hubert Extractor
        if isinstance(times, list) and len(times) >= 3:
            times[0] += (t1 - t0)
            
        p_len = audio0.shape[0] // self.window

        # 4. Strict length alignment between feats, pitch, and pitchf
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
        if pitch is not None and pitchf is not None:
            if pitch.shape[1] < p_len:
                p_len = pitch.shape[1]
            if pitchf.shape[1] < p_len:
                p_len = pitchf.shape[1]
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]
        feats = feats[:, :p_len, :]

        # 5. Apply Consonant & Unvoiced Protection (blending converted feats with clean feats0)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats0[:, :p_len, :]
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1.0
            pitchff[pitchf <= 0] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1.0 - pitchff)
            feats = feats.to(feats0.dtype)
            
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        
        # Safe time aggregation for Synthesizer Infer
        if isinstance(times, list) and len(times) >= 3:
            times[2] += (t2 - t1)
            
        return audio1

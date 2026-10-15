# ==============================================================================
# Made And Checked By DELTA SYNTH & All Agentic AI
# Original by RVC-Project
# File: infer/modules/uvr5/preprocess.py | Version: 1.3
# Description: [EN] UVR5 Audio Preprocessing / [TH] การเตรียมข้อมูลเสียงสำหรับ UVR5
# ==============================================================================

import os
import sys
import subprocess
import logging
import gc
import numpy as np
import torch
import librosa
import soundfile as sf

from infer.modules.uvr5.vr_network import nets
from infer.modules.uvr5.vr_network import spec_utils

logger = logging.getLogger(__name__)

class AudioPre:
    def __init__(self, agg, model_path, device, is_half):
        self.model_path = model_path
        self.device = device
        self.is_half = is_half
        self.model_name = os.path.basename(model_path)
        self.agg = agg
        self.window_size = 512
        self.batch_size = 4
        self.crop_size = 256
        
        self.data = {
            1: {"mid_side": False, "mid_side_b2": False, "mid_side_b2_ogg": False, "coef": 100},
            2: {"mid_side": False, "mid_side_b2": False, "mid_side_b2_ogg": False, "coef": 100},
        }
        
        self.mp = nets.ModelParameters("infer/modules/uvr5/vr_network/modelparams/4band_v2.json")
        self.model = nets.CascadedASPPNet(self.mp.param["bins"] * 2)
        cpk = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(cpk)
        self.model.eval()
        if self.is_half:
            self.model = self.model.half()
        self.model = self.model.to(self.device)

    def _path_audio_(self, music_file, ins_root=None, vocal_root=None, format0="flac", is_hp3=False):
        if ins_root is None and vocal_root is None:
            return "No save root specified."
            
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)

        inp_path = music_file
        
        # [ZERO-DEFECT FIX]: Safe FFmpeg execution
        temp_dir = os.environ.get("TEMP", os.path.join(os.getcwd(), "TEMP"))
        os.makedirs(temp_dir, exist_ok=True)
        tmp_path = os.path.join(temp_dir, f"{name}.reformatted.wav")
        
        try:
            cmd = ['ffmpeg', '-y', '-i', inp_path, '-vn', '-ar', '44100', '-ac', '2', '-sample_fmt', 's16', tmp_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"🔴 ล้มเหลวในการแปลงไฟล์ด้วย FFmpeg (FFmpeg Error) on {name}: {str(e)}")
            raise RuntimeError(f"FFmpeg failed to format audio: {name}")

        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])
        
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:
                X_wave[d], _ = librosa.core.load(tmp_path, sr=bp["sr"], res_type=bp["res_type"])
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.core.resample(
                    X_wave[d + 1],
                    orig_sr=self.mp.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )
                
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            
            if d == bands_n and self.data[d]["coef"] > 1:
                X_spec_s[d] = spec_utils.merge_artifacts(X_spec_s[d], self.data[d])

        X_mag = np.abs(X_spec_s[1])
        X_phase = np.angle(X_spec_s[1])
        
        v_mag = np.zeros_like(X_mag)
        length = X_mag.shape[2]
        overlap = self.crop_size // 2
        padded_mag = np.pad(X_mag, ((0, 0), (0, 0), (0, self.crop_size)), mode='constant')
        
        for i in range(0, length, overlap):
            crop = padded_mag[:, :, i:i+self.crop_size]
            if crop.shape[2] < self.crop_size:
                break
                
            x_t = torch.from_numpy(crop).unsqueeze(0).to(self.device)
            if self.is_half:
                x_t = x_t.half()
                
            with torch.no_grad():
                pred = self.model(x_t)
                
            v_mag[:, :, i:i+self.crop_size] += pred.squeeze(0).cpu().numpy()

        v_mag /= 2.0
        v_mag = v_mag[:, :, :length]
        
        v_mag = v_mag * (self.agg / 10.0)
        i_mag = X_mag - v_mag

        y_spec_m = v_mag * np.exp(1.j * X_phase)
        v_wave = spec_utils.spectrogram_to_wave(
            y_spec_m, hl=self.mp.param["band"][1]["hl"], n_fft=self.mp.param["band"][1]["n_fft"]
        )
        
        i_spec_m = i_mag * np.exp(1.j * X_phase)
        i_wave = spec_utils.spectrogram_to_wave(
            i_spec_m, hl=self.mp.param["band"][1]["hl"], n_fft=self.mp.param["band"][1]["n_fft"]
        )

        vocal_name = f"vocal_{name}.{format0}"
        ins_name = f"instrument_{name}.{format0}"
        
        if is_hp3:
            sf.write(os.path.join(vocal_root, vocal_name), i_wave.T, self.mp.param["band"][1]["sr"])
            sf.write(os.path.join(ins_root, ins_name), v_wave.T, self.mp.param["band"][1]["sr"])
        else:
            sf.write(os.path.join(vocal_root, vocal_name), v_wave.T, self.mp.param["band"][1]["sr"])
            sf.write(os.path.join(ins_root, ins_name), i_wave.T, self.mp.param["band"][1]["sr"])

        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
                
        # [ZERO-DEFECT FIX]: Clear Tensor memory cache to prevent memory leaks in loop
        del X_wave, y_wave, X_spec_s, y_spec_s, X_mag, X_phase, v_mag, i_mag, y_spec_m, v_wave, i_spec_m, i_wave
        gc.collect()

class AudioPreDeEcho(AudioPre):
    def __init__(self, agg, model_path, device, is_half):
        super().__init__(agg, model_path, device, is_half)

    def _path_audio_(self, music_file, ins_root=None, vocal_root=None, format0="flac", is_hp3=False):
        if ins_root is None and vocal_root is None:
            return "No save root specified."
            
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)

        inp_path = music_file
        temp_dir = os.environ.get("TEMP", os.path.join(os.getcwd(), "TEMP"))
        os.makedirs(temp_dir, exist_ok=True)
        tmp_path = os.path.join(temp_dir, f"{name}.reformatted.wav")
        
        try:
            cmd = ['ffmpeg', '-y', '-i', inp_path, '-vn', '-ar', '44100', '-ac', '2', '-sample_fmt', 's16', tmp_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"🔴 ล้มเหลวในการแปลงไฟล์ด้วย FFmpeg (FFmpeg Error) on {name}: {str(e)}")
            raise RuntimeError(f"FFmpeg failed to format audio: {name}")

        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])
        
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:
                X_wave[d], _ = librosa.core.load(tmp_path, sr=bp["sr"], res_type=bp["res_type"])
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:
                X_wave[d] = librosa.core.resample(
                    X_wave[d + 1],
                    orig_sr=self.mp.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )
                
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            
            if d == bands_n and self.data[d]["coef"] > 1:
                X_spec_s[d] = spec_utils.merge_artifacts(X_spec_s[d], self.data[d])

        X_mag = np.abs(X_spec_s[1])
        X_phase = np.angle(X_spec_s[1])
        
        v_mag = np.zeros_like(X_mag)
        length = X_mag.shape[2]
        overlap = self.crop_size // 2
        padded_mag = np.pad(X_mag, ((0, 0), (0, 0), (0, self.crop_size)), mode='constant')
        
        for i in range(0, length, overlap):
            crop = padded_mag[:, :, i:i+self.crop_size]
            if crop.shape[2] < self.crop_size:
                break
                
            x_t = torch.from_numpy(crop).unsqueeze(0).to(self.device)
            if self.is_half:
                x_t = x_t.half()
                
            with torch.no_grad():
                pred = self.model(x_t)
                
            v_mag[:, :, i:i+self.crop_size] += pred.squeeze(0).cpu().numpy()

        v_mag /= 2.0
        v_mag = v_mag[:, :, :length]
        
        v_mag = v_mag * (self.agg / 10.0)
        i_mag = X_mag - v_mag

        y_spec_m = v_mag * np.exp(1.j * X_phase)
        v_wave = spec_utils.spectrogram_to_wave(
            y_spec_m, hl=self.mp.param["band"][1]["hl"], n_fft=self.mp.param["band"][1]["n_fft"]
        )
        
        i_spec_m = i_mag * np.exp(1.j * X_phase)
        i_wave = spec_utils.spectrogram_to_wave(
            i_spec_m, hl=self.mp.param["band"][1]["hl"], n_fft=self.mp.param["band"][1]["n_fft"]
        )

        vocal_name = f"vocal_{name}.{format0}"
        ins_name = f"instrument_{name}.{format0}"
        
        if is_hp3:
            sf.write(os.path.join(ins_root, ins_name), i_wave.T, self.mp.param["band"][1]["sr"])
            sf.write(os.path.join(vocal_root, vocal_name), v_wave.T, self.mp.param["band"][1]["sr"])
        else:
            sf.write(os.path.join(ins_root, ins_name), v_wave.T, self.mp.param["band"][1]["sr"])
            sf.write(os.path.join(vocal_root, vocal_name), i_wave.T, self.mp.param["band"][1]["sr"])

        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        
        # [ZERO-DEFECT FIX]: Clear Tensor memory cache to prevent memory leaks in loop
        del X_wave, y_wave, X_spec_s, y_spec_s, X_mag, X_phase, v_mag, i_mag, y_spec_m, v_wave, i_spec_m, i_wave
        gc.collect()
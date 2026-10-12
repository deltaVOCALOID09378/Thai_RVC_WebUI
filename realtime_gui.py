# ==============================================================================
# File Name: realtime_gui.py
# Version: 2.5
# Location: Root Directory
# Function: RVC Real-Time Voice Conversion GUI / หน้าต่างแปลงเสียงเรียลไทม์
# Credit: Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
#         Original by Patiphat Wongyai (Delta) & RVC Project
# Description: Custom DELTA SYNTH Styling, VRAM Safe Release, and I/O Optimization
# ==============================================================================
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

now_dir = os.path.dirname(os.path.abspath(__file__))

from tools.file_io import read_text

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = "4"

realtime_config_path = os.path.join(now_dir, "configs", "config.json")
flag_vc = False

def printt(strr, *args):
    """Bilingual Real-time Log Flusher"""
    if len(args) == 0:
        print(strr, flush=True)
    else:
        print(strr % args, flush=True)

def clear_vram_safe():
    """Force garbage collection and VRAM release safely."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        printt("✅ [SYSTEM] คืนค่า VRAM เรียบร้อยแล้ว (VRAM Cleared Successfully)")
    except Exception as e:
        printt(f"⚠️ [SYSTEM] ไม่สามารถคืนค่า VRAM ได้ (VRAM Clear Failed): {e}")


if __name__ == "__main__":
    import json
    import re
    import time
    import traceback

    import librosa
    from tools.torchgate import TorchGate
    import numpy as np
    import FreeSimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    from configs.config import Config
    from infer import rtrvc as rvc_for_realtime
    from i18n.i18n import I18nAuto
    from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph

    i18n = I18nAuto()

    # ==========================================
    # DELTA SYNTH CORPORATE THEME STYLING
    # ==========================================
    sg.LOOK_AND_FEEL_TABLE['DeltaSynthTheme'] = {
        'BACKGROUND': '#1A1A1A',
        'TEXT': '#F0F0F0',
        'INPUT': '#242424',
        'TEXT_INPUT': '#F0F0F0',
        'SCROLL': '#CC2200',
        'BUTTON': ('#FFFFFF', '#CC2200'),
        'PROGRESS': ('#CC2200', '#1A1A1A'),
        'BORDER': 1,
        'SLIDER_DEPTH': 0,
        'PROGRESS_DEPTH': 0
    }
    sg.theme('DeltaSynthTheme')
    sg.set_options(font=('Leelawadee UI', 10))

    class GUIConfig:
        def __init__(self) :
            self.pth_path = ""
            self.index_path = ""
            self.pitch = 0
            self.formant=0.0
            self.sr_type = "sr_model"
            self.block_time = 0.25
            self.threhold = -60
            self.crossfade_time = 0.05
            self.extra_time = 2.5
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.rms_mix_rate = 0.0
            self.index_rate = 0.0
            self.f0method = "rmvpe"
            self.sg_hostapi = ""
            self.wasapi_exclusive = False
            self.sg_input_device = ""
            self.sg_output_device = ""

    class GUI:
        def __init__(self) :
            self.gui_config = GUIConfig()
            self.config = Config()
            printt("RVC_CUDA_GRAPH=%s", os.environ.get("RVC_CUDA_GRAPH", "0"))
            self.function = "vc"
            self.delay_time = 0
            self.hostapis = None
            self.input_devices = None
            self.output_devices = None
            self.input_devices_indices = None
            self.output_devices_indices = None
            self.stream = None
            self.update_devices()
            self.launcher()

        def load(self):
            try:
                data = json.loads(read_text(realtime_config_path))
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
                if data.get("f0method") not in ("pm", "rmvpe", "fcpe"):
                    data["f0method"] = "rmvpe"
                data["pm"] = data["f0method"] == "pm"
                data["rmvpe"] = data["f0method"] == "rmvpe"
                data["fcpe"] = data["f0method"] == "fcpe"
                if data["sg_hostapi"] in self.hostapis:
                    self.update_devices(hostapi_name=data["sg_hostapi"])
                    if (
                        data["sg_input_device"] not in self.input_devices
                        or data["sg_output_device"] not in self.output_devices
                    ):
                        self.update_devices()
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
                else:
                    data["sg_hostapi"] = self.hostapis[0]
                    data["sg_input_device"] = self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ]
                    data["sg_output_device"] = self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ]
            except:
                with open(realtime_config_path, "w", encoding="utf8") as j:
                    data = {
                        "pth_path": "",
                        "index_path": "",
                        "sg_hostapi": self.hostapis[0],
                        "sg_wasapi_exclusive": False,
                        "sg_input_device": self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ],
                        "sg_output_device": self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ],
                        "sr_type": "sr_model",
                        "threhold": -60,
                        "pitch": 0,
                        "formant": 0.0,
                        "index_rate": 0,
                        "rms_mix_rate": 0,
                        "block_time": 0.25,
                        "crossfade_length": 0.05,
                        "extra_time": 2.5,
                        "f0method": "rmvpe",
                    }
            return data

        def launcher(self):
            data = self.load()
            layout = [
                [
                    sg.Frame(
                        title="🎙️ โหลดโมเดลเสียง (Voice Model Loader)",
                        layout=[
                            [
                                sg.Input(default_text=data.get("pth_path", ""), key="pth_path", size=(50, 1)),
                                sg.FileBrowse("เลือกไฟล์ .pth (Browse)", initial_folder=os.path.join(os.getcwd(), "assets/weights"), file_types=(("PTH Files", "*.pth"),)),
                            ],
                            [
                                sg.Input(default_text=data.get("index_path", ""), key="index_path", size=(50, 1)),
                                sg.FileBrowse("เลือกไฟล์ .index (Browse)", initial_folder=os.path.join(os.getcwd(), "logs"), file_types=(("Index Files", "*.index"),)),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        title="🎧 อุปกรณ์เสียงและฮาร์ดแวร์ (Audio Interfaces)",
                        layout=[
                            [
                                sg.Text("ประเภทอุปกรณ์ (Host API)"),
                                sg.Combo(self.hostapis, key="sg_hostapi", default_value=data.get("sg_hostapi", ""), enable_events=True, size=(20, 1)),
                                sg.Checkbox("เปิดโหมด WASAPI พิเศษ (Exclusive Mode)", key="sg_wasapi_exclusive", default=data.get("sg_wasapi_exclusive", False), enable_events=True),
                            ],
                            [
                                sg.Text("อุปกรณ์นำเข้า (Input)    "),
                                sg.Combo(self.input_devices, key="sg_input_device", default_value=data.get("sg_input_device", ""), enable_events=True, size=(45, 1)),
                            ],
                            [
                                sg.Text("อุปกรณ์ส่งออก (Output) "),
                                sg.Combo(self.output_devices, key="sg_output_device", default_value=data.get("sg_output_device", ""), enable_events=True, size=(45, 1)),
                            ],
                            [
                                sg.Button("รีเฟรชอุปกรณ์ (Refresh)", key="reload_devices"),
                                sg.Radio("อิงตามโมเดล (Model SR)", "sr_type", key="sr_model", default=data.get("sr_model", True), enable_events=True),
                                sg.Radio("อิงตามฮาร์ดแวร์ (Device SR)", "sr_type", key="sr_device", default=data.get("sr_device", False), enable_events=True),
                                sg.Text("อัตราสุ่มตัวอย่าง (SR):"), sg.Text("", key="sr_stream", text_color="#FF4422"),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        title="⚙️ ตั้งค่าระดับเสียงและโทน (Voice Parameters)",
                        layout=[
                            [
                                sg.Text("เกณฑ์ตอบสนอง (Threshold dB)"),
                                sg.Slider(range=(-60, 0), key="threhold", resolution=1, orientation="h", default_value=data.get("threhold", -60), enable_events=True),
                            ],
                            [
                                sg.Text("เปลี่ยนคีย์เสียง (Pitch Shifting)"),
                                sg.Slider(range=(-16, 16), key="pitch", resolution=1, orientation="h", default_value=data.get("pitch", 0), enable_events=True),
                            ],
                            [
                                sg.Text("ปรับความหนาบาง (Formant Shift)"),
                                sg.Slider(range=(-2, 2), key="formant", resolution=0.05, orientation="h", default_value=data.get("formant", 0.0), enable_events=True),
                            ],
                            [
                                sg.Text("อิทธิพลของดัชนี (Index Ratio)  "),
                                sg.Slider(range=(0.0, 1.0), key="index_rate", resolution=0.01, orientation="h", default_value=data.get("index_rate", 0), enable_events=True),
                            ],
                            [
                                sg.Text("การผสานความดัง (RMS Mix)      "),
                                sg.Slider(range=(0.0, 1.0), key="rms_mix_rate", resolution=0.01, orientation="h", default_value=data.get("rms_mix_rate", 0), enable_events=True),
                            ],
                            [
                                sg.Text("อัลกอริทึมวิเคราะห์ (Pitch Algo)"),
                                sg.Radio("pm", "f0method", key="pm", default=data.get("pm", False), enable_events=True),
                                sg.Radio("rmvpe", "f0method", key="rmvpe", default=data.get("rmvpe", True), enable_events=True),
                                sg.Radio("fcpe", "f0method", key="fcpe", default=data.get("fcpe", False), enable_events=True),
                            ],
                        ],
                    ),
                    sg.Frame(
                        title="⚡ ตั้งค่าประสิทธิภาพและกันกระตุก (Performance)",
                        layout=[
                            [
                                sg.Text("ขนาดบัฟเฟอร์ (Block Time)     "),
                                sg.Slider(range=(0.02, 1.5), key="block_time", resolution=0.01, orientation="h", default_value=data.get("block_time", 0.25), enable_events=True),
                            ],
                            [
                                sg.Text("ระยะเชื่อมต่อ (Crossfade Length)"),
                                sg.Slider(range=(0.01, 0.15), key="crossfade_length", resolution=0.01, orientation="h", default_value=data.get("crossfade_length", 0.05), enable_events=True),
                            ],
                            [
                                sg.Text("ระยะวิเคราะห์ล่วงหน้า (Extra Time)"),
                                sg.Slider(range=(0.05, 5.00), key="extra_time", resolution=0.01, orientation="h", default_value=data.get("extra_time", 2.5), enable_events=True),
                            ],
                            [
                                sg.Checkbox("ตัดเสียงรบกวนอินพุต (In-Denoise)", key="I_noise_reduce", enable_events=True),
                                sg.Checkbox("ตัดเสียงรบกวนเอาต์พุต (Out-Denoise)", key="O_noise_reduce", enable_events=True),
                            ],
                        ],
                    ),
                ],
                [
                    sg.Button("▶ เริ่มสตรีมและแปลงเสียง (Start Voice Conversion)", key="start_vc", button_color=('#FFFFFF', '#CC2200')),
                    sg.Button("⏹ หยุดการทำงาน (Stop & Clear VRAM)", key="stop_vc", button_color=('#FFFFFF', '#1A1A1A')),
                    sg.Radio("ฟังเสียงจริง (Monitor Input)", "function", key="im", default=False, enable_events=True),
                    sg.Radio("ส่งออกเสียงแปลง (Output Voice)", "function", key="vc", default=True, enable_events=True),
                    sg.Text("ความหน่วงอัลกอริทึม (Algo Latency):"), sg.Text("0", key="delay_time", text_color="#FF4422"),
                    sg.Text("เวลาประมวลผล (Infer Time):"), sg.Text("0", key="infer_time", text_color="#FF4422"),
                ],
            ]
            self.window = sg.Window("DELTA SYNTH - Real-time RVC Studio", layout=layout, finalize=True)
            self.event_handler()

        def event_handler(self):
            global flag_vc
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.stop_stream()
                    exit()
                if event == "reload_devices" or event == "sg_hostapi":
                    self.gui_config.sg_hostapi = values["sg_hostapi"]
                    self.update_devices(hostapi_name=values["sg_hostapi"])
                    if self.gui_config.sg_hostapi not in self.hostapis:
                        self.gui_config.sg_hostapi = self.hostapis[0]
                    self.window["sg_hostapi"].Update(values=self.hostapis)
                    self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                    if (self.gui_config.sg_input_device not in self.input_devices and len(self.input_devices) > 0):
                        self.gui_config.sg_input_device = self.input_devices[0]
                    self.window["sg_input_device"].Update(values=self.input_devices)
                    self.window["sg_input_device"].Update(value=self.gui_config.sg_input_device)
                    if self.gui_config.sg_output_device not in self.output_devices:
                        self.gui_config.sg_output_device = self.output_devices[0]
                    self.window["sg_output_device"].Update(values=self.output_devices)
                    self.window["sg_output_device"].Update(value=self.gui_config.sg_output_device)

                if event == "start_vc" and not flag_vc:
                    if self.set_values(values) == True:
                        printt("🟢 CUDA Ready Status: %s", torch.cuda.is_available())
                        self.start_vc()
                        settings = {
                            "pth_path": values["pth_path"],
                            "index_path": values["index_path"],
                            "sg_hostapi": values["sg_hostapi"],
                            "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "sr_type": ["sr_model", "sr_device"][
                                [values["sr_model"], values["sr_device"]].index(True)
                            ],
                            "threhold": values["threhold"],
                            "pitch": values["pitch"],
                            "rms_mix_rate": values["rms_mix_rate"],
                            "index_rate": values["index_rate"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time": values["extra_time"],
                            "f0method": ["pm", "rmvpe", "fcpe"][
                                [values["pm"], values["rmvpe"], values["fcpe"]].index(True)
                            ],
                        }
                        with open(realtime_config_path, "w", encoding="utf8") as j:
                            json.dump(settings, j)
                        if self.stream is not None:
                            self.delay_time = (self.stream.latency[-1] + values["block_time"] + values["crossfade_length"] + 0.01)
                        if values["I_noise_reduce"]:
                            self.delay_time += min(values["crossfade_length"], 0.04)
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.window["delay_time"].update(int(np.round(self.delay_time * 1000)))

                # Hot Update Parameters
                if event == "threhold":
                    self.gui_config.threhold = values["threhold"]
                elif event == "pitch":
                    self.gui_config.pitch = values["pitch"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_key(values["pitch"])
                elif event == "formant":
                    self.gui_config.formant = values["formant"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_formant(values["formant"])
                elif event == "index_rate":
                    self.gui_config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event == "rms_mix_rate":
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                elif event in ["pm", "rmvpe", "fcpe"]:
                    self.gui_config.f0method = event
                elif event == "I_noise_reduce":
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]
                    if self.stream is not None:
                        self.delay_time += (1 if values["I_noise_reduce"] else -1) * min(values["crossfade_length"], 0.04)
                        self.window["delay_time"].update(int(np.round(self.delay_time * 1000)))
                elif event == "O_noise_reduce":
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]
                elif event in ["vc", "im"]:
                    self.function = event
                elif event == "stop_vc" or event != "start_vc":
                    self.stop_stream()

        def set_values(self, values):
            if len(values["pth_path"].strip()) == 0:
                sg.popup("กรุณาเลือกไฟล์โมเดล .pth ก่อน (Please select a .pth model file)", title="Warning", background_color='#CC2200')
                return False
            if len(values["index_path"].strip()) == 0:
                sg.popup("กรุณาเลือกไฟล์ดัชนี .index ก่อน (Please select an .index feature file)", title="Warning", background_color='#CC2200')
                return False
            pattern = re.compile(r"[\u4e00-\u9fff]+")
            if pattern.findall(values["pth_path"]):
                sg.popup("ชื่อหรือพาธไฟล์ .pth ห้ามมีตัวอักษรจีน (Path cannot contain Chinese characters)", title="Error", background_color='#CC2200')
                return False
            if pattern.findall(values["index_path"]):
                sg.popup("ชื่อหรือพาธไฟล์ .index ห้ามมีตัวอักษรจีน (Path cannot contain Chinese characters)", title="Error", background_color='#CC2200')
                return False
            
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            self.gui_config.sg_hostapi = values["sg_hostapi"]
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
            self.gui_config.sg_input_device = values["sg_input_device"]
            self.gui_config.sg_output_device = values["sg_output_device"]
            self.gui_config.pth_path = values["pth_path"]
            self.gui_config.index_path = values["index_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][ [values["sr_model"], values["sr_device"]].index(True) ]
            self.gui_config.threhold = values["threhold"]
            self.gui_config.pitch = values["pitch"]
            self.gui_config.formant = values["formant"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
            self.gui_config.index_rate = values["index_rate"]
            self.gui_config.f0method = ["pm", "rmvpe", "fcpe"][ [values["pm"], values["rmvpe"], values["fcpe"]].index(True) ]
            return True

        def start_vc(self):
            torch.cuda.empty_cache()
            self.rvc = rvc_for_realtime.RVC(
                self.gui_config.pitch,
                self.gui_config.formant,
                self.gui_config.pth_path,
                self.gui_config.index_path,
                self.gui_config.index_rate,
                self.config,
                self.rvc if hasattr(self, "rvc") else None,
            )
            self.gui_config.samplerate = (
                self.rvc.tgt_sr if self.gui_config.sr_type == "sr_model" else self.get_device_samplerate()
            )
            self.gui_config.channels = self.get_device_channels()
            self.zc = self.gui_config.samplerate // 100
            self.block_frame = int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc)) * self.zc
            self.block_frame_16k = 160 * self.block_frame // self.zc
            self.crossfade_frame = int(np.round(self.gui_config.crossfade_time * self.gui_config.samplerate / self.zc)) * self.zc
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = int(np.round(self.gui_config.extra_time * self.gui_config.samplerate / self.zc)) * self.zc
            
            self.input_wav = torch.zeros(self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame, device=self.config.device, dtype=torch.float32)
            self.input_wav_denoise = self.input_wav.clone()
            self.input_wav_res = torch.zeros(160 * self.input_wav.shape[0] // self.zc, device=self.config.device, dtype=torch.float32)
            self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
            self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
            self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
            self.nr_buffer = self.sola_buffer.clone()
            self.output_buffer = self.input_wav.clone()
            self.skip_head = self.extra_frame // self.zc
            self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
            
            self.fade_in_window = (
                torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)) ** 2
            )
            self.fade_out_window = 1 - self.fade_in_window
            self.resampler = tat.Resample(orig_freq=self.gui_config.samplerate, new_freq=16000, dtype=torch.float32).to(self.config.device)
            
            if self.rvc.tgt_sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(orig_freq=self.rvc.tgt_sr, new_freq=self.gui_config.samplerate, dtype=torch.float32).to(self.config.device)
            else:
                self.resampler2 = None
                
            self.tg = TorchGate(sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9).to(self.config.device)
            self.prewarm_cuda_graph()
            self.start_stream()

        def prewarm_cuda_graph(self):
            if not cuda_graph_enabled(self.config.device):
                return
            try:
                printt("🔥 [SYSTEM] กำลังวอร์มอัปกราฟ CUDA (Warming up CUDA Graph)...")
                samples = self.input_wav_res.shape[0]
                phase = torch.arange(samples, device=self.config.device, dtype=torch.float32)
                probe = 0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0)
                self.input_wav_res.copy_(probe)

                if self.gui_config.I_noise_reduce:
                    short = self.input_wav[-self.sola_buffer_frame - self.block_frame :].unsqueeze(0)
                    self.tg(short, self.input_wav.unsqueeze(0))

                resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
                run_cuda_graph(self.resampler, "realtime-input-resample", lambda audio: self.resampler(audio), resample_input)

                inferred = self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, self.gui_config.f0method)
                if self.resampler2 is not None:
                    inferred = run_cuda_graph(self.resampler2, "realtime-output-resample", lambda audio: self.resampler2(audio), inferred)
                if self.gui_config.O_noise_reduce:
                    self.tg(inferred.unsqueeze(0), self.output_buffer.unsqueeze(0))
                torch.cuda.synchronize(self.config.device)
                printt("✅ [SYSTEM] วอร์มอัปเสร็จสมบูรณ์ (CUDA Graph warmup complete)")
            except Exception:
                printt(traceback.format_exc())
            finally:
                self.input_wav.zero_()
                self.input_wav_denoise.zero_()
                self.input_wav_res.zero_()
                self.output_buffer.zero_()
                self.sola_buffer.zero_()
                self.nr_buffer.zero_()
                self.rvc.cache_pitch.zero_()
                self.rvc.cache_pitchf.zero_()

        def start_stream(self):
            global flag_vc
            if not flag_vc:
                flag_vc = True
                extra_settings = sd.WasapiSettings(exclusive=True) if ("WASAPI" in self.gui_config.sg_hostapi and self.gui_config.sg_wasapi_exclusive) else None
                self.stream = sd.Stream(
                    callback=self.audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.gui_config.samplerate,
                    channels=self.gui_config.channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
                self.stream.start()

        def stop_stream(self):
            global flag_vc
            if flag_vc:
                flag_vc = False
                if self.stream is not None:
                    self.stream.abort()
                    self.stream.close()
                    self.stream = None
                clear_vram_safe() # Explicit Memory Cleanup when stopped

        def audio_callback(self, indata, outdata, frames, times, status):
            global flag_vc
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            
            if self.gui_config.threhold > -60:
                indata = np.append(self.rms_buffer, indata)
                rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
                self.rms_buffer[:] = indata[-4 * self.zc :]
                indata = indata[2 * self.zc - self.zc // 2 :]
                db_threhold = (librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold)
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
                indata = indata[self.zc // 2 :]
                
            self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
            
            if self.gui_config.I_noise_reduce:
                self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
                input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
                input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
                input_wav[: self.sola_buffer_frame] *= self.fade_in_window
                input_wav[: self.sola_buffer_frame] += (self.nr_buffer * self.fade_out_window)
                self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
                self.nr_buffer[:] = input_wav[self.block_frame :]
                resample_input = self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
                self.input_wav_res[-self.block_frame_16k - 160 :] = run_cuda_graph(
                    self.resampler, "realtime-input-resample", lambda audio: self.resampler(audio), resample_input
                )[160:]
            else:
                resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
                self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = run_cuda_graph(
                    self.resampler, "realtime-input-resample", lambda audio: self.resampler(audio), resample_input
                )[160:]
                
            if self.function == "vc":
                infer_wav = self.rvc.infer(self.input_wav_res, self.block_frame_16k, self.skip_head, self.return_length, self.gui_config.f0method)
                if self.resampler2 is not None:
                    infer_wav = run_cuda_graph(
                        self.resampler2, "realtime-output-resample", lambda audio: self.resampler2(audio), infer_wav
                    )
            elif self.gui_config.I_noise_reduce:
                infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
            else:
                infer_wav = self.input_wav[self.extra_frame :].clone()
                
            if self.gui_config.O_noise_reduce and self.function == "vc":
                self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
                
            if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
                input_wav = self.input_wav_denoise[self.extra_frame :] if self.gui_config.I_noise_reduce else self.input_wav[self.extra_frame :]
                rms1 = librosa.feature.rms(y=input_wav[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
                rms1 = torch.from_numpy(rms1).to(self.config.device)
                rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
                
                rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc)
                rms2 = torch.from_numpy(rms2).to(self.config.device)
                rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                infer_wav *= torch.pow(rms1 / rms2, 1.0 - self.gui_config.rms_mix_rate)
                
            conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
            
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
                
            infer_wav = infer_wav[sola_offset:]
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (self.sola_buffer * self.fade_out_window)
            self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
            
            outdata[:] = infer_wav[: self.block_frame].repeat(self.gui_config.channels, 1).t().cpu().numpy()
            total_time = time.perf_counter() - start_time
            if flag_vc:
                self.window["infer_time"].update(int(total_time * 1000))

        def update_devices(self, hostapi_name=None):
            global flag_vc
            flag_vc = False
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if hostapi_name not in self.hostapis:
                hostapi_name = self.hostapis[0]
            self.input_devices = [d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name]
            self.output_devices = [d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name]
            self.input_devices_indices = [d["index"] if "index" in d else d["name"] for d in devices if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name]
            self.output_devices_indices = [d["index"] if "index" in d else d["name"] for d in devices if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name]

        def set_devices(self, input_device, output_device):
            self.input_devices_indices[self.input_devices.index(input_device)]
            sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
            sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
            printt("🎙️ อุปกรณ์รับเสียง (Input Device): %s: %s", str(sd.default.device[0]), input_device)
            printt("🔊 อุปกรณ์ขยายเสียง (Output Device): %s: %s", str(sd.default.device[1]), output_device)

        def get_device_samplerate(self):
            return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])

        def get_device_channels(self):
            max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
            max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
            return min(max_input_channels, max_output_channels, 2)

    gui = GUI()
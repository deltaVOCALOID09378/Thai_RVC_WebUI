# ==============================================================================
# File: onnx_inference_demo.py
# Path: tools\onnx_inference_demo.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 2.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# Description: Studio-Grade Audio Processing & Conversion Engine
import soundfile

from ..infer.lib.infer_pack.onnx_inference import OnnxRVC

hop_size = 512
sampling_rate = 40000  # Sample rate
f0_up_key = 0  # Pitch shift
sid = 0  # Speaker ID
f0_method = "dio"  # F0
model_path = "ShirohaRVC.onnx"
vec_name = "vec-256-layer-9"  # f"pretrained/{vec_name}.onnx" onnxvec
wav_path = "123.wav"  # ByteIO
out_path = "out.wav"  # ByteIO

model = OnnxRVC(
    model_path, vec_path=vec_name, sr=sampling_rate, hop_size=hop_size, device="cuda"
)

audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)

soundfile.write(out_path, audio, sampling_rate)

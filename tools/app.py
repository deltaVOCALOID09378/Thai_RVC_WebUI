# -*- coding: utf-8 -*-
# ==============================================================================
# File: app.py
# Path: tools\app.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 2.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
# Description: Lightweight RVC Inference Demo Interface (Thai & English)

import logging
import os
import gradio as gr
from dotenv import load_dotenv

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.modules.vc.modules import VC

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

i18n = I18nAuto()
logger.info(i18n)

load_dotenv()
config = Config()
vc = VC(config)

weight_root = os.getenv("weight_root", "assets/weights")
weight_uvr5_root = os.getenv("weight_uvr5_root", "assets/uvr5_weights")
index_root = os.getenv("index_root", "logs")
names = []
hubert_model = None

if os.path.exists(weight_root):
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)

index_paths = []
if os.path.exists(index_root):
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))


app = gr.Blocks(title="DELTA RVC Live Demo")
with app:
    with gr.Tabs():
        with gr.TabItem("การอนุมานเสียงสด (Live Demo)"):
            gr.Markdown(
                value="""
                ### 🎙️ DELTA SYNTH RVC Live Demo (การแปลงเสียงสดระดับสตูดิโอ)
                ทดสอบการแปลงเสียงด้วยโมเดล RVC ได้อย่างรวดเร็ว (Fast Single Inference Testing)
                """
            )
            sid = gr.Dropdown(label="เลือกโมเดลเสียงเป้าหมาย (Select Target Voice)", choices=sorted(names))
            with gr.Column():
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="หมายเลขผู้พูด (Speaker ID)",
                    value=0,
                    visible=False,
                    interactive=True,
                )
            sid.change(fn=vc.get_vc, inputs=[sid], outputs=[spk_item])
            gr.Markdown(
                value="💡 **คำแนะนำการปรับคีย์ (Pitch Shifting Tips):** ชาย ➡️ หญิง ปรับ `+12`, หญิง ➡️ ชาย ปรับ `-12` (Male to Female +12, Female to Male -12)"
            )
            vc_input3 = gr.Audio(label="อัปโหลดไฟล์เสียงต้นฉบับ (Upload Source Audio)")
            vc_transform0 = gr.Number(label="ปรับระดับเสียง (Pitch Shift: Semitones)", value=0)
            f0method0 = gr.Radio(
                label="อัลกอริทึมสกัดระดับเสียง (Pitch Extraction Algorithm)",
                choices=["pm", "harvest", "crepe", "rmvpe"],
                value="rmvpe",
                interactive=True,
            )
            filter_radius0 = gr.Slider(
                minimum=0,
                maximum=7,
                label="รัศมีตัวกรองเสียงรบกวน (Median Filter Radius)",
                value=3,
                step=1,
                interactive=True,
            )
            with gr.Column():
                file_index1 = gr.Textbox(
                    label="เส้นทางไฟล์ดัชนีระบุเอง (Custom Index Path)",
                    value="",
                    interactive=False,
                    visible=False,
                )
            file_index2 = gr.Dropdown(
                label="เลือกไฟล์ดัชนีอัตโนมัติ (Auto-Select Index Path)",
                choices=sorted(index_paths),
                interactive=True,
            )
            index_rate1 = gr.Slider(
                minimum=0,
                maximum=1,
                label="สัดส่วนการใช้อิทธิพลดัชนี (Index Influence Ratio)",
                value=0.75,
                interactive=True,
            )
            resample_sr0 = gr.Slider(
                minimum=0,
                maximum=48000,
                label="อัตราสุ่มตัวอย่างซ้ำ (Resampling Rate, 0 = Off)",
                value=0,
                step=1,
                interactive=True,
            )
            rms_mix_rate0 = gr.Slider(
                minimum=0,
                maximum=1,
                label="ความเข้มข้นของการผสานระดับเสียง (Volume Envelope Mix Rate)",
                value=1.0,
                interactive=True,
            )
            protect0 = gr.Slider(
                minimum=0,
                maximum=0.5,
                label="การปกป้องพยัญชนะและเสียงหายใจ (Breath & Consonant Protection)",
                value=0.33,
                step=0.01,
                interactive=True,
            )
            f0_file = gr.File(label="ไฟล์เส้นโค้งระดับเสียง (F0 Curve File, Optional)", visible=False)
            but0 = gr.Button("🚀 เริ่มแปลงเสียง (Start Convert)", variant="primary")
            vc_output1 = gr.Textbox(label="สถานะการประมวลผล (Operation Status & Log)")
            vc_output2 = gr.Audio(label="ผลลัพธ์ไฟล์เสียง (Output Audio)")
            but0.click(
                vc.vc_single,
                [
                    spk_item,
                    vc_input3,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index1,
                    file_index2,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2],
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7866, quiet=True)

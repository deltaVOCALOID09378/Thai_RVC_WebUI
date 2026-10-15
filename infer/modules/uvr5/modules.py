# ==============================================================================
# Made And Checked By DELTA SYNTH & All Agentic AI
# Original by RVC-Project
# File: infer/modules/uvr5/modules.py | Version: 1.3
# Description: [EN] UVR5 Vocal Separation Module / [TH] โมดูลแยกเสียงร้องและเสียงดนตรี
# ==============================================================================
import os
import traceback
import logging
import gc
from configs.config import Config

logger = logging.getLogger(__name__)
config = Config()

def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n")
        save_root_vocal = save_root_vocal.strip(" ").strip('"').strip("\n")
        save_root_ins = save_root_ins.strip(" ").strip('"').strip("\n")
        is_hp3 = "HP3" in model_name
        
        if model_name == "onnx_dereverb_By_FoxJoy":
            from infer.modules.uvr5.mdxnet import MDXNetDereverb
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            from infer.modules.uvr5.preprocess import AudioPre, AudioPreDeEcho
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root", "assets/uvr5_weights"), 
                    model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )
            
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
            
        for path in paths:
            inp_path = os.path.join(inp_root, path) if inp_root != "" else path
            if not os.path.isfile(inp_path):
                continue
            try:
                # [ZERO-DEFECT FIX]: Handle Dereverb method signature mismatch dynamically
                if model_name == "onnx_dereverb_By_FoxJoy":
                    pre_fun.path_audio(inp_path, save_root_vocal, save_root_ins, format0)
                else:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                    )
                success_msg = f"🟢 แยกเสียงสำเร็จ (Separation Successful): {os.path.basename(inp_path)}"
                infos.append(success_msg)
                logger.info(success_msg)
                yield "\n".join(infos)
            except Exception as e:
                error_msg = f"🔴 เกิดข้อผิดพลาดกับไฟล์ (Error processing file) {os.path.basename(inp_path)}: {str(e)}"
                infos.append(error_msg)
                logger.error(error_msg)
                yield "\n".join(infos)
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"🔴 ข้อผิดพลาดร้ายแรง (Critical Error):\n{error_trace}")
        yield f"🔴 เกิดข้อผิดพลาดระบบ (System Error):\n{error_trace}"
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except Exception:
            pass
        
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("✅ คืนค่าพื้นที่หน่วยความจำเรียบร้อยแล้ว (Memory Cleared Successfully)")
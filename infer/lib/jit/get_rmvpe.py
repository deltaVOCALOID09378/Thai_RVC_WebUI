# ==============================================================================
# File: get_rmvpe.py
# Path: upstream_clean_reference\Retrieval-based-Voice-Conversion-WebUI-2.2.231006\infer\lib\jit\get_rmvpe.py
# Description: RVC WebUI Core Module / โมดูลหลัก RVC WebUI
# Version: 1.0
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
import torch


def get_rmvpe(model_path="assets/rmvpe/rmvpe.pt", device=torch.device("cpu")):
    from infer.lib.rmvpe import E2E

    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model

# ==============================================================================
# File: __init__.py
# Path: tools\pymss_core\modules\vocal_remover\__init__.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from .uvr_lib_v5.vr_network.model_param_init import ModelParameters
from .uvr_lib_v5.vr_network.nets import BaseASPPNet, CascadedASPPNet, determine_model_capacity
from .uvr_lib_v5.vr_network.nets_new import BaseNet, CascadedNet

__all__ = (
    "BaseASPPNet",
    "BaseNet",
    "CascadedASPPNet",
    "CascadedNet",
    "ModelParameters",
    "determine_model_capacity",
)

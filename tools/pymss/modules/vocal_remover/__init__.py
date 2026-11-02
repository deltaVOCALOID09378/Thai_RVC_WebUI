# ==============================================================================
# File: __init__.py
# Path: tools\pymss\modules\vocal_remover\__init__.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from pymss_core.modules.vocal_remover import (
    BaseASPPNet,
    BaseNet,
    CascadedASPPNet,
    CascadedNet,
    ModelParameters,
    determine_model_capacity,
)

from .vr_separator import VRSeparator

__all__ = (
    "BaseASPPNet",
    "BaseNet",
    "CascadedASPPNet",
    "CascadedNet",
    "ModelParameters",
    "VRSeparator",
    "determine_model_capacity",
)

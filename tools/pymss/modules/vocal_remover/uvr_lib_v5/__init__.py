# ==============================================================================
# File: __init__.py
# Path: tools\pymss\modules\vocal_remover\uvr_lib_v5\__init__.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from pymss.modules._core_shims import alias_submodules

alias_submodules(
    __name__,
    "pymss_core.modules.vocal_remover.uvr_lib_v5",
    (
        "vr_network",
        "vr_network.layers",
        "vr_network.layers_new",
        "vr_network.model_param_init",
        "vr_network.nets",
        "vr_network.nets_new",
    ),
)

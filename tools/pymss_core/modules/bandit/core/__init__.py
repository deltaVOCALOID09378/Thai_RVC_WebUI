# ==============================================================================
# File: __init__.py
# Path: tools\pymss_core\modules\bandit\core\__init__.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
__all__ = ("MultiMaskMultiSourceBandSplitRNNSimple",)


def __getattr__(name):
    if name == "MultiMaskMultiSourceBandSplitRNNSimple":
        from .model import MultiMaskMultiSourceBandSplitRNNSimple

        return MultiMaskMultiSourceBandSplitRNNSimple
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

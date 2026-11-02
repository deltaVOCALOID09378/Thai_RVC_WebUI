# ==============================================================================
# File: tfmodel.py
# Path: tools\pymss_core\modules\bandit_v2\tfmodel.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from ..bandit.tfmodel import (
    ResidualRNN,
    TimeFrequencyModellingModule,
    Transpose,
    _SeqBandModellingPreset,
)


class SeqBandModellingModule(_SeqBandModellingPreset):
    @staticmethod
    def _preset_runtime_options(n_modules, parallel_mode):
        return {
            "sequential_transpose": not parallel_mode,
            "checkpoint_segments": None if parallel_mode else n_modules,
        }


__all__ = ("ResidualRNN", "SeqBandModellingModule", "TimeFrequencyModellingModule", "Transpose")

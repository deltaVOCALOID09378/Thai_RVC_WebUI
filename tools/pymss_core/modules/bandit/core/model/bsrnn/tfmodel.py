# ==============================================================================
# File: tfmodel.py
# Path: tools\pymss_core\modules\bandit\core\model\bsrnn\tfmodel.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from ....tfmodel import (
    ResidualRNN,
    TimeFrequencyModellingModule,
    _SeqBandModellingPreset,
)


class SeqBandModellingModule(_SeqBandModellingPreset):
    pass


__all__ = ("ResidualRNN", "SeqBandModellingModule", "TimeFrequencyModellingModule")

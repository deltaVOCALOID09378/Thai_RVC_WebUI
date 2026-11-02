# ==============================================================================
# File: bandsplit.py
# Path: tools\pymss_core\modules\bandit_v2\bandsplit.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from ..bandit.bandsplit import (
    SequentialNormFC as NormFC,
    _ConfiguredBandSplitModule,
)


class BandSplitModule(_ConfiguredBandSplitModule):
    norm_fc_cls = NormFC
    complex_order = "freq_reim"
    flatten_input = True


__all__ = ("BandSplitModule", "NormFC")

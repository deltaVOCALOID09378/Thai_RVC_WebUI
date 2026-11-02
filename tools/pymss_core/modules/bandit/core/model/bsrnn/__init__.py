# ==============================================================================
# File: __init__.py
# Path: tools\pymss_core\modules\bandit\core\model\bsrnn\__init__.py
# Description: RVC WebUI Core Module / โมดูลระบบ RVC WebUI
# Version: 1.1
# Made And Checked By DELTA SYNTH & All Code Agentic AI Engine
# [ Original by "RVC-Project" ]
# ==============================================================================
from abc import ABC
from typing import Iterable, Mapping, Union

from torch import nn


class BandsplitCoreBase(nn.Module, ABC):
    band_split: nn.Module
    tf_model: nn.Module
    mask_estim: Union[nn.Module, Mapping[str, nn.Module], Iterable[nn.Module]]

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def mask(x, m):
        return x * m

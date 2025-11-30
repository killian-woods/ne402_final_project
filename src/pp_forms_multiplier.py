from functools import cached_property

import numpy as np

from water import Water


class TwoPhaseFormsMultiplier:
    def __init__(self):
        self.water = Water()

    def discrete_2p_forms_multiplier(self, x: float) -> float:
        return self._static_1 + self._static_2 * x

    @cached_property
    def _static_1(self) -> float:
        return 1  # assuming equilibrium means that rho_f = rho_l in 2p region

    @cached_property
    def _static_2(self) -> float:
        rho_f = self.water.liquid_water.rho
        rho_l = rho_f  # equilibrium
        rho_g = self.water.vapor_water.rho
        return (1 / rho_g - 1 / rho_l) / (1 / rho_f)

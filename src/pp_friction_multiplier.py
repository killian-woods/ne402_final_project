from functools import cached_property

import numpy as np

from water import Water


class TwoPhaseFrictionMultiplier:
    def __init__(self):
        self.water = Water()

    def discrete_2p_friction_multiplier(self, x: float) -> float:
        if x <= 0:
            print("x <= 0 for phi_lo. phi_lo = 1")
            return 1
        chi = self.chi(x=x)
        chi = max(chi, 0.0001)
        return (1 + 20 / chi + 1 / chi**2) * ((1 - x) ** 1.8)

    def chi(self, x: float) -> float:
        if x <= 0:
            return 99999999999999
        return np.sqrt(self._chi_static * (((1 - x) / x) ** 1.8))

    # i doubt the performance gains from turning this part into a cached property will matter but...
    # eh... i might as well. any bit helps, right? right?? please tell me im right
    @cached_property
    def _chi_static(self) -> float:
        mu_f = self.water.liquid_water.mu
        mu_g = self.water.vapor_water.mu
        rho_f = self.water.liquid_water.rho
        rho_g = self.water.vapor_water.rho

        return ((mu_f / mu_g) ** 0.2) * (rho_g / rho_f)

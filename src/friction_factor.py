from pathlib import Path
from functools import cache, cached_property

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

from reactor import Reactor
from enthalpy import Enthalpy
from water import Water
from quality import Quality
from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.units.consts import G, G_C
from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot
from utilities.units.iapws97_imperial import ImperialIAPWS97


class FrictionFactor:
    def __init__(self):
        self.reactor = Reactor

    def smooth_friction_factor(self, reynolds: float) -> float:
        if reynolds <= 0:
            raise ValueError("Reynolds number is unrealistic")
        if reynolds <= 2000:
            return 64 / reynolds
        if reynolds <= 30000:
            return 0.3164 * (reynolds ** (-0.25))
        return 0.184 * (reynolds ** (-0.2))

    def rough_friction_factor(self, reynolds: float, D_e: float) -> float:

        # if the material isnt smooth (the reactor core), it is made out of turned stainless steel
        # and has an absolute roughness, as prescribed by the Reactor class
        def LHS(f: float) -> float:
            return 1 / np.sqrt(f)

        def RHS(f: float) -> float:
            return -2 * np.log10(
                (self.reactor.steel_abs_roughness / D_e) / 3.7
                + (2.51) / (reynolds * np.sqrt(f))
            )

        def residual(f: float) -> float:
            return LHS(f) - RHS(f)

        sol = root_scalar(
            residual, bracket=[0.0000000001, 0.9999999999], method="brentq"
        )
        if not sol.converged:
            raise RuntimeError(f"colebrook-White solver failed: {sol.flag}")
        return sol.root

from pathlib import Path
from functools import cache, cached_property

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

from reactor import Reactor
from enthalpy import Enthalpy
from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot
from utilities.units.iapws97_imperial import ImperialIAPWS97


class Water:
    def __init__(self):
        self.reactor = Reactor
        self.enthalpy = Enthalpy()

    @cache
    def avg_water(
        self, z: float, avg_mflux: float, hot_mflux: float = None
    ) -> ImperialIAPWS97:
        h = self.enthalpy.avg_enthalpy(z=z, avg_mflux=avg_mflux)
        return self._water(h=h)

    @cache
    def hot_water(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> ImperialIAPWS97:
        h = self.enthalpy.hot_enthalpy(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        return self._water(h=h)

    def _water(self, h: float) -> ImperialIAPWS97:
        water = ImperialIAPWS97(
            P=self.reactor.pressure, h=h, use_inches=True, use_hours=True
        )
        return water

    @cached_property
    def liquid_water(self) -> ImperialIAPWS97:
        return ImperialIAPWS97(
            P=self.reactor.pressure, x=0, use_inches=True, use_hours=True
        )

    @cached_property
    def vapor_water(self) -> ImperialIAPWS97:
        return ImperialIAPWS97(
            P=self.reactor.pressure, x=1, use_inches=True, use_hours=True
        )

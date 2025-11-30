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


class VoidFraction:
    def __init__(self):
        self.reactor = Reactor
        self.enthalpy = Enthalpy()
        self.water = Water()
        self.quality = Quality()

    def avg_void_fraction(
        self, z: float, avg_mflux: float, hot_mflux: float = None
    ) -> float:
        x = self.quality.avg_quality(z=z, avg_mflux=avg_mflux)
        water = self.water.avg_water(z=z, avg_mflux=avg_mflux)
        ans = self._void_fraction(
            avg_mflux=avg_mflux, hot_mflux=avg_mflux, x=x, water=water
        )
        return ans

    def hot_void_fraction(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        x = self.quality.hot_quality(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        water = self.water.hot_water(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        return self._void_fraction(
            avg_mflux=avg_mflux, hot_mflux=hot_mflux, x=x, water=water
        )

    def _void_fraction(
        self, avg_mflux: float, hot_mflux: float, x: float, water: ImperialIAPWS97
    ) -> float:
        if x <= 0:
            return 0
        Co = self._Co(x=x)
        V_gj = self._V_gj(water=water)
        rho_g = self.water.vapor_water.rho
        rho_l = self.water.liquid_water.rho

        ans = (
            Co * (1 + ((1 - x) / x) * rho_g / rho_l) + (rho_g * V_gj) / (hot_mflux * x)
        ) ** -1
        if ans < 0:
            ans = 0
        elif ans > 1:
            ans = 1
        return ans

    def _Co(self, x: float) -> float:
        beta = self._beta(x)
        b = self._b

        return beta * (1 + (1 / beta - 1) ** b) if beta > 0 else 0

    def _beta(self, x: float) -> float:
        rho_g = self.water.vapor_water.rho
        rho_l = self.water.liquid_water.rho

        return x / (x + (1 - x) * (rho_g / rho_l)) if x > 0 else 0

    @cached_property
    def _b(self) -> float:
        return (self.water.vapor_water.rho / self.water.liquid_water.rho) ** (0.1)

    def _V_gj(self, water: ImperialIAPWS97) -> float:

        sigma = water.sigma * 12
        g2 = G * G_C  # dont worry about time units
        rho_g = self.water.vapor_water.rho * 12**3
        rho_f = self.water.liquid_water.rho * 12**3

        # converted everything to per/ft before calculating
        return (
            1.41 * (((sigma * g2 * (rho_f - rho_g)) / (rho_f**2)) ** (1 / 4)) * 12
        ) * 3600
        # converted answer (V_gj) into in/hr cause it's easier to work with

    def plot_void_fractions(
        self,
        hot_mflux: float = None,
        avg_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        if hot_mflux is None or avg_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        alphahot_z = [
            self.hot_void_fraction(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        alphaavg_z = [self.avg_void_fraction(z=zi, avg_mflux=avg_mflux) for zi in z]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, alphahot_z, label=r"$\alpha_{hot}(z)$", color=palette.next())
        ax.plot(z, alphaavg_z, label=r"$\alpha_{avg}(z)$", color=palette.next())

        hot_z_d = self.enthalpy.hot_z_d(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
        avg_z_d = self.enthalpy.avg_z_d(avg_mflux=avg_mflux, hot_mflux=avg_mflux)

        if hot_z_d is not None:
            ax.axvline(
                hot_z_d,
                color=palette.COLORS[0],
                linestyle="--",
                linewidth=1,
                label=r"$z_{d,hot}$",
            )

        if avg_z_d is not None:
            ax.axvline(
                avg_z_d,
                color=palette.COLORS[1],
                linestyle="--",
                linewidth=1,
                label=r"$z_{d,avg}$",
            )

        format_plot(
            ax,
            title="Axial Void Fraction Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Void Fraction $\alpha(z)$ [-]",
            grid=True,
            legend=True,
        )

        if avg_z_d is not None and hot_z_d is not None:
            ax.get_lines()[-2].set_linewidth(1)
        if avg_z_d is not None or hot_z_d is not None:
            ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="void_fractions_distributions")

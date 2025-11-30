from pathlib import Path
from functools import cache

import numpy as np
import matplotlib.pyplot as plt

from reactor import Reactor
from enthalpy import Enthalpy
from water import Water
from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot


class Quality:
    def __init__(self):
        self.reactor = Reactor
        self.enthalpy = Enthalpy()
        self.water = Water()

    @cache
    def avg_quality(self, z: float, avg_mflux: float, hot_mflux: float = None) -> float:
        h = self.enthalpy.avg_enthalpy(z=z, avg_mflux=avg_mflux)
        return self._quality(h=h)

    @cache
    def hot_quality(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        h = self.enthalpy.hot_enthalpy(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        return self._quality(h=h)

    def _quality(self, h: float) -> float:
        top = h - self.water.liquid_water.h
        bot = self.water.vapor_water.h - self.water.liquid_water.h

        sol = min(max(top / bot, 0), 1)
        return sol

    def plot_qualities(
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

        xhot_z = [
            self.hot_quality(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux) for zi in z
        ]
        xavg_z = [self.avg_quality(z=zi, avg_mflux=avg_mflux) for zi in z]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, xhot_z, label=r"$x_{hot}(z)$", color=palette.next())
        ax.plot(z, xavg_z, label=r"$x_{avg}(z)$", color=palette.next())

        hot_z_d = self.enthalpy.hot_z_d(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
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
            title="Axial Quality Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Quality $x(z)$ [-]",
            grid=True,
            legend=True,
        )

        if avg_z_d is not None and hot_z_d is not None:
            ax.get_lines()[-2].set_linewidth(1)
        if avg_z_d is not None or hot_z_d is not None:
            ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="quality_distributions")

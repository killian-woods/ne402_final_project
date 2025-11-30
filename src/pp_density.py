from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from reactor import Reactor
from water import Water
from void_fraction import VoidFraction
from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot


class TwoPhaseDensity:
    def __init__(self):
        self.reactor = Reactor
        self.void_fraction = VoidFraction()
        self.enthalpy = self.void_fraction.enthalpy
        self.water = self.void_fraction.water

    def avg_mixture_density(
        self, z: float, avg_mflux: float, hot_mflux: float = None
    ) -> float:
        z = min(z, self.reactor.H_core)
        alpha_g = self.void_fraction.avg_void_fraction(z=z, avg_mflux=avg_mflux)
        if alpha_g <= 0:
            return self.water.avg_water(
                z=z,
                avg_mflux=avg_mflux,
            ).rho
        return self._mixture_density(alpha_g=alpha_g)

    def hot_mixture_density(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> float:
        z = min(z, self.reactor.H_core)
        alpha_g = self.void_fraction.hot_void_fraction(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
        )
        if alpha_g <= 0:
            return self.water.hot_water(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            ).rho
        return self._mixture_density(alpha_g=alpha_g)

    def _mixture_density(self, alpha_g: float) -> float:
        alpha_l = 1 - alpha_g
        return (
            self.water.liquid_water.rho * alpha_l + self.water.vapor_water.rho * alpha_g
        )

    def plot_densities(
        self,
        hot_mflux: float = None,
        avg_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        palette = DefaultPalette()
        z = np.linspace(0, self.reactor.H_core, num_points)
        rho_hot_z = [
            self.hot_mixture_density(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        rho_avg_z = [self.avg_mixture_density(z=zi, avg_mflux=avg_mflux) for zi in z]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, rho_hot_z, label=r"$\rho_{hot}(z)$", color=palette.next())
        ax.plot(z, rho_avg_z, label=r"$\rho_{avg}(z)$", color=palette.next())

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

        ax.axhline(
            self.water.liquid_water.rho,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$\rho_f$",
        )
        ax.axhline(
            self.water.vapor_water.rho,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$\rho_g$",
        )

        format_plot(
            ax,
            title="Axial Mixture Density Distribution",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Mixture Density $\rho(z)$ [lbm/in^3]",
            grid=True,
            legend=True,
        )

        if avg_z_d is not None and hot_z_d is not None:
            ax.get_lines()[-4].set_linewidth(1)
        if avg_z_d is not None or hot_z_d is not None:
            ax.get_lines()[-3].set_linewidth(1)

        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="mixture_density_distribution")

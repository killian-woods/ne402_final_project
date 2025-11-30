from pathlib import Path
from functools import cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

from reactor import Reactor
from heat_flux import HeatFlux
from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot
from utilities.units.iapws97_imperial import ImperialIAPWS97


class Enthalpy:
    def __init__(self):
        self.reactor = Reactor
        self.qflux = HeatFlux()
        self.liquid_water = ImperialIAPWS97(
            P=self.reactor.pressure, x=0, use_inches=True, use_hours=True
        )
        self.vapor_water = ImperialIAPWS97(
            P=self.reactor.pressure, x=1, use_inches=True, use_hours=True
        )

    @cache
    def avg_enthalpy(
        self, z: float, avg_mflux: float, hot_mflux: float = None
    ) -> float:
        return self._enthalpy(
            avg_mflux=avg_mflux,
            hot_mflux=avg_mflux,
            qflux_integrated=self.qflux.avg_heat_flux_integrated(z=z),
        )

    @cache
    def hot_enthalpy(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        return self._enthalpy(
            avg_mflux=avg_mflux,
            hot_mflux=hot_mflux,
            qflux_integrated=self.qflux.hot_heat_flux_integrated(z=z),
        )

    def _enthalpy(
        self, avg_mflux: float, hot_mflux: float, qflux_integrated: float
    ) -> float:
        coeff = self.reactor.core_power / (
            (avg_mflux * self.reactor.A * self.reactor.num_rods)
            * (self.vapor_water.h - self.reactor.feed_water.h)
        )
        h_in = coeff * self.reactor.feed_water.h + (1 - coeff) * self.liquid_water.h
        return (
            h_in
            + (self.reactor.P_w / ((hot_mflux * self.reactor.A) * self.reactor.gamma_f))
            * qflux_integrated
        )

    def avg_z_d(self, avg_mflux: float, hot_mflux: float) -> float:
        _LHS = self.liquid_water.h

        def residual(z: float) -> float:
            LHS = _LHS
            RHS = self.avg_enthalpy(z=z, avg_mflux=avg_mflux)
            return LHS - RHS

        return self._z_d(residual=residual)

    def hot_z_d(self, avg_mflux: float, hot_mflux: float) -> float:
        _LHS = self.liquid_water.h

        def residual(z: float) -> float:
            LHS = _LHS
            RHS = self.hot_enthalpy(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            return LHS - RHS

        return self._z_d(residual=residual)

    def _z_d(self, residual: callable) -> float:
        if residual(z=0) < 0:
            return 0
        if residual(z=self.reactor.H_core) > 0:
            return None
        sol = root_scalar(
            residual,
            bracket=[0, self.reactor.H_core],
            method="brentq",
        )

        if not sol.converged:
            raise ArithmeticError(f"z_d solver did not converge. {sol.flag}")
        return sol.root

    def plot_enthalpys(
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

        hhot_z = [
            self.hot_enthalpy(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        havg_z = [self.avg_enthalpy(z=zi, avg_mflux=avg_mflux) for zi in z]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, hhot_z, label=r"$h_{hot}(z)$", color=palette.next())
        ax.plot(z, havg_z, label=r"$h_{avg}(z)$", color=palette.next())

        hot_z_d = self.hot_z_d(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
        avg_z_d = self.avg_z_d(avg_mflux=avg_mflux, hot_mflux=avg_mflux)

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
            self.liquid_water.h,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$h_{f}$",
        )

        format_plot(
            ax,
            title="Axial Enthalpy Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Enthalpy $h(z)$ [BTU/(lbm)]",
            grid=True,
            legend=True,
        )

        if avg_z_d is not None and hot_z_d is not None:
            ax.get_lines()[-3].set_linewidth(1)
        if avg_z_d is not None or hot_z_d is not None:
            ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="enthalpy_distributions")

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import quad

from reactor import Reactor
from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot


class HeatFlux:
    def __init__(self):
        self.reactor = Reactor

        self.avg_avg = self.reactor.rod_power / (
            np.pi * self.reactor.D_o * self.reactor.H_core
        )
        self.hot_avg = self.avg_avg * self.reactor.F_q / self.reactor.F_z

        self.avg_lambda = self._compute_lambda()
        self.hot_lambda = self.avg_lambda
        self.avg_q0 = self._compute_q0("avg")
        self.hot_q0 = self._compute_q0("hot")

    def shape(self, z: float, lamb: float) -> float:
        x = np.pi * (self.reactor.H_core + lamb - z) / (self.reactor.H_core + 2 * lamb)
        return x * np.sin(x)

    def _compute_lambda(self) -> float:

        sol = root_scalar(
            self._lambda_residual,
            x0=self.reactor.H_core * 0.05,
            x1=self.reactor.H_core * 0.25,
            method="secant",
        )
        if not sol.converged:
            raise ArithmeticError("_compute_lambda did not converge.")
        return sol.root

    def _lambda_residual(self, lamb: float) -> float:
        sol = self._lambda_max(lamb) / self._lambda_avg(lamb) - self.reactor.F_z
        return sol

    def _lambda_max(self, lamb: float) -> float:
        H_core = self.reactor.H_core

        def residual(z: float) -> float:
            return ((np.pi * -1) / (H_core + 2 * lamb)) * (
                np.sin((np.pi * (H_core + lamb - z)) / (H_core + 2 * lamb))
                + ((np.pi * (H_core + lamb - z)) / (H_core + 2 * lamb))
                * np.cos((np.pi * (H_core + lamb - z)) / (H_core + 2 * lamb))
            )

        sol = root_scalar(residual, bracket=(0, H_core), method="brentq")

        if not sol.converged:
            raise ArithmeticError("_lambda_max did not converge.")
        z_max = sol.root

        def q_func(z: float) -> float:
            sol = ((np.pi * (H_core + lamb - z)) / (H_core + 2 * lamb)) * np.sin(
                (np.pi * (H_core + lamb - z)) / (H_core + 2 * lamb)
            )
            return sol

        return q_func(z_max)

    def _lambda_avg(self, lamb: float) -> float:
        H_core = self.reactor.H_core

        def _integral(z: float) -> float:
            sol = self.shape(z=z, lamb=lamb)
            return sol

        # sol = (1 / H_core) * (_integral(H_core) - _integral(0))
        integral_part, _ = quad(_integral, 0, H_core)
        sol = (1 / H_core) * integral_part
        return sol

    def _compute_q0(self, avg_or_hot: str) -> float:
        if avg_or_hot is None or (avg_or_hot != "avg" and avg_or_hot != "hot"):
            raise ValueError("Invalid avg_or_hot arg!")
        if self.avg_lambda is None:
            self.avg_lambda = self._compute_lambda()
        if self.hot_lambda is None:
            self.hot_lambda = self.avg_lambda

        if avg_or_hot == "avg":
            rod_flux = self.avg_avg
            lamb = self.avg_lambda
        elif avg_or_hot == "hot":
            rod_flux = self.hot_avg
            lamb = self.hot_lambda
        else:
            raise ValueError("wtf happened here")

        int_avg = self._lambda_avg(lamb=lamb)

        return rod_flux / int_avg

    def avg_heat_flux(self, z: float) -> float:
        return self._heat_flux(z=z, lamb=self.avg_lambda) * self.avg_q0

    def hot_heat_flux(self, z: float) -> float:
        return self._heat_flux(z=z, lamb=self.hot_lambda) * self.hot_q0

    def _heat_flux(self, z: float, lamb: float) -> float:
        H_core = self.reactor.H_core
        sol = ((np.pi * (H_core + lamb - z)) / (H_core + 2 * lamb)) * np.sin(
            (np.pi * (H_core + lamb - z)) / (H_core + 2 * lamb)
        )
        return sol

    def avg_heat_flux_integrated(self, z: float) -> float:
        return self._heat_flux_integrated(z=z, lamb=self.avg_lambda, q0=self.avg_q0)

    def hot_heat_flux_integrated(self, z: float) -> float:
        return self._heat_flux_integrated(z=z, lamb=self.hot_lambda, q0=self.hot_q0)

    def _heat_flux_integrated(self, z: float, lamb: float, q0: float) -> float:
        H_core = self.reactor.H_core
        H_e = H_core + 2 * lamb

        def _integral(z: float) -> float:
            sol = self.shape(z=z, lamb=lamb)
            return sol

        integral_part, _ = quad(_integral, a=0, b=z)
        sol = q0 * integral_part

        return sol

    def plot_heat_fluxs(
        self,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        qhot_z = [self.hot_heat_flux(zi) for zi in z]
        qavg_z = [self.avg_heat_flux(zi) for zi in z]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, qhot_z, label=r"$q''_{hot}(z)$", color=palette.next())
        ax.plot(z, qavg_z, label=r"$q''_{avg}(z)$", color=palette.next())

        format_plot(
            ax,
            title="Axial Heat Flux Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Heat Flux $q''(z)$ [BTU/(hr·in²)]",
            grid=True,
            legend=True,
        )

        save_plot(fig, folder=results_dir, name="heat_flux_distributions")

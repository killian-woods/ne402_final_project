from pathlib import Path
from functools import cache, cached_property

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize

from reactor import Reactor
from heat_flux import HeatFlux
from water import Water
from avg_flux_from_hot_flux import avg_flux_from_hot_flux
from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot
from utilities.units.iapws97_imperial import ImperialIAPWS97
from utilities.units.conversions import convert_units


class CriticalHeatFlux:
    def __init__(self):
        self.reactor = Reactor
        self.qflux = HeatFlux()
        self.water = Water()
        self.enthalpy = self.water.enthalpy
        self.critical_pressure = 22.12  # MPa

    def crit_enthalpy(
        self, z: float, avg_mflux: float, hot_mflux: float, q0_crit: float
    ) -> float:
        qflux_integrated = self.crit_heat_flux_integrated(z=z, q0_crit=q0_crit)
        coeff = self.reactor.core_power / (
            (avg_mflux * self.reactor.A * self.reactor.num_rods)
            * (self.enthalpy.vapor_water.h - self.reactor.feed_water.h)
        )
        h_in = (
            coeff * self.reactor.feed_water.h
            + (1 - coeff) * self.enthalpy.liquid_water.h
        )
        return (
            h_in
            + (self.reactor.P_w / ((hot_mflux * self.reactor.A) * self.reactor.gamma_f))
            * qflux_integrated
        )

    def crit_quality(
        self, z: float, avg_mflux: float, hot_mflux: float, q0_crit: float
    ) -> float:
        top = (
            self.crit_enthalpy(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=q0_crit
            )
            - self.water.liquid_water.h
        )
        bot = self.water.vapor_water.h - self.water.liquid_water.h
        return min(max(top / bot, 0), 1)

    def crit_water(
        self, z: float, avg_mflux: float, hot_mflux: float, q0_crit: float
    ) -> ImperialIAPWS97:
        h = self.crit_enthalpy(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=q0_crit
        )
        return ImperialIAPWS97(
            P=self.reactor.pressure, h=h, use_inches=True, use_hours=True
        )

    def crit_heat_flux(self, z: float, q0_crit: float) -> float:
        return self.qflux.hot_heat_flux(z=z) / self.qflux.hot_q0 * q0_crit

    def crit_heat_flux_integrated(self, z: float, q0_crit: float) -> float:
        return self.qflux.hot_heat_flux_integrated(z=z) / self.qflux.hot_q0 * q0_crit

    @cache
    def calculate_mflux_for_1dot2(
        self, avg_mflux: float, hot_mflux_guess: float
    ) -> float:
        def residual(hot_mflux: float) -> float:
            _avg_mflux = avg_flux_from_hot_flux(
                avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            LHS = 1.2
            RHS = self.calculate_min_CPR(avg_mflux=_avg_mflux, hot_mflux=hot_mflux)
            # print(RHS)
            return LHS - RHS

        sol = root_scalar(
            residual,
            x0=hot_mflux_guess,
            x1=hot_mflux_guess * 1.1,
            method="secant",
        )
        if not sol.converged:
            raise ArithmeticError(f"1.2 CPR solver did not converge. {sol.flag}")
        hot_mflux = sol.root
        return hot_mflux

    def calculate_critical_heat_flux_q0(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> float:
        # 1. solve for H_0,crit
        # 2. solve for q''0,crit
        # 3. return q''0,crit
        def residual(H_0_crit: float) -> float:
            return self._h0crit_resiudal(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, H_0_crit=H_0_crit
            )

        sol = root_scalar(
            residual,
            bracket=[0, z * 0.9],
            method="brentq",
        )
        if not sol.converged:
            raise ArithmeticError(f"H_0_crit solver did not converge. {sol.flag}")
        H_0_crit = sol.root
        # print(H_0_crit)

        return self._q0(avg_mflux=avg_mflux, hot_mflux=hot_mflux, H_0_crit=H_0_crit)

    def calculate_CPR(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        q0_crit = self.calculate_critical_heat_flux_q0(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
        )
        return q0_crit / self.qflux.hot_q0

    def calculate_min_CPR(self, avg_mflux: float, hot_mflux: float) -> float:
        def residual(z: float) -> float:
            return self.calculate_CPR(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)

        sol = minimize(
            residual, x0=[self.reactor.H_core * 0.75], bounds=[(0, self.reactor.H_core)]
        )
        if not sol.success:
            raise ArithmeticError(f"min CPR solver did not converge. {sol.message}")

        MCPR = sol.fun
        return MCPR

    def _q0(self, avg_mflux: float, hot_mflux: float, H_0_crit: float) -> float:
        gamma_f = self.reactor.gamma_f
        G = hot_mflux
        A = self.reactor.A
        P_w = self.reactor.P_w
        mdot = G * A
        shape_0_to_h0crit = (
            self.qflux.hot_heat_flux_integrated(z=H_0_crit)
        ) / self.qflux.hot_q0
        h_in = self.enthalpy.hot_enthalpy(z=0, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        h_f = self.water.liquid_water.h

        if shape_0_to_h0crit <= 0:
            return 0
        return (gamma_f * mdot * (h_f - h_in)) / (P_w * shape_0_to_h0crit)

    def _h0crit_resiudal(
        self, z: float, avg_mflux: float, hot_mflux: float, H_0_crit: float
    ) -> float:
        # units are a MESS here
        a = self._a_CISE4(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        b = self._b_CISE4(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        G = hot_mflux
        A = self.reactor.A
        P_w = self.reactor.P_w
        mdot = G * A
        L_crit = z - H_0_crit
        gamma_f = self.reactor.gamma_f

        def _x_crit() -> float:
            L_crit_metric = convert_units(L_crit, "length", "in", "m")
            x_crit = a * L_crit_metric / (L_crit_metric + b)
            # print(x_crit, a, b, L_crit_metric)
            return x_crit

        x_crit = _x_crit()

        h_in = self.enthalpy.hot_enthalpy(z=0, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        h_f = self.water.liquid_water.h
        h_g = self.water.vapor_water.h
        h_fg = h_g - h_f

        shape_h0crit_to_z_crit = (
            self.qflux.hot_heat_flux_integrated(z=(H_0_crit + L_crit))
            - self.qflux.hot_heat_flux_integrated(z=H_0_crit)
        ) / self.qflux.hot_q0
        shape_0_to_h0crit = (
            self.qflux.hot_heat_flux_integrated(z=H_0_crit)
        ) / self.qflux.hot_q0

        LHS = (x_crit * mdot * gamma_f * h_fg) * (P_w * shape_0_to_h0crit)
        RHS = (mdot * gamma_f * (h_f - h_in)) * (P_w * shape_h0crit_to_z_crit)
        return LHS - RHS

    def _a_CISE4(self, avg_mflux: float, hot_mflux: float) -> float:
        G_crit = self._G_crit
        P = self._metric_pressure
        P_c = self.critical_pressure
        G = self.mflux_to_metric(hot_mflux)
        if G < G_crit:
            return 1 / (1 + 1.481e-4 * ((1 - P / P_c) ** -3) * G)
        return (1 - P / P_c) * ((G / 1000) ** (-1 / 3))

    def _b_CISE4(self, avg_mflux: float, hot_mflux: float) -> float:
        P = self._metric_pressure
        P_c = self.critical_pressure
        G = self.mflux_to_metric(hot_mflux)
        D_e = convert_units(self.reactor.D_e, "length", "in", "m")

        return 0.199 * ((P_c / P - 1) ** (0.4)) * G * (D_e ** (1.4))

    def mflux_to_metric(self, mflux: float) -> float:
        # lbm/in^2.hr to kg/m^2.s
        converting = convert_units(mflux, "mass", "lbm", "kg")
        converting = convert_units(converting, "length", "in", "m", -2, -2)
        converting = convert_units(converting, "time", "hr", "s", -1, -1)
        return converting

    @cached_property
    def _G_crit(self) -> float:
        p_metric = self._metric_pressure
        return 3375 * ((1 - p_metric / self.critical_pressure) ** 3)

    @cached_property
    def _metric_pressure(self) -> float:
        return convert_units(self.reactor.pressure, "pressure", "psia", "MPa")

    def plot_CPR_graph(
        self,
        avg_mflux: float = None,
        hot_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        if avg_mflux is None or hot_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(
            self.enthalpy.hot_z_d(avg_mflux=avg_mflux, hot_mflux=hot_mflux),
            self.reactor.H_core,
            num_points,
        )

        cpr_z = [
            self.calculate_CPR(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, cpr_z, label=r"$CPR(z)$", color=palette.next())

        hot_z_d = self.enthalpy.hot_z_d(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        ax.axhline(
            1.2,
            color=palette.COLORS[-1],
            linestyle="--",
            linewidth=1,
            label=r"$CPR = 1.2$",
        )
        ax.axvline(
            hot_z_d,
            color=palette.COLORS[-2],
            linestyle="--",
            linewidth=1,
            label=r"$z_{d,hot}$",
        )

        ax.set_xlim(left=0)

        format_plot(
            ax,
            title="Axial CPR Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"CPR $CPR(z)$ [-]",
            grid=True,
            legend=True,
        )

        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="CPR_distribution")

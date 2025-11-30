from pathlib import Path
from functools import cache, cached_property

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad

from reactor import Reactor
from heat_flux import HeatFlux
from water import Water
from quality import Quality
from void_fraction import VoidFraction
from pp_friction_multiplier import TwoPhaseFrictionMultiplier
from pp_forms_multiplier import TwoPhaseFormsMultiplier
from critical_heat_flux import CriticalHeatFlux
from avg_flux_from_hot_flux import avg_flux_from_hot_flux
from pp_density import TwoPhaseDensity

from dimensionless_numbers import calc_reynolds
from consts import (
    NUM_PLOT_POINTS,
    FIGS_PATH,
    CRIT_FIGS_PATH,
    AVG_FIGS_PATH,
    HOT_FIGS_PATH,
)

from utilities.data.palettes import DefaultPalette, PastelPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot
from utilities.units.iapws97_imperial import ImperialIAPWS97
from utilities.units.conversions import convert_units
from utilities.units.consts import G_C_in_hr, G_C, G


class TemperatureAnalysis:
    def __init__(self, q0_crit: float, type_name: str):
        self.reactor = Reactor
        self.cqflux = CriticalHeatFlux()
        self.enthalpy = self.cqflux.enthalpy
        self.water = self.cqflux.water
        self.q0_crit = q0_crit
        self.pp_friction_factor = TwoPhaseFrictionMultiplier()
        self.type_name = type_name
        self.quality = Quality()
        self.void_fraction = VoidFraction()
        self.pp_density = TwoPhaseDensity()
        self.pp_forms_factor = TwoPhaseFormsMultiplier()

    def bulk_temperature(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        return self.cqflux.crit_water(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
        ).T

    def wall_temperature(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        if (
            self._pre_sat_wall_temp(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            < self.water.liquid_water.T
        ):
            return self._pre_sat_wall_temp(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
        return self._2p_wall_temp(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)

    def _pre_sat_wall_temp(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        h_1p = self.convective_coeff_1p(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        qflux = self.cqflux.crit_heat_flux(z=z, q0_crit=self.q0_crit)
        T_bulk = self.bulk_temperature(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        return T_bulk + qflux / h_1p

    def _2p_wall_temp(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        h_lo = self.convective_coeff_lo(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        qflux = self.cqflux.crit_heat_flux(z=z, q0_crit=self.q0_crit)
        T_bulk = self.bulk_temperature(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        T_sat = self.water.liquid_water.T

        def residual(T_wall: float) -> float:
            LHS = qflux
            RHS = h_lo * (T_wall - T_bulk) + self.convective_coeff_2p(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, T_wall=T_wall
            ) * (T_wall - T_sat)
            return LHS - RHS

        bot_limit, top_limit = T_bulk * 1.001, 704

        # print(residual(bot_limit), residual(top_limit))
        sol = root_scalar(residual, bracket=[bot_limit, top_limit], method="brentq")
        if not sol.converged:
            raise ArithmeticError(f"2p wall temp solver did not converge. {sol.flag}")
        T_wall = sol.root

        return T_wall

    def averaged_enthalpy(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        if z <= 0:
            return self.cqflux.crit_enthalpy(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )

        def integrand(z: float) -> float:
            return self.cqflux.crit_enthalpy(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )

        result, _ = quad(integrand, 0, z)
        result /= z
        return result

    @cache
    def averaged_water(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> ImperialIAPWS97:
        h_avg = self.averaged_enthalpy(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        return ImperialIAPWS97(
            P=self.reactor.pressure, h=h_avg, use_inches=True, use_hours=True
        )

    def convective_coeff_1p(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> float:
        averaged_water = self.averaged_water(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
        )
        reynolds = calc_reynolds(
            mflux=hot_mflux, D_e=self.reactor.D_e, mu=averaged_water.mu
        )
        prandtl = averaged_water.Prandt
        C = 0.042 * (self.reactor.S / self.reactor.D_o) - 0.024
        k = averaged_water.k
        return C * (k / self.reactor.D_e) * reynolds**0.8 * prandtl ** (1 / 3)

    def convective_coeff_lo(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> float:
        averaged_water = self.averaged_water(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
        )
        mu = averaged_water.mu
        cp = averaged_water.c_p
        k = averaged_water.k
        Prandtl = averaged_water.Prandt
        x = self.cqflux.crit_quality(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
        )
        D_e = self.reactor.D_e
        C1 = 0.023 * (((hot_mflux * (1 - x) * D_e) / mu) ** 0.8)
        C2 = (Prandtl) ** 0.4
        C3 = k / D_e
        F = self._get_F(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        return C1 * C2 * C3 * F

    def convective_coeff_2p(
        self, z: float, avg_mflux: float, hot_mflux: float, T_wall: float
    ) -> float:
        averaged_water = self.averaged_water(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
        )
        k_f = self.water.liquid_water.k
        c_pf = self.water.liquid_water.c_p
        rho_f = self.water.liquid_water.rho
        gc = G_C_in_hr
        sigma = averaged_water.sigma
        mu_f = self.water.liquid_water.mu
        h_fg = self.water.vapor_water.h - self.water.liquid_water.h
        rho_g = self.water.vapor_water.rho
        nu_fg = 1 / rho_g - 1 / rho_f
        T_sat = self.water.liquid_water.T

        C1 = 0.00122
        C2 = (k_f**0.79 * c_pf**0.45 * rho_f**0.49 * gc**0.25) / (
            sigma**0.5 * mu_f**0.29 * h_fg**0.24 * rho_g**0.24
        )
        C3 = (T_wall - T_sat) ** 0.24
        C4 = (
            ImperialIAPWS97(T=T_wall, x=0).P - ImperialIAPWS97(T=T_sat, x=0).P
        ) ** 0.75
        S = self._get_S(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        return C1 * C2 * C3 * C4 * S

    def _get_F(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        chi = self.pp_friction_factor.chi(
            x=self.cqflux.crit_quality(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )
        )

        if 1 / chi <= 0.1:
            return 1
        return 2.35 * ((1 / chi + 0.213) ** 0.736)

    def _get_J(self, T_wall: float) -> float:
        c_pf = self.water.liquid_water.c_p
        h_fg = self.water.vapor_water.h - self.water.liquid_water.h
        T_sat = self.water.liquid_water.T

        return c_pf * (T_wall - T_sat) / h_fg

    def _get_S(self, z: float, avg_mflux: float, hot_mflux: float) -> float:

        mu = self.averaged_water(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux).mu
        reynolds_1p = calc_reynolds(
            mflux=(
                hot_mflux
                * (
                    1
                    - self.cqflux.crit_quality(
                        z=z,
                        avg_mflux=avg_mflux,
                        hot_mflux=hot_mflux,
                        q0_crit=self.q0_crit,
                    )
                )
            ),
            D_e=self.reactor.D_e,
            mu=mu,
        )
        F = self._get_F(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        reynolds_2p = reynolds_1p * (F ** (10 / 8))
        return 1 / (1 + 2.53e-6 * (reynolds_2p**1.17))
        return 0.9622 - 0.5822 * np.arctan(reynolds_2p / (6.18e4))

    def get_z_wall_sat(self, avg_mflux: float, hot_mflux: float) -> float:

        def residual(z: float) -> float:
            return self.water.liquid_water.T - self.wall_temperature(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )

        sol = root_scalar(residual, bracket=[0, 25], method="brentq")
        if not sol.converged:
            raise ArithmeticError(f"z_wall,sat solver did not converge. {sol.flag}")
        z_wall_sat = sol.root

        return z_wall_sat

    def fuel_outer_temperature(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> float:
        T_cladding = self.wall_temperature(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
        )
        qflux = self.cqflux.crit_heat_flux(z=z, q0_crit=self.q0_crit)
        D_o = self.reactor.D_o
        D_i = self.reactor.D_i
        H_G = self.reactor.H_G
        k_c = self.reactor.k_clad
        R_tot = ((D_o / 2) / ((D_i / 2) * H_G)) + (
            (D_o / 2 * np.log(D_o / D_i)) / (k_c)
        )
        # print(f"qflux: {qflux}, R_tot: {R_tot}")

        return T_cladding + qflux * R_tot

    def fuel_centerline_temperature(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> float:
        minimum = self.fuel_outer_temperature(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
        )
        maximum = self.reactor.T_melt * 10

        def residual(T: float) -> float:
            return self._fuel_centerline_temperature_residual(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, T=T
            )

        sol = root_scalar(residual, bracket=[minimum, maximum], method="brentq")
        if not sol.converged:
            raise ArithmeticError(f"z_wall,sat solver did not converge. {sol.flag}")
        T_fi = sol.root

        return T_fi

    def _fuel_centerline_temperature_residual(
        self, z: float, avg_mflux: float, hot_mflux: float, T: float
    ) -> float:
        qflux = (
            self.cqflux.crit_heat_flux(z=z, q0_crit=self.q0_crit) * 144
        )  # put in ft^2
        r_o = self.reactor.D_o / 2 / 12
        LHS = self._conductivity_integral(T)
        RHS = (
            self._conductivity_integral(
                self.fuel_outer_temperature(
                    z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
                )
            )
            + (qflux * r_o) / 2
        )
        return LHS - RHS

    def _conductivity_integral(self, T: float) -> float:
        return 3978.1 * np.log((692.6 + T) / 692.6) + ((6.02366 * 1e-12) / 4) * (
            (T + 460) ** 4 - 460**4
        )

    def max_fuel_centerline_temperature(
        self, avg_mflux: float, hot_mflux: float, z_not_temp: bool = False
    ) -> float:
        def residual(z: float) -> float:
            z = float(z)
            return -1 * self.fuel_centerline_temperature(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )

        sol = minimize(
            residual, x0=[self.reactor.H_core * 0.25], bounds=[(0, self.reactor.H_core)]
        )
        if not sol.success:
            raise ArithmeticError(f"max T_fi solver did not converge. {sol.message}")
        if not z_not_temp:
            T_fi_max = sol.fun
            return T_fi_max * -1
        z_max = sol.x
        return z_max

    def calculate_mflux_for_meltdown(
        self, avg_mflux: float, hot_mflux_guess: float
    ) -> float:
        def residual(hot_mflux: float) -> float:
            _avg_mflux = avg_flux_from_hot_flux(
                avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            LHS = self.reactor.T_melt
            RHS = self.max_fuel_centerline_temperature(
                avg_mflux=_avg_mflux, hot_mflux=hot_mflux
            )
            print(LHS - RHS)
            return LHS - RHS

        bot_bound = 4e3
        top_bound = 1e6

        print(residual(bot_bound), residual(hot_mflux_guess), residual(top_bound))
        sol = root_scalar(residual, bracket=[bot_bound, top_bound], method="brentq")
        if not sol.converged:
            raise ArithmeticError(f"1.2 CPR solver did not converge. {sol.flag}")
        hot_mflux = sol.root
        return hot_mflux

    def crit_void_fraction(self, z: float, avg_mflux: float, hot_mflux: float) -> float:
        x = self.cqflux.crit_quality(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
        )
        water = self.cqflux.crit_water(
            z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
        )
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

    def plot_bulk_temp(
        self,
        avg_mflux: float = None,
        hot_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        if avg_mflux is None or hot_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        T_bulk = [
            self.bulk_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, T_bulk, label=r"$T_{\infty}(z)$", color=palette.next())

        T_sat = self.water.liquid_water.T
        ax.axhline(
            T_sat,
            color=palette.COLORS[-1],
            linestyle="--",
            linewidth=1,
            label=r"$T_{sat}$",
        )

        format_plot(
            ax,
            title=f"Axial Bulk Temperature Distribution\nWith {self.type_name} Heat Flux",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Bulk Temperature $T_{\infty}(z)$ [°F]",
            grid=True,
            legend=True,
        )

        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="T_bulk_crit_distribution")

    def plot_wall_temp(
        self,
        avg_mflux: float = None,
        hot_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        if avg_mflux is None or hot_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        T_bulk = [
            self.wall_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, T_bulk, label=r"$T_{co}(z)$", color=palette.next())

        T_sat = self.water.liquid_water.T

        ax.axhline(
            T_sat,
            color=palette.COLORS[-1],
            linestyle="--",
            linewidth=1,
            label=r"$T_{sat}$",
        )

        """
        z_wall_sat = self.get_z_wall_sat(mflux=mflux)

        ax.axvline(
            z_wall_sat,
            color=palette.COLORS[-1],
            linestyle="--",
            linewidth=1,
            label=r"$z_{wall,sat}$",
        )
        """

        format_plot(
            ax,
            title=f"Axial Wall Temperature Distribution\nWith {self.type_name} Heat Flux",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Wall Temperature $T_{co}(z)$ [°F]",
            grid=True,
            legend=True,
        )

        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="T_wall_crit_distribution")

    def plot_outer_fuel_temp(
        self,
        avg_mflux: float = None,
        hot_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        if avg_mflux is None or hot_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        T_bulk = [
            self.fuel_outer_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, T_bulk, label=r"$T_{fo}(z)$", color=palette.next())

        format_plot(
            ax,
            title=f"Axial Outer Fuel Temperature Distribution\nWith {self.type_name} Heat Flux",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Outer Fuel Temperature $T_{fo}(z)$ [°F]",
            grid=True,
            legend=False,
        )

        save_plot(fig, folder=results_dir, name="T_fo_crit_distribution")

    def plot_fuel_centerline_temp(
        self,
        avg_mflux: float = None,
        hot_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        if avg_mflux is None or hot_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        T_bulk = [
            self.fuel_centerline_temperature(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, T_bulk, label=r"$T_{fi}(z)$", color=palette.next())

        T_max = self.max_fuel_centerline_temperature(
            avg_mflux=avg_mflux, hot_mflux=hot_mflux, z_not_temp=True
        )
        T_sat = self.reactor.T_melt

        ax.axvline(
            T_max,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$z_{max}$",
        )

        ax.axhline(
            T_sat,
            color=palette.COLORS[-1],
            linestyle="--",
            linewidth=1,
            label=r"$T_{melt}$",
        )

        format_plot(
            ax,
            title=f"Axial Fuel Centerline Temperature Distribution\nWith {self.type_name} Heat Flux",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Fuel Centerline Temperature $T_{fi}(z)$ [°F]",
            grid=True,
            legend=True,
        )

        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="T_fi_crit_distribution")

    def crit_z_d(self, avg_mflux: float, hot_mflux: float) -> float:
        _LHS = self.water.liquid_water.h

        def residual(z: float) -> float:
            LHS = _LHS
            RHS = self.cqflux.crit_enthalpy(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )
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

    def crit_mixture_density(
        self, z: float, avg_mflux: float, hot_mflux: float
    ) -> float:
        z = min(z, self.reactor.H_core)
        alpha_g = self.crit_void_fraction(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        if alpha_g <= 0:
            return self.cqflux.crit_water(
                z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            ).rho
        return self._mixture_density(alpha_g=alpha_g)

    def _mixture_density(self, alpha_g: float) -> float:
        alpha_l = 1 - alpha_g
        return (
            self.water.liquid_water.rho * alpha_l + self.water.vapor_water.rho * alpha_g
        )

    def plot_all_temp_dists(
        self,
        avg_mflux: float = None,
        hot_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = FIGS_PATH,
    ) -> None:
        if avg_mflux is None or hot_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        T_bulk = [
            self.bulk_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        T_wall = [
            self.wall_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        T_fo = [
            self.fuel_outer_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        T_fi = [
            self.fuel_centerline_temperature(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, T_bulk, label=r"$T_{\infty}(z)$", color=palette.next())
        ax.plot(z, T_wall, label=r"$T_{co}(z)$", color=palette.next())
        ax.plot(z, T_fo, label=r"$T_{fo}(z)$", color=palette.next())
        ax.plot(z, T_fi, label=r"$T_{fi}(z)$", color=palette.next())

        format_plot(
            ax,
            title=f"Axial Temperature Distributions\nWith {self.type_name} Heat Flux",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Temperature $T_(z)$ [°F]",
            grid=True,
            legend=True,
        )

        save_plot(fig, folder=results_dir, name="temperature_distributions")

    def plot_all_enthalpys(
        self,
        hot_mflux: float = None,
        avg_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ) -> None:
        if hot_mflux is None or avg_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        hcrit_z = [
            self.cqflux.crit_enthalpy(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )
            for zi in z
        ]
        hhot_z = [
            self.enthalpy.hot_enthalpy(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        havg_z = [self.enthalpy.avg_enthalpy(z=zi, avg_mflux=avg_mflux) for zi in z]

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(z, hcrit_z, label=r"$h_{crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, hhot_z, label=r"$h_{hot}(z)$", color=palette.next())
        ax.plot(z, havg_z, label=r"$h_{avg}(z)$", color=palette.next())

        crit_z_d = self.crit_z_d(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        hot_z_d = self.enthalpy.hot_z_d(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
        avg_z_d = self.enthalpy.avg_z_d(avg_mflux=avg_mflux, hot_mflux=avg_mflux)

        if crit_z_d is not None:
            ax.axvline(
                crit_z_d,
                color=palette.COLORS[-2],
                linestyle="--",
                linewidth=1,
                label=r"$z_{d,crit}$",
            )

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
            self.water.liquid_water.h,
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

        ax.get_lines()[-4].set_linewidth(1)
        ax.get_lines()[-3].set_linewidth(1)
        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="enthalpy_distributions")

    def plot_all_qualities(
        self,
        hot_mflux: float = None,
        avg_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ) -> None:
        if hot_mflux is None or avg_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        xcrit_z = [
            self.cqflux.crit_quality(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )
            for zi in z
        ]

        xhot_z = [
            self.quality.hot_quality(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        xavg_z = [self.quality.avg_quality(z=zi, avg_mflux=avg_mflux) for zi in z]

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(z, xcrit_z, label=r"$x_{crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, xhot_z, label=r"$x_{hot}(z)$", color=palette.next())
        ax.plot(z, xavg_z, label=r"$x_{avg}(z)$", color=palette.next())

        crit_z_d = self.crit_z_d(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        hot_z_d = self.enthalpy.hot_z_d(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
        avg_z_d = self.enthalpy.avg_z_d(avg_mflux=avg_mflux, hot_mflux=avg_mflux)

        if crit_z_d is not None:
            ax.axvline(
                crit_z_d,
                color=palette.COLORS[-2],
                linestyle="--",
                linewidth=1,
                label=r"$z_{d,crit}$",
            )

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

        ax.get_lines()[-3].set_linewidth(1)
        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="quality_distributions")

    def plot_all_void_fractions(
        self,
        hot_mflux: float = None,
        avg_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ) -> None:
        if hot_mflux is None or avg_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        alphacrit_z = [
            self.crit_void_fraction(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]

        alphahot_z = [
            self.void_fraction.hot_void_fraction(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            for zi in z
        ]
        alphaavg_z = [
            self.void_fraction.avg_void_fraction(z=zi, avg_mflux=avg_mflux) for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, alphacrit_z, label=r"$\alpha_{crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, alphahot_z, label=r"$\alpha_{hot}(z)$", color=palette.next())
        ax.plot(z, alphaavg_z, label=r"$\alpha_{avg}(z)$", color=palette.next())

        crit_z_d = self.crit_z_d(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        hot_z_d = self.enthalpy.hot_z_d(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
        avg_z_d = self.enthalpy.avg_z_d(avg_mflux=avg_mflux, hot_mflux=avg_mflux)

        if crit_z_d is not None:
            ax.axvline(
                crit_z_d,
                color=palette.COLORS[-2],
                linestyle="--",
                linewidth=1,
                label=r"$z_{d,crit}$",
            )

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

        ax.get_lines()[-3].set_linewidth(1)
        if avg_z_d is not None and hot_z_d is not None:
            ax.get_lines()[-2].set_linewidth(1)
        if avg_z_d is not None or hot_z_d is not None:
            ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="void_fractions_distributions")

    def plot_all_densities(
        self,
        hot_mflux: float = None,
        avg_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ) -> None:
        if hot_mflux is None or avg_mflux is None:
            raise ValueError("mflux missing!")

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        rho_crit_z = [
            self.crit_mixture_density(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        rho_hot_z = [
            self.pp_density.hot_mixture_density(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            for zi in z
        ]
        rho_avg_z = [
            self.pp_density.avg_mixture_density(z=zi, avg_mflux=avg_mflux) for zi in z
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, rho_crit_z, label=r"$\rho_{crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, rho_hot_z, label=r"$\rho_{hot}(z)$", color=palette.next())
        ax.plot(z, rho_avg_z, label=r"$\rho_{avg}(z)$", color=palette.next())

        palette.next()

        crit_z_d = self.crit_z_d(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        hot_z_d = self.enthalpy.hot_z_d(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
        avg_z_d = self.enthalpy.avg_z_d(avg_mflux=avg_mflux, hot_mflux=avg_mflux)

        if crit_z_d is not None:
            ax.axvline(
                crit_z_d,
                color=palette.COLORS[-2],
                linestyle="--",
                linewidth=1,
                label=r"$z_{d,crit}$",
            )

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

        ax.get_lines()[-5].set_linewidth(1)
        ax.get_lines()[-4].set_linewidth(1)
        ax.get_lines()[-3].set_linewidth(1)
        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="mixture_density_distribution")

    def plot_all_heat_fluxes(
        self,
        hot_mflux: float = None,
        avg_mflux: float = None,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ) -> None:

        palette = DefaultPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        qcrit_z = [self.cqflux.crit_heat_flux(z=zi, q0_crit=self.q0_crit) for zi in z]
        qhot_z = [self.enthalpy.qflux.hot_heat_flux(zi) for zi in z]
        qavg_z = [self.enthalpy.qflux.avg_heat_flux(zi) for zi in z]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, qcrit_z, label=r"$q''_{crit}(z)$", color=palette.COLORS[-2])
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

    def plot_philos(
        self,
        avg_mflux: float,
        hot_mflux: float,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ):

        palette = DefaultPalette()

        # i doubt the performance gains from turning this part into a cached property will matter but...
        # eh... i might as well. any bit helps, right? right?? please tell me im right
        @cache
        def _chi_static() -> float:
            mu_f = self.water.liquid_water.mu
            mu_g = self.water.vapor_water.mu
            rho_f = self.water.liquid_water.rho
            rho_g = self.water.vapor_water.rho

            return ((mu_f / mu_g) ** 0.2) * (rho_g / rho_f)

        def _chi(x: float) -> float:
            if x <= 0:
                return None
            return np.sqrt(_chi_static() * (((1 - x) / x) ** 1.8))

        def discrete_2p_friction_multiplier(x: float) -> float:
            if x <= 0:
                return 1
            chi = _chi(x=x)
            return (1 + 20 / chi + 1 / chi**2) * ((1 - x) ** 1.8)

        z = np.linspace(0, self.reactor.H_core, num_points)

        xcrit_z = [
            self.cqflux.crit_quality(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )
            for zi in z
        ]

        xhot_z = [
            self.quality.hot_quality(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        xavg_z = [self.quality.avg_quality(z=zi, avg_mflux=avg_mflux) for zi in z]

        crit_philo = [discrete_2p_friction_multiplier(xi) for xi in xcrit_z]
        hot_philo = [discrete_2p_friction_multiplier(xi) for xi in xhot_z]
        avg_philo = [discrete_2p_friction_multiplier(xi) for xi in xavg_z]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, crit_philo, label=r"$\phi^2_{lo,crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, hot_philo, label=r"$\phi^2_{lo,hot}(z)$", color=palette.next())
        ax.plot(z, avg_philo, label=r"$\phi^2_{lo,avg}(z)$", color=palette.next())

        format_plot(
            ax,
            title="Two Phase Friction Multiplier Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Two Phase Friction Multiplier $\phi^2_{lo}(z)$ [-]",
            grid=True,
            legend=True,
        )

        save_plot(fig, folder=results_dir, name="philo_distributions")

    def plot_forms_factor(
        self,
        avg_mflux: float,
        hot_mflux: float,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ):

        palette = DefaultPalette()
        pp_forms = TwoPhaseFormsMultiplier()

        z = np.linspace(0, self.reactor.H_core, num_points)

        xcrit_z = [
            self.cqflux.crit_quality(
                z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
            )
            for zi in z
        ]

        xhot_z = [
            self.quality.hot_quality(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]
        xavg_z = [self.quality.avg_quality(z=zi, avg_mflux=avg_mflux) for zi in z]

        crit_psi = [pp_forms.discrete_2p_forms_multiplier(x=xi) for xi in xcrit_z]
        hot_psi = [pp_forms.discrete_2p_forms_multiplier(x=xi) for xi in xhot_z]
        avg_psi = [pp_forms.discrete_2p_forms_multiplier(x=xi) for xi in xavg_z]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, crit_psi, label=r"$\Psi_{crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, hot_psi, label=r"$\Psi_{hot}(z)$", color=palette.next())
        ax.plot(z, avg_psi, label=r"$\Psi_{avg}(z)$", color=palette.next())

        format_plot(
            ax,
            title="Two Phase Forms Multiplier Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Two Phase Friction Multiplier $\Psi^2_{lo}(z)$ [-]",
            grid=True,
            legend=True,
        )

        save_plot(fig, folder=results_dir, name="psi_distributions")

    def plot_twall_tbulk_and_chi(
        self,
        avg_mflux: float,
        hot_mflux: float,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = CRIT_FIGS_PATH,
    ) -> None:

        palette = PastelPalette()

        z = np.linspace(0, self.reactor.H_core, num_points)

        T_bulk = [
            self.bulk_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]

        T_wall = [
            self.wall_temperature(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            for zi in z
        ]

        @cache
        def _chi_static() -> float:
            mu_f = self.water.liquid_water.mu
            mu_g = self.water.vapor_water.mu
            rho_f = self.water.liquid_water.rho
            rho_g = self.water.vapor_water.rho

            return ((mu_f / mu_g) ** 0.2) * (rho_g / rho_f)

        def _chi(x: float) -> float:
            if x <= 0:
                return None
            return np.sqrt(_chi_static() * (((1 - x) / x) ** 1.8))

        if results_dir == CRIT_FIGS_PATH:
            x_z = [
                self.cqflux.crit_quality(
                    z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux, q0_crit=self.q0_crit
                )
                for zi in z
            ]
        elif results_dir == HOT_FIGS_PATH:
            x_z = [
                self.quality.hot_quality(z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
                for zi in z
            ]

        else:
            x_z = [self.quality.avg_quality(z=zi, avg_mflux=avg_mflux) for zi in z]

        chi_z = [_chi(xi) for xi in x_z]
        chi_z = [1 / v if v is not None else None for v in chi_z]

        def find_tsat_z():
            T_sat = self.water.liquid_water.T

            # First find where residual changes sign
            prev_res = T_sat - self.bulk_temperature(
                z=0, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            for i in range(1, num_points):
                zi = z[i]
                res = T_sat - self.bulk_temperature(
                    z=zi, avg_mflux=avg_mflux, hot_mflux=hot_mflux
                )

                if np.sign(prev_res) != np.sign(res):
                    # Found the first bracket that contains the root
                    a = z[i - 1]
                    b = z[i]
                    sol = root_scalar(
                        lambda zz: T_sat
                        - self.bulk_temperature(
                            z=zz, avg_mflux=avg_mflux, hot_mflux=hot_mflux
                        ),
                        bracket=[a, b],
                        method="brentq",
                    )
                    return sol.root

                prev_res = res

            raise RuntimeError("No saturation crossing found")

        tsat_z = find_tsat_z()

        def find_chi_0dot1() -> float:
            def residual(z: float) -> float:
                LHS = 0.1
                if results_dir == CRIT_FIGS_PATH:
                    _RHS = _chi(
                        x=self.cqflux.crit_quality(
                            z=z,
                            avg_mflux=avg_mflux,
                            hot_mflux=hot_mflux,
                            q0_crit=self.q0_crit,
                        )
                    )
                elif results_dir == HOT_FIGS_PATH:
                    _RHS = _chi(
                        x=self.quality.hot_quality(
                            z=z,
                            avg_mflux=avg_mflux,
                            hot_mflux=hot_mflux,
                        )
                    )
                else:
                    _RHS = _chi(
                        x=self.quality.avg_quality(
                            z=z,
                            avg_mflux=avg_mflux,
                        )
                    )

                if _RHS is not None:
                    RHS = 1 / _RHS
                    return LHS - RHS
                RHS = 0
                return LHS - RHS

            sol = root_scalar(
                residual, bracket=[0, self.reactor.H_core], method="brentq"
            )
            return sol.root

        z_chi_0dot1 = find_chi_0dot1()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, T_wall, label=r"$T_{wall}(z)$", color=palette.next(), linewidth=1)
        ax.plot(z, T_bulk, label=r"$T_{\infty}(z)$", color=palette.next(), linewidth=1)

        T_sat = self.water.liquid_water.T
        ax.axhline(
            T_sat,
            color=palette.COLORS[-1],
            linestyle="--",
            linewidth=1,
            label=r"$T_{sat}$",
        )

        ax.axvline(
            tsat_z,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$z_{T_{sat}}$",
        )

        ax_chi = ax.twinx()
        ax_chi.set_yscale("log")
        chi_color = palette.next()
        ax_chi.spines["right"].set_color(chi_color)
        ax_chi.tick_params(
            axis="y", colors=chi_color, which="major", labelsize=12, direction="out"
        )
        ax_chi.tick_params(axis="y", colors=chi_color, which="minor", direction="out")
        ax_chi.minorticks_on()

        ax_chi.yaxis.label.set_color(chi_color)
        ax_chi.plot(z, chi_z, label=r"$1/\chi(z)$", color=chi_color, linewidth=1)
        ax_chi.axvline(
            z_chi_0dot1,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$z_{1/\chi = 10}$",
        )
        ax_chi.axhline(
            1 / 10,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$1/\chi = 10$",
        )

        format_plot(
            ax,
            title=(
                "Axial Bulk Temperature and Wall Temperature Distributions and "
                + r"$1/\chi$"
                + f" \nWith {self.type_name} Heat Flux"
            ),
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Temperature $T(z)$ [°F]",
            grid=False,
            legend=False,
        )

        format_plot(
            ax_chi,
            title=None,
            xlabel=None,
            ylabel=r"Inverse of Martinelli Parameter $1/\chi(z)$ [-]",
            grid=False,
            legend=False,
        )

        rcParams["font.family"] = "serif"
        rcParams["font.serif"] = ["Times New Roman"]

        # ax_chi.set_ylabel(r"Martinelli Parameter $\chi(z)$", fontsize=12, fontname="Times New Roman")

        for line in ax.get_lines():
            line.set_linewidth(1)

        for line in ax_chi.get_lines():
            line.set_linewidth(1)

        # --- Build combined legend from both axes ---
        lines = ax.get_lines() + ax_chi.get_lines()
        labels = [line.get_label() for line in lines]

        # Place legend on the left axis (or anywhere you prefer)
        ax.legend(
            lines,
            labels,
            fontsize=12,
            loc="upper right",
            ncol=1,
            frameon=True,
            framealpha=0.75,
        )
        ax_chi.legend(
            lines,
            labels,
            fontsize=12,
            loc="upper right",
            ncol=1,
            frameon=True,
            framealpha=0.75,
        )

        for line in ax.get_lines():
            line.set_linewidth(2.5)

        for line in ax_chi.get_lines():
            line.set_linewidth(2.5)

        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)
        ax_chi.get_lines()[-2].set_linewidth(1)
        ax_chi.get_lines()[-1].set_linewidth(1)

        ax.set_zorder(10)
        ax.patch.set_visible(False)
        ax_chi.set_zorder(1)

        save_plot(fig, folder=results_dir, name="T_wall_T_bulk_chi_distribution")

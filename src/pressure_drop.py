from pathlib import Path
from functools import cache, cached_property

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import quad

from reactor import Reactor
from enthalpy import Enthalpy
from water import Water
from quality import Quality
from pp_density import TwoPhaseDensity
from pp_friction_multiplier import TwoPhaseFrictionMultiplier
from pp_forms_multiplier import TwoPhaseFormsMultiplier
from void_fraction import VoidFraction
from friction_factor import FrictionFactor
from dimensionless_numbers import calc_reynolds

from consts import NUM_PLOT_POINTS, FIGS_PATH

from utilities.data.palettes import DefaultPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot

from utilities.units.consts import G, G_C_in_hr
from utilities.units.iapws97_imperial import ImperialIAPWS97


class PressureDrop:
    # find pressure drop as a function of mass flux
    def __init__(self):
        self.void_fraction = VoidFraction()
        self.water = self.void_fraction.water
        self.enthalpy = self.void_fraction.enthalpy
        self.reactor = self.void_fraction.reactor
        self.quality = self.void_fraction.quality
        self.pp_friction_mult = TwoPhaseFrictionMultiplier()
        self.pp_forms_mult = TwoPhaseFormsMultiplier()
        self.pp_density = TwoPhaseDensity()
        self.friction_factor = FrictionFactor()

    def avg_args(self, avg_mflux: float, please_print: bool = False) -> dict:
        return {
            "avg_mflux": avg_mflux,
            "hot_mflux": avg_mflux,
            "please_print": please_print,
            "quality_func": self.quality.avg_quality,
            "void_fraction_func": self.void_fraction.avg_void_fraction,
            "z_d_func": self.enthalpy.avg_z_d,
            "water_func": self.water.avg_water,
            "pp_density_func": self.pp_density.avg_mixture_density,
        }

    def hot_args(self, avg_mflux: float, hot_mflux, please_print: bool = False) -> dict:
        return {
            "avg_mflux": avg_mflux,
            "hot_mflux": hot_mflux,
            "please_print": please_print,
            "quality_func": self.quality.hot_quality,
            "void_fraction_func": self.void_fraction.hot_void_fraction,
            "z_d_func": self.enthalpy.hot_z_d,
            "water_func": self.water.hot_water,
            "pp_density_func": self.pp_density.hot_mixture_density,
        }

    def find_avg_mass_flux(self, deltaP: float) -> float:

        def residual(avg_mflux: float) -> float:
            LHS = deltaP
            RHS = self.avg_total_losses(avg_mflux=avg_mflux) if avg_mflux > 0 else 0
            return LHS - RHS

        return self._find_mass_flux(residual=residual)

    def find_hot_mass_flux(self, avg_mflux: float, deltaP: float) -> float:
        def residual(hot_mflux: float) -> float:
            LHS = deltaP
            # print(mflux)
            RHS = (
                self.hot_total_losses(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
                if hot_mflux > 0
                else 0
            )
            return LHS - RHS

        return self._find_mass_flux(residual=residual)

    def find_mass_flux_from_core_deltaP(self, deltaP: float) -> float:
        def residual(avg_mflux: float) -> float:
            LHS = deltaP
            RHS = self.avg_core_losses(avg_mflux=avg_mflux) if avg_mflux > 0 else 0
            return LHS - RHS

        return self._find_mass_flux(residual=residual)

    def _find_mass_flux(self, residual: callable) -> float:
        bot_bound = 5e3
        top_bound = 5e4

        # print("!!!")
        # print(residual(bot_bound), residual(top_bound))

        sol = root_scalar(residual, bracket=[bot_bound, top_bound], method="brentq")

        if not sol.converged:
            raise ArithmeticError(f"mflux solver did not converge. {sol.flag}")
        return sol.root

    def avg_total_losses(self, avg_mflux: float, please_print: bool = False) -> float:
        core_losses = self.avg_core_losses(
            avg_mflux=avg_mflux, please_print=please_print
        )
        chimney_losses = self.avg_chimney_losses(
            avg_mflux=avg_mflux, please_print=please_print
        )
        downcomer_losses = self.avg_downcomer_losses(
            avg_mflux=avg_mflux, please_print=please_print
        )

        total_losses = core_losses + chimney_losses + downcomer_losses
        return total_losses

    def hot_total_losses(self, avg_mflux: float, hot_mflux: float) -> float:
        core_losses = self.hot_core_losses(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        chimney_losses = self.avg_chimney_losses(avg_mflux=avg_mflux)
        downcomer_losses = self.avg_downcomer_losses(avg_mflux=avg_mflux)
        # ^^^ these two are avg because theyre still getting the avg water out of the core...

        total_losses = core_losses + chimney_losses + downcomer_losses
        return total_losses

    def avg_chimney_losses(self, avg_mflux: float, please_print: bool = False) -> float:
        return self._chimney_losses(
            **self.avg_args(avg_mflux=avg_mflux, please_print=please_print)
        )

    def avg_downcomer_losses(
        self, avg_mflux: float, please_print: bool = False
    ) -> float:
        return self._downcomer_losses(
            **self.avg_args(avg_mflux=avg_mflux, please_print=please_print)
        )

    def avg_core_losses(self, avg_mflux: float, please_print: bool = False) -> float:
        return self._core_losses(
            **self.avg_args(avg_mflux=avg_mflux, please_print=please_print)
        )

    def hot_core_losses(
        self, avg_mflux: float, hot_mflux: float, please_print: bool = False
    ) -> float:
        return self._core_losses(
            **self.hot_args(
                avg_mflux=avg_mflux, hot_mflux=hot_mflux, please_print=please_print
            )
        )

    def _core_losses(
        self,
        avg_mflux: float,
        hot_mflux: float,
        please_print: bool,
        quality_func: callable,
        void_fraction_func: callable,
        z_d_func: callable,
        water_func: callable,
        pp_density_func: callable,
    ) -> float:
        H_core = self.reactor.H_core
        z_d = z_d_func(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
        rho_f = self.water.liquid_water.rho
        rho_l = rho_f  # equilibrium
        rho_g = self.water.vapor_water.rho
        if z_d is None or z_d > H_core:
            z_d = None
        D_e = self.reactor.D_e

        def _accel_losses() -> float:
            x = quality_func(z=H_core, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
            alpha_g = void_fraction_func(
                z=H_core, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            alpha_l = 1 - alpha_g
            # print(f"x: {x}, alpha_g: {alpha_g}")
            ans = (
                (hot_mflux**2)
                / (G_C_in_hr)
                * (
                    ((1 - x) ** 2 / (alpha_l * rho_l) + (x) ** 2 / (alpha_g * rho_g))
                    - 1 / pp_density_func(z=0, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
                )
            )
            return ans

        def _friction_losses() -> float:
            def _p_friction_losses(z: float) -> float:
                reynolds = calc_reynolds(
                    mflux=hot_mflux,
                    D_e=D_e,
                    mu=water_func(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux).mu,
                )
                f = self.friction_factor.smooth_friction_factor(reynolds=reynolds)
                rho = water_func(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux).rho
                return f / rho

            def _pp_friction_losses(z: float) -> float:
                reynolds = calc_reynolds(
                    mflux=hot_mflux,
                    D_e=D_e,
                    mu=water_func(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux).mu,
                )
                f = self.friction_factor.smooth_friction_factor(reynolds=reynolds)
                rho = water_func(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux).rho
                pp_mult = self.pp_friction_mult.discrete_2p_friction_multiplier(
                    x=quality_func(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
                )
                if pp_mult > 100:
                    print(pp_mult)
                return (f / rho) * pp_mult

            if z_d is not None:
                p_friction_losses, _ = quad(_p_friction_losses, a=0, b=z_d)
                pp_friction_losses, _ = quad(_pp_friction_losses, a=z_d, b=H_core)
            if z_d <= 0:
                p_friction_losses = 0
                pp_friction_losses, _ = quad(_pp_friction_losses, a=0, b=H_core)
            else:
                p_friction_losses, _ = quad(_p_friction_losses, a=0, b=H_core)
                pp_friction_losses = 0
            return (hot_mflux**2 / (2 * G_C_in_hr * D_e)) * (
                p_friction_losses + pp_friction_losses
            )

        def _forms_losses() -> float:
            num_grids = self.reactor.num_grids
            grid_gap = H_core / (num_grids - 1)
            if z_d is None or z_d <= 0:
                num_grids_1p = 1
            elif z_d >= H_core:
                num_grids_1p = num_grids
            else:
                num_grids_1p = int(z_d // grid_gap + 1)
            num_grids_2p = num_grids - num_grids_1p

            loss_1p = self.reactor.loss_inlet / pp_density_func(
                z=0, avg_mflux=avg_mflux, hot_mflux=hot_mflux
            )
            for i in range(num_grids_1p):
                grid_idx = i
                z_i = grid_gap * grid_idx
                rho_i = water_func(z=z_i, avg_mflux=avg_mflux, hot_mflux=hot_mflux).rho
                loss_1p += self.reactor.loss_grid / rho_i

            loss_2p = (
                self.reactor.loss_outlet
                * self.pp_forms_mult.discrete_2p_forms_multiplier(
                    quality_func(z=H_core, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
                )
                / water_func(z=H_core, avg_mflux=avg_mflux, hot_mflux=hot_mflux).rho
            )

            if num_grids_2p > 0:
                for i in range(num_grids_2p):
                    grid_idx = num_grids_1p + i
                    z_i = grid_gap * grid_idx
                    rho_i = water_func(
                        z=z_i, avg_mflux=avg_mflux, hot_mflux=hot_mflux
                    ).rho
                    psi_i = self.pp_forms_mult.discrete_2p_forms_multiplier(
                        quality_func(z=z_i, avg_mflux=avg_mflux, hot_mflux=hot_mflux)
                    )
                    loss_2p += self.reactor.loss_grid * psi_i / rho_i

            losses = (hot_mflux**2 / (2 * G_C_in_hr)) * (loss_1p + loss_2p)

            return losses

        def _elevation_losses() -> float:
            def _integral(z: float) -> float:
                return pp_density_func(z=z, avg_mflux=avg_mflux, hot_mflux=hot_mflux)

            results, _ = quad(_integral, a=0, b=H_core)
            return results

        accel_losses = _accel_losses()
        friction_losses = _friction_losses()
        forms_losses = _forms_losses()
        elevation_losses = _elevation_losses()

        total_losses = accel_losses + friction_losses + forms_losses + elevation_losses
        if please_print:
            print(
                f"core losses: {accel_losses, friction_losses, forms_losses, elevation_losses}"
            )
        return total_losses

    def _chimney_losses(
        self,
        avg_mflux: float,
        hot_mflux: float,
        please_print: bool,
        quality_func: callable,
        void_fraction_func: callable,
        z_d_func: callable,
        water_func: callable,
        pp_density_func: callable,
    ) -> float:
        avg_mflux = (
            avg_mflux * self.reactor.A * self.reactor.num_rods / self.reactor.A_ch
        )
        hot_mflux = (
            hot_mflux * self.reactor.A * self.reactor.num_rods / self.reactor.A_ch
        )
        DeltaH = self.reactor.H_chimney
        H_chimney = self.reactor.H_chimney + self.reactor.H_core
        H_core = self.reactor.H_core
        z_d = z_d_func(avg_mflux=avg_mflux, hot_mflux=avg_mflux)
        rho_f = self.water.liquid_water.rho
        rho_l = rho_f  # equilibrium
        rho_g = self.water.vapor_water.rho
        if z_d is None or z_d > H_core:
            z_d = None
        D_e = self.reactor.D_e_ch

        def _accel_losses() -> float:
            return 0

        def _friction_losses() -> float:
            reynolds = calc_reynolds(
                mflux=avg_mflux,
                D_e=D_e,
                mu=water_func(z=H_core, avg_mflux=avg_mflux, hot_mflux=avg_mflux).mu,
            )
            f = self.friction_factor.rough_friction_factor(reynolds=reynolds, D_e=D_e)
            rho = water_func(z=H_core, avg_mflux=avg_mflux, hot_mflux=avg_mflux).rho
            pp_mult = self.pp_friction_mult.discrete_2p_friction_multiplier(
                x=quality_func(z=H_core, avg_mflux=avg_mflux, hot_mflux=avg_mflux)
            )

            pp_friction_losses = (f / rho) * pp_mult * DeltaH
            return (avg_mflux**2 / (2 * G_C_in_hr * D_e)) * (pp_friction_losses)

        def _forms_losses() -> float:
            return 0

        def _elevation_losses() -> float:
            losses = (
                pp_density_func(z=H_core, avg_mflux=avg_mflux, hot_mflux=avg_mflux)
                * DeltaH
            )
            return losses

        accel_losses = _accel_losses()
        friction_losses = _friction_losses()
        forms_losses = _forms_losses()
        elevation_losses = _elevation_losses()

        total_losses = accel_losses + friction_losses + forms_losses + elevation_losses
        if please_print:
            print(
                f"chimney losses: {accel_losses, friction_losses, forms_losses, elevation_losses}"
            )
        return total_losses

    def _downcomer_losses(
        self,
        avg_mflux: float,
        hot_mflux: float,
        please_print: bool,
        quality_func: callable,
        void_fraction_func: callable,
        z_d_func: callable,
        water_func: callable,
        pp_density_func: callable,
    ) -> float:
        avg_mflux = (
            avg_mflux * self.reactor.A * self.reactor.num_rods / self.reactor.A_v
        )
        hot_mflux = (
            hot_mflux * self.reactor.A * self.reactor.num_rods / self.reactor.A_v
        )
        DeltaH = self.reactor.H_chimney + self.reactor.H_core - 0
        H_chimney = self.reactor.H_chimney + self.reactor.H_core
        H_core = self.reactor.H_core
        z_d = z_d_func(avg_mflux=avg_mflux, hot_mflux=avg_mflux)
        rho_f = self.water.liquid_water.rho
        rho_l = rho_f  # equilibrium
        rho_g = self.water.vapor_water.rho
        if z_d is None or z_d > H_core:
            z_d = None
        D_e = self.reactor.D_e_v

        def _accel_losses() -> float:
            return 0

        def _friction_losses() -> float:

            reynolds = calc_reynolds(
                mflux=avg_mflux,
                D_e=D_e,
                mu=water_func(z=0, avg_mflux=avg_mflux, hot_mflux=avg_mflux).mu,
            )
            f = self.friction_factor.rough_friction_factor(reynolds=reynolds, D_e=D_e)
            rho = water_func(z=0, avg_mflux=avg_mflux, hot_mflux=avg_mflux).rho

            p_friction_losses = (f / rho) * DeltaH
            return (avg_mflux**2 / (2 * G_C_in_hr * D_e)) * (p_friction_losses)

        def _forms_losses() -> float:
            rho = water_func(z=0, avg_mflux=avg_mflux, hot_mflux=avg_mflux).rho
            return (avg_mflux**2 * self.reactor.loss_downcomer) / (2 * G_C_in_hr * rho)

        def _elevation_losses() -> float:
            losses = (
                pp_density_func(z=0, avg_mflux=avg_mflux, hot_mflux=avg_mflux)
                * DeltaH
                * -1
            )
            return losses  # negative! going down

        accel_losses = _accel_losses()
        friction_losses = _friction_losses()
        forms_losses = _forms_losses()
        elevation_losses = _elevation_losses()

        total_losses = accel_losses + friction_losses + forms_losses + elevation_losses
        if please_print:
            print(
                f"downcomer losses: {accel_losses, friction_losses, forms_losses, elevation_losses}"
            )
        return total_losses

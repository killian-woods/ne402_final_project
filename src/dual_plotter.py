from pathlib import Path
from functools import cache, cached_property

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad

from temperature_analysis import TemperatureAnalysis
from pressure_drop import PressureDrop
from avg_flux_from_hot_flux import avg_flux_from_hot_flux
from pp_friction_multiplier import TwoPhaseFrictionMultiplier
from pp_forms_multiplier import TwoPhaseFormsMultiplier

from consts import (
    NUM_PLOT_POINTS,
    BOTH_FIGS_PATH,
    HOT_FIGS_PATH,
    CRIT_FIGS_PATH,
    AVG_FIGS_PATH,
    FIGS_PATH,
)

from utilities.data.palettes import DefaultPalette, BoldPalette
from utilities.data.format_plot import format_plot
from utilities.data.save_plot import save_plot
from utilities.units.iapws97_imperial import ImperialIAPWS97
from utilities.units.conversions import convert_units
from utilities.units.consts import G_C_in_hr


class DualPlotter:
    def __init__(
        self,
        crit_anals: TemperatureAnalysis,
        hot_anals: TemperatureAnalysis,
        avg_anals: TemperatureAnalysis,
        hot_mflux: float,
        avg_mflux: float,
    ):
        self.crit_anals = crit_anals
        self.hot_anals = hot_anals
        self.avg_anals = avg_anals
        self.crit_mflux = hot_mflux
        self.hot_mflux = self.crit_mflux
        self.avg_mflux = avg_mflux

    def plot_all_solo(self) -> None:
        self.plot_all_crit()
        self.plot_all_hot()
        self.plot_all_avg()

    def plot_all_crit(self) -> None:
        args = {
            "avg_mflux": self.avg_mflux,
            "hot_mflux": self.hot_mflux,
            "results_dir": CRIT_FIGS_PATH,
        }
        self.crit_anals.plot_bulk_temp(**args)
        self.crit_anals.plot_wall_temp(**args)
        self.crit_anals.plot_outer_fuel_temp(**args)
        self.crit_anals.plot_fuel_centerline_temp(**args)
        self.crit_anals.plot_all_temp_dists(**args)
        self.crit_anals.plot_twall_tbulk_and_chi(**args)

    def plot_all_hot(self) -> None:
        args = {
            "avg_mflux": self.avg_mflux,
            "hot_mflux": self.hot_mflux,
            "results_dir": HOT_FIGS_PATH,
        }
        self.hot_anals.plot_bulk_temp(**args)
        self.hot_anals.plot_wall_temp(**args)
        self.hot_anals.plot_outer_fuel_temp(**args)
        self.hot_anals.plot_fuel_centerline_temp(**args)
        self.hot_anals.plot_all_temp_dists(**args)
        self.hot_anals.plot_twall_tbulk_and_chi(**args)

    def plot_all_avg(self) -> None:
        args = {
            "avg_mflux": self.avg_mflux,
            "hot_mflux": self.avg_mflux,
            "results_dir": AVG_FIGS_PATH,
        }

        self.avg_anals.plot_bulk_temp(**args)
        self.avg_anals.plot_wall_temp(**args)
        self.avg_anals.plot_outer_fuel_temp(**args)
        self.avg_anals.plot_fuel_centerline_temp(**args)
        self.avg_anals.plot_all_temp_dists(**args)
        self.avg_anals.plot_twall_tbulk_and_chi(**args)

    def plot_all_bulk_temps(
        self,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = BOTH_FIGS_PATH,
    ) -> None:
        palette = DefaultPalette()

        z = np.linspace(0, self.crit_anals.reactor.H_core, num_points)

        crit_T_bulk = [
            self.crit_anals.bulk_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        hot_T_bulk = [
            self.hot_anals.bulk_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        avg_T_bulk = [
            self.avg_anals.bulk_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.avg_mflux
            )
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, crit_T_bulk, label=r"$T_{\infty,crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, hot_T_bulk, label=r"$T_{\infty,hot}(z)$", color=palette.next())
        ax.plot(z, avg_T_bulk, label=r"$T_{\infty,avg}(z)$", color=palette.next())

        format_plot(
            ax,
            title="Axial Bulk Temperature Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Bulk Temperatures $T_{\infty}(z)$ [°F]",
            grid=True,
            legend=True,
        )

        save_plot(fig, folder=results_dir, name="T_bulk_distributions")

    def plot_all_wall_temps(
        self,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = BOTH_FIGS_PATH,
    ) -> None:
        palette = DefaultPalette()

        z = np.linspace(0, self.crit_anals.reactor.H_core, num_points)

        crit_T_bulk = [
            self.crit_anals.wall_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        hot_T_bulk = [
            self.hot_anals.wall_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        avg_T_bulk = [
            self.avg_anals.wall_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.avg_mflux
            )
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, crit_T_bulk, label=r"$T_{co,crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, hot_T_bulk, label=r"$T_{co,hot}(z)$", color=palette.next())
        ax.plot(z, avg_T_bulk, label=r"$T_{co,avg}(z)$", color=palette.next())

        T_sat = self.crit_anals.water.liquid_water.T
        ax.axhline(
            T_sat,
            color=palette.COLORS[-1],
            linestyle="--",
            linewidth=1,
            label=r"$T_{sat}$",
        )

        format_plot(
            ax,
            title="Axial Wall Temperature Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Wall Temperatures $T_{co}(z)$ [°F]",
            grid=True,
            legend=True,
        )

        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="T_wall_distributions")

    def plot_all_outer_fuel_temps(
        self,
        num_points: int = NUM_PLOT_POINTS,
        results_dir: Path = BOTH_FIGS_PATH,
    ) -> None:
        palette = DefaultPalette()

        z = np.linspace(0, self.crit_anals.reactor.H_core, num_points)

        crit_T_bulk = [
            self.crit_anals.fuel_outer_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        hot_T_bulk = [
            self.hot_anals.fuel_outer_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        avg_T_bulk = [
            self.avg_anals.fuel_outer_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.avg_mflux
            )
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, crit_T_bulk, label=r"$T_{fo,crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, hot_T_bulk, label=r"$T_{fo,hot}(z)$", color=palette.next())
        ax.plot(z, avg_T_bulk, label=r"$T_{fo,avg}(z)$", color=palette.next())

        format_plot(
            ax,
            title="Axial Outer Fuel Temperature Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Outer Fuel Temperatures $T_{fo}(z)$ [°F]",
            grid=True,
            legend=True,
        )

        save_plot(fig, folder=results_dir, name="T_fo_distributions")

    def plot_all_fuel_centerline_temps(
        self,
        num_points: int = NUM_PLOT_POINTS // 2,
        results_dir: Path = BOTH_FIGS_PATH,
    ) -> None:
        palette = BoldPalette()

        z = np.linspace(0, self.crit_anals.reactor.H_core, num_points)

        crit_T_bulk = [
            self.crit_anals.fuel_centerline_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        hot_T_bulk = [
            self.hot_anals.fuel_centerline_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux
            )
            for zi in z
        ]
        avg_T_bulk = [
            self.avg_anals.fuel_centerline_temperature(
                z=zi, avg_mflux=self.avg_mflux, hot_mflux=self.avg_mflux
            )
            for zi in z
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, crit_T_bulk, label=r"$T_{fi,crit}(z)$", color=palette.COLORS[-2])
        ax.plot(z, hot_T_bulk, label=r"$T_{fi,hot}(z)$", color=palette.next())
        ax.plot(z, avg_T_bulk, label=r"$T_{fi,avg}(z)$", color=palette.next())

        crit_max = self.crit_anals.max_fuel_centerline_temperature(
            avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux, z_not_temp=True
        )
        hot_max = self.hot_anals.max_fuel_centerline_temperature(
            avg_mflux=self.avg_mflux, hot_mflux=self.hot_mflux, z_not_temp=True
        )
        avg_max = self.avg_anals.max_fuel_centerline_temperature(
            avg_mflux=self.avg_mflux, hot_mflux=self.avg_mflux, z_not_temp=True
        )
        T_sat = self.crit_anals.reactor.T_melt

        ax.axvline(
            crit_max,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$z_{max,crit}$",
        )

        ax.axvline(
            hot_max,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$z_{max,hot}$",
        )

        ax.axvline(
            avg_max,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$z_{max,avg}$",
        )

        palette.next()

        ax.axhline(
            T_sat,
            color=palette.next(),
            linestyle="--",
            linewidth=1,
            label=r"$T_{melt}$",
        )

        format_plot(
            ax,
            title="Axial Fuel Centerline Temperature Distributions",
            xlabel=r"Axial Position $z$ [in]",
            ylabel=r"Fuel Centerline Temperatures $T_{fi}(z)$ [°F]",
            grid=True,
            legend=True,
        )

        ax.get_lines()[-4].set_linewidth(1)
        ax.get_lines()[-3].set_linewidth(1)
        ax.get_lines()[-2].set_linewidth(1)
        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="T_fi_distributions")

    def plot_mfluxes_vs_deltaP(
        self,
        num_points: int = NUM_PLOT_POINTS // 10,
        results_dir: Path = FIGS_PATH,
    ) -> None:

        palette = DefaultPalette()
        dP = np.linspace(10, 60, num_points)

        deltaP = PressureDrop()

        print("Starting with avg_mflux calculations...")
        avg_mflux = [deltaP.find_avg_mass_flux(dPi) for dPi in dP]
        print("Finished with avg_mflux calculations")
        print("Starting with hot_mflux calculations...")
        hot_mflux = [
            deltaP.find_hot_mass_flux(avg_Gi, dPi) for dPi, avg_Gi in zip(dP, avg_mflux)
        ]
        print("Finished with hot_mflux calculations")
        print("Starting with extrapolated_avg_mflux calculations...")
        extrapolated_avg_mflux = [
            avg_flux_from_hot_flux(avg_mflux=avg_Gi, hot_mflux=hot_Gi)
            for avg_Gi, hot_Gi in zip(avg_mflux, hot_mflux)
        ]
        print("Finished with extrapolated_avg_mflux calculations")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dP, avg_mflux, label=r"$G_{avg}(\Delta P_p)$", color=palette.next())
        ax.plot(dP, hot_mflux, label=r"$G_{hot}(\Delta P_p)$", color=palette.next())
        ax.plot(
            dP,
            extrapolated_avg_mflux,
            label=r"Extrapolated $G_{avg}(\Delta P_p)$",
            color=palette.COLORS[-2],
            linestyle="--",
            linewidth=1,
        )

        format_plot(
            ax,
            title=r"Mass Fluxes vs. Total System Pressure Drop, $\Delta P_p$",
            xlabel=r"Total System Pressure Drop, $\Delta P_p$ [psia]",
            ylabel=r"Mass Flux $G(z)$ [lbm/in^2.hr]",
            grid=True,
            legend=True,
        )

        ax.get_lines()[-1].set_linewidth(1)

        save_plot(fig, folder=results_dir, name="dP_mflux_distributions")

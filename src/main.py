from reactor import Reactor
from heat_flux import HeatFlux
from enthalpy import Enthalpy
from quality import Quality
from void_fraction import VoidFraction
from pressure_drop import PressureDrop
from pp_density import TwoPhaseDensity
from critical_heat_flux import CriticalHeatFlux
from temperature_analysis import TemperatureAnalysis
from dual_plotter import DualPlotter
from avg_flux_from_hot_flux import avg_flux_from_hot_flux

from consts import CRIT_FIGS_PATH, HOT_FIGS_PATH, AVG_FIGS_PATH, BOTH_FIGS_PATH

qflux = HeatFlux()
h = Enthalpy()
x = Quality()
alpha = VoidFraction()
mixture_density = TwoPhaseDensity()
cqflux = CriticalHeatFlux()

qflux.plot_heat_fluxs()

print(qflux.avg_q0, qflux.hot_q0, qflux.avg_lambda)
print(
    qflux.avg_heat_flux_integrated(Reactor.H_core)
    * 3.14
    * Reactor.D_o
    / (Reactor.rod_power)
)

deltaP = PressureDrop()

guessed_total_pressure = 30  # psia
avg_mflux = deltaP.find_avg_mass_flux(deltaP=guessed_total_pressure)
print(f"\navg_mflux: {avg_mflux}")

hot_mflux = deltaP.find_hot_mass_flux(
    avg_mflux=avg_mflux, deltaP=guessed_total_pressure
)

print(f"\nhot_mflux: {hot_mflux}")

print(cqflux.calculate_min_CPR(avg_mflux=avg_mflux, hot_mflux=hot_mflux))
print(cqflux.calculate_CPR(z=Reactor.H_core, avg_mflux=avg_mflux, hot_mflux=hot_mflux))


hot_mflux = cqflux.calculate_mflux_for_1dot2(
    avg_mflux=avg_mflux, hot_mflux_guess=hot_mflux
)

print(f"\nhot_mflux: {hot_mflux}")

avg_mflux = avg_flux_from_hot_flux(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
cqflux.plot_CPR_graph(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
q0_crit = cqflux.calculate_critical_heat_flux_q0(
    z=Reactor.H_core, avg_mflux=avg_mflux, hot_mflux=hot_mflux
)
print(f"q0_crit: {q0_crit}")

crit_anals = TemperatureAnalysis(q0_crit=q0_crit, type_name="Critical")

if (
    crit_anals.max_fuel_centerline_temperature(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
    >= Reactor.T_melt
):
    hot_mflux = crit_anals.calculate_mflux_for_meltdown(
        avg_mflux=avg_mflux, hot_mflux_guess=hot_mflux
    )
    print("!!!")
    print(hot_mflux)

deltaP_core = deltaP.hot_core_losses(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
avg_mflux = deltaP.find_mass_flux_from_core_deltaP(deltaP=deltaP_core)
print(hot_mflux, avg_mflux)

hot_anals = TemperatureAnalysis(q0_crit=qflux.hot_q0, type_name="Hot")
avg_anals = TemperatureAnalysis(q0_crit=qflux.avg_q0, type_name="Average")

dp = DualPlotter(
    crit_anals=crit_anals,
    hot_anals=hot_anals,
    avg_anals=avg_anals,
    hot_mflux=hot_mflux,
    avg_mflux=avg_mflux,
)

dp.plot_all_solo()

crit_anals.plot_all_heat_fluxes(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
crit_anals.plot_all_enthalpys(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
crit_anals.plot_all_qualities(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
crit_anals.plot_all_void_fractions(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
crit_anals.plot_all_densities(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
crit_anals.plot_philos(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
crit_anals.plot_forms_factor(avg_mflux=avg_mflux, hot_mflux=hot_mflux)

dp.plot_all_bulk_temps()
dp.plot_all_wall_temps()
dp.plot_all_outer_fuel_temps()
dp.plot_all_fuel_centerline_temps()

h.plot_enthalpys(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
x.plot_qualities(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
alpha.plot_void_fractions(hot_mflux=hot_mflux, avg_mflux=avg_mflux)
mixture_density.plot_densities(hot_mflux=hot_mflux, avg_mflux=avg_mflux)

print(
    f"final pressure loss of system: {deltaP.avg_total_losses(avg_mflux=avg_mflux, please_print=True)}"
)
print(
    f"final core pressure loss of system: {deltaP.avg_core_losses(avg_mflux=avg_mflux, please_print=True)}"
)
print(
    f"this is how much hot_mflux would differ if you calculate it from the approximated avg_mflux: {deltaP.find_hot_mass_flux(
    avg_mflux=avg_mflux, deltaP=deltaP.avg_total_losses(avg_mflux=avg_mflux)
)}. not a lot! "
)
print(
    f"centerline values:\ncrit: {dp.crit_anals.max_fuel_centerline_temperature(avg_mflux=avg_mflux,hot_mflux=hot_mflux,z_not_temp=True), dp.crit_anals.max_fuel_centerline_temperature(avg_mflux=avg_mflux,hot_mflux=hot_mflux,z_not_temp=False)}\nhot:{dp.hot_anals.max_fuel_centerline_temperature(avg_mflux=avg_mflux,hot_mflux=hot_mflux,z_not_temp=True), dp.hot_anals.max_fuel_centerline_temperature(avg_mflux=avg_mflux,hot_mflux=hot_mflux,z_not_temp=False)}\navg: {dp.avg_anals.max_fuel_centerline_temperature(avg_mflux=avg_mflux,hot_mflux=avg_mflux,z_not_temp=True), dp.avg_anals.max_fuel_centerline_temperature(avg_mflux=avg_mflux,hot_mflux=avg_mflux,z_not_temp=False)}"
)

print(
    "Working on final graph. This will take a LONG time (>30 minutes on a lower-end computer)."
)
dp.plot_mfluxes_vs_deltaP()

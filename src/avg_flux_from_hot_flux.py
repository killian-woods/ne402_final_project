from pressure_drop import PressureDrop

DELTAP = PressureDrop()


def avg_flux_from_hot_flux(avg_mflux: float, hot_mflux: float) -> float:
    hot_core_deltaP = DELTAP.hot_core_losses(avg_mflux=avg_mflux, hot_mflux=hot_mflux)
    true_avg_mflux = DELTAP.find_mass_flux_from_core_deltaP(deltaP=hot_core_deltaP)
    return true_avg_mflux

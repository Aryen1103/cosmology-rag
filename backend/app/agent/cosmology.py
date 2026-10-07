"""Flat ΛCDM background quantities, so the agent can turn a paper's parameters into
distances and ages instead of doing the arithmetic in its head.

Radiation is neglected (well under 0.1% below z ~ 10), which keeps the age integral
analytic. numpy only, like the rest of the backend.
"""
from __future__ import annotations

import numpy as np

C_KM_S = 299_792.458
HUBBLE_TIME_GYR = 977.792  # 1 / (1 km/s/Mpc) in Gyr
_GRID_POINTS = 4097


def _age_gyr(z: float, h0: float, om0: float) -> float:
    # Exact for flat matter + Λ: t(a) = 2 / (3 H0 √ΩΛ) · asinh(√(ΩΛ/Ωm) · a^(3/2))
    ol0 = 1.0 - om0
    a = 1.0 / (1.0 + z)
    return HUBBLE_TIME_GYR / h0 * 2 / (3 * np.sqrt(ol0)) * float(np.arcsinh(np.sqrt(ol0 / om0) * a**1.5))


def flat_lcdm(z: float, h0: float, om0: float) -> dict[str, float]:
    """Hubble rate, distances (Mpc) and times (Gyr) at redshift z for flat ΛCDM."""
    if not 0 <= z <= 1e4:
        raise ValueError("z must be between 0 and 10000")
    if not 20 <= h0 <= 150:
        raise ValueError("H0 must be between 20 and 150 km/s/Mpc")
    if not 0 < om0 < 1:
        raise ValueError("Om0 must be strictly between 0 and 1")

    e_of_z = lambda zz: np.sqrt(om0 * (1 + zz) ** 3 + 1 - om0)  # noqa: E731
    # Integrate in ln(1+z) so the grid stays fine at low z even when z is large.
    x = np.linspace(0.0, np.log1p(z), _GRID_POINTS)
    zz = np.expm1(x)
    comoving = C_KM_S / h0 * float(np.trapezoid((1 + zz) / e_of_z(zz), x))
    age_now, age_then = _age_gyr(0.0, h0, om0), _age_gyr(z, h0, om0)

    return {
        "hubble_rate_km_s_mpc": h0 * float(e_of_z(z)),
        "comoving_distance_mpc": comoving,
        "luminosity_distance_mpc": (1 + z) * comoving,
        "angular_diameter_distance_mpc": comoving / (1 + z),
        "lookback_time_gyr": age_now - age_then,
        "age_at_z_gyr": age_then,
        "age_today_gyr": age_now,
    }

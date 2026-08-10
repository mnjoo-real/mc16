"""Consistency tests for the Step-6A tau_lag scale diagnostic."""

import numpy as np

from hysteresis_scale import (integrate_linear_lag, linear_metrics,
                              linear_steady_state, loop_area,
                              relaxation_loss_power, rotating_metrics)
from magnetic_carousel import MU0, Params


def main():
    p = Params(); B0 = .020

    # 1. Quasistatic loop area vanishes linearly with frequency.
    w1 = linear_metrics(B0, 1e-3, p.tau_lag, p.chi_eff)["loop_area_exact"]
    w2 = linear_metrics(B0, 2e-3, p.tau_lag, p.chi_eff)["loop_area_exact"]
    assert abs(w2/w1-2) < 1e-9

    # 2. tau=0 is the reversible equilibrium response with zero loop area.
    _, H, M = linear_steady_state(B0, 20.42, 0, p.chi_eff)
    assert np.max(np.abs(M-p.chi_eff*H)) < 1e-10
    assert abs(loop_area(H, M)) < 1e-8

    # 3. A numerical prescribed-field integration reaches the exact periodic
    # solution and repeats over consecutive steady cycles.
    omega = 53.41
    t, Hn, Mn = integrate_linear_lag(B0, omega, p.tau_lag, p.chi_eff,
                                     steps_per_cycle=1600, cycles=10)
    n = 1600
    _, _, Mex = linear_steady_state(B0, omega, p.tau_lag, p.chi_eff, n=n)
    assert np.max(np.abs(Mn[-(n+1):]-Mex))/np.max(np.abs(Mex)) < 2e-8
    assert np.max(np.abs(Mn[-(n+1):]-Mn[-(2*n+1):-(n)]))/np.max(np.abs(Mex)) < 2e-8

    # 4. Trapezoidal loop area converges at second order in phase spacing.
    exact = linear_metrics(B0, omega, p.tau_lag, p.chi_eff)["loop_area_exact"]
    errors = []
    for ns in (128, 256, 512):
        _, Hs, Ms = linear_steady_state(B0, omega, p.tau_lag, p.chi_eff, n=ns)
        errors.append(abs(loop_area(Hs, Ms)-exact))
    assert errors[1] < .27*errors[0] and errors[2] < .27*errors[1]

    # 5. Circular-field torque has the passive sign.
    rm = rotating_metrics(B0, omega, p.tau_lag, p.volume, p.chi_eff)
    assert rm["torque"] > 0

    # 6. Loop work, relaxation dissipation, and rotating mechanical work are
    # mutually consistent, including the factor of two for circular fields.
    t, H, M = linear_steady_state(B0, omega, p.tau_lag, p.chi_eff, n=8192)
    pd = relaxation_loss_power(H, M, p.tau_lag, p.chi_eff)
    Ed = np.trapezoid(pd, t)
    El = loop_area(H, M)
    assert abs(Ed/El-1) < 2e-7
    assert abs(rm["energy_density_per_cycle"]/(2*El)-1) < 2e-7
    assert abs(rm["power"]/(omega*rm["torque"])-1) < 1e-14

    print(f"hysteresis-scale tests passed: Wloop={exact:.6f} J/m3, "
          f"tau_rot={rm['torque']:.6e} Nm, low-frequency ratio={w2/w1:.9f}")


if __name__ == "__main__":
    main()

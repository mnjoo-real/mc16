"""Mandatory validation tests for the isolated conducting-sphere kernel."""

import numpy as np

from magnetic_diffusion_sphere import (fit_reduced_response, response_from_kappa_a,
                                       response_l, static_scattered_l)

A = .006
SIGMA = 6e6


def main():
    # 1. omega -> 0: exact l=1 static permeable-sphere moment.
    for mur in (10.0, 50.0, 100.0, 500.0):
        r = response_l(1, 1e-9, A, SIGMA, mur)
        expected = 4*np.pi*A**3*(mur-1)/(mur+2)
        rel = abs(r.dipole_moment_per_H0-expected)/abs(expected)
        assert rel < 1e-10

    # mu_r=1 has zero static polarization but finite passive AC response.
    r_nonmag = response_l(1, 1e-9, A, SIGMA, 1.0)
    assert abs(r_nonmag.dipole_moment_per_H0) < 1e-15
    assert response_l(1, 100.0, A, SIGMA, 1.0).scattered.imag < 0

    # 2. sigma -> 0 exactly recovers every static mode.
    for l in range(1, 7):
        r = response_l(l, 123.0, A, 0.0, 100.0)
        assert r.scattered == static_scattered_l(l, 100.0)

    # 3/4/5/6. Finite, bounded, passive and continuous for |kappa*a| 1e-3..30.
    ka = np.geomspace(1e-3, 30.0, 1200)
    for l in range(1, 7):
        vals = np.array([response_from_kappa_a(l, x, 100.0)[0] for x in ka])
        assert np.all(np.isfinite(vals))
        assert np.max(np.abs(vals)) < 3.0
        assert np.max(vals.imag) < 1e-10       # exp(+iwt) passive sign
        jumps = np.abs(np.diff(vals))/np.maximum(1.0, np.abs(vals[:-1]))
        assert np.max(jumps) < .03
    # Directly straddle the analytic-series/direct-Bessel switch.
    left = response_from_kappa_a(1, .249999, 100.0)[0]
    right = response_from_kappa_a(1, .250001, 100.0)[0]
    assert abs(left-right) < 1e-6

    # 7. Higher-l low-frequency limits are the independently derived
    # (l+1)(mu_r-1)/(l*mu_r+l+1).
    for mur in (1.0, 10.0, 100.0, 500.0):
        for l in range(1, 7):
            r = response_l(l, 1e-9, A, SIGMA, mur)
            assert abs(r.scattered-static_scattered_l(l, mur)) < 1e-9

    # Physical l=1 Joule power is nonnegative throughout the sweep.
    for omega in np.geomspace(1e-6, 1e7, 300):
        r = response_l(1, omega, A, SIGMA, 100.0)
        assert r.joule_loss_per_H0_sq >= 0.0

    # 8. Reduced model has strictly stable poles and exceeds requested accuracy.
    reduced = fit_reduced_response(1, A, SIGMA, 100.0, order=2)
    assert np.all(reduced.poles < 0.0)
    assert reduced.max_magnitude_error < .01
    assert reduced.max_phase_error_deg < 1.0
    print("sphere diffusion tests passed: "
          f"ROM mag={100*reduced.max_magnitude_error:.4g}% "
          f"phase={reduced.max_phase_error_deg:.4g} deg, "
          f"poles={reduced.poles}")


if __name__ == "__main__":
    main()

"""Validation tests for the independent Maxwell-stress wrench evaluator."""

import numpy as np

from magnetic_diffusion_sphere import MU0
from magnetic_diffusion_wrench import (axisymmetric_linear_gradient_modes,
                                       maxwell_wrench, rotating_uniform_analytic,
                                       uniform_field_modes)

A = .006
SIGMA = 6e6
MUR = 100.0


def norm(x):
    return np.linalg.norm(x)


def main():
    # 1. Uniform static field: no force or torque.
    modes = uniform_field_modes((1e4, 0, 0), 0.0, A, 0.0, MUR)
    F, T = maxwell_wrench(modes, A, cycle_average=False)
    assert norm(F) < 1e-10 and norm(T) < 1e-12

    # 2. Linearly oscillating uniform field: zero mean wrench.
    modes = uniform_field_modes((1e4, 0, 0), 53.41, A, SIGMA, MUR)
    F, T = maxwell_wrench(modes, A, cycle_average=True)
    assert norm(F) < 1e-10 and norm(T) < 1e-12

    # 3/4. Circular field: zero force, passive-sign torque, and P=Omega*tau.
    omega = 72.26; B0 = .02; H0 = B0/MU0
    modes = uniform_field_modes((H0, -1j*H0, 0), omega, A, SIGMA, MUR)
    F, T = maxwell_wrench(modes, A, cycle_average=True)
    tau_exact, joule = rotating_uniform_analytic(B0, omega, A, SIGMA, MUR)
    assert norm(F) < 1e-9
    assert T[2] > 0
    assert abs(T[2]/tau_exact-1) < 2e-7
    assert abs(omega*T[2]/joule-1) < 2e-7

    # 5. Enclosing-radius invariance.
    values = []
    for ratio in (1.05, 1.2, 1.5, 2.0):
        values.append(maxwell_wrench(modes, A, ratio*A, 20, 40, True)[1][2])
    assert np.ptp(values)/abs(np.mean(values)) < 2e-7

    # 6. Angular quadrature convergence.
    values = []
    for nmu, nphi in ((8, 16), (12, 24), (20, 40), (32, 64)):
        values.append(maxwell_wrench(modes, A, 1.2*A, nmu, nphi, True)[1][2])
    assert abs(values[-1]/tau_exact-1) < 2e-7
    assert abs(values[-1]-values[-2])/abs(tau_exact) < 2e-7

    # 7. Static l=1/l=2 force bridge to the analytic permeable-sphere result.
    Hc, gradient = 1e4, 2e6
    modes = axisymmetric_linear_gradient_modes(Hc, gradient, 0, A, 0, MUR)
    F, T = maxwell_wrench(modes, A, 1.2*A, 20, 40, False)
    alpha_H = 4*np.pi*A**3*(MUR-1)/(MUR+2)
    F_exact = MU0*alpha_H*Hc*gradient
    assert abs(F[2]/F_exact-1) < 2e-8
    assert norm(F[:2]) < 1e-10

    # 8. The axisymmetric mirror-symmetric benchmark has no artificial torque.
    assert norm(T) < 1e-12
    print(f"diffusion wrench tests passed: rotating tau={tau_exact:.6e} Nm, "
          f"static Fz={F_exact:.6e} N, radius spread={np.ptp(values):.3e}")


if __name__ == "__main__":
    main()

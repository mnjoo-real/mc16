"""Prescribed-field diagnostics for the production model's relaxation law.

This module does not implement rate-independent hysteresis.  It isolates the
current constitutive closure in magnetization-density form,

    dM/dt = (M_eq(H)-M)/tau,
    M_eq = min(chi_eff |H|, Ms) H/|H|,

for a stationary material sample.  The production point moment is m=V M; its
corotational ``omega_body x m`` term is deliberately absent in prescribed
material-frame tests because the prescribed H is already in that frame.
"""

from __future__ import annotations

import numpy as np

from magnetic_carousel import MU0


def equilibrium_M(H, chi_eff=3.0, Ms=1.7e6):
    """Reversible equilibrium magnetization for scalar or vector H [A/m]."""
    H = np.asarray(H, dtype=float)
    if H.ndim == 0:
        return np.clip(chi_eff*H, -Ms, Ms)
    hn = np.linalg.norm(H, axis=-1)
    mag = np.minimum(chi_eff*hn, Ms)
    return mag[..., None]*H/np.maximum(hn[..., None], 1e-300)


def linear_steady_state(B0, omega, tau, chi_eff=3.0, n=2048,
                        cycles=1, kind="linear"):
    """Exact periodic trajectory for the unsaturated first-order lag law.

    Returned arrays include both endpoints of ``cycles`` complete periods.
    ``kind='linear'`` uses B=B0*cos(theta)*x. ``kind='circular'`` uses
    B=B0*(cos(theta)*x+sin(theta)*y).
    """
    if omega < 0 or tau < 0 or B0 < 0:
        raise ValueError("B0, omega, and tau must be nonnegative")
    theta = np.linspace(0, 2*np.pi*cycles, n*cycles+1)
    H0 = B0/MU0
    q = omega*tau
    if kind == "linear":
        H = H0*np.cos(theta)
        M = chi_eff*H0*(np.cos(theta)+q*np.sin(theta))/(1+q*q)
    elif kind == "circular":
        H = H0*np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
        M = chi_eff*H0/(1+q*q)*np.column_stack(
            (np.cos(theta)+q*np.sin(theta),
             np.sin(theta)-q*np.cos(theta), np.zeros_like(theta)))
    else:
        raise ValueError("kind must be 'linear' or 'circular'")
    if tau == 0 or omega == 0:
        M = chi_eff*H
    t = theta/omega if omega > 0 else theta
    return t, H, M


def integrate_linear_lag(B0, omega, tau, chi_eff=3.0, Ms=1.7e6,
                         steps_per_cycle=1000, cycles=8):
    """RK4 integration used to check periodic convergence independently."""
    if omega <= 0:
        H = np.array((B0/MU0,)); M = equilibrium_M(H, chi_eff, Ms)
        return np.array((0.0,)), H, M
    period = 2*np.pi/omega; dt = period/steps_per_cycle
    t = np.arange(steps_per_cycle*cycles+1)*dt
    H = B0/MU0*np.cos(omega*t)
    M = np.empty_like(t); M[0] = 0.0
    if tau == 0:
        return t, H, equilibrium_M(H, chi_eff, Ms)
    def rhs(tt, mm):
        return (equilibrium_M(B0/MU0*np.cos(omega*tt), chi_eff, Ms)-mm)/tau
    for i in range(len(t)-1):
        ti, mi = t[i], M[i]
        k1 = rhs(ti, mi)
        k2 = rhs(ti+.5*dt, mi+.5*dt*k1)
        k3 = rhs(ti+.5*dt, mi+.5*dt*k2)
        k4 = rhs(ti+dt, mi+dt*k3)
        M[i+1] = mi+dt*(k1+2*k2+2*k3+k4)/6
    return t, H, M


def loop_area(H, M):
    """Return mu0 integral H dM [J/m^3/cycle] for a closed scalar loop."""
    H = np.asarray(H); M = np.asarray(M)
    return float(MU0*np.sum(.5*(H[:-1]+H[1:])*np.diff(M)))


def relaxation_loss_power(H, M, tau, chi_eff=3.0):
    """Instantaneous dissipated power density [W/m^3] in the linear law.

    For free-energy density psi=mu0*M^2/(2 chi)-mu0*H*M and gradient-flow
    kinetics Mdot=(chi H-M)/tau, the nonnegative dissipation is
    mu0 |chi H-M|^2/(chi tau).  Works componentwise for vector fields.
    """
    H, M = np.asarray(H), np.asarray(M)
    if tau == 0:
        return np.zeros(H.shape[:-1] if H.ndim > 1 else H.shape)
    residual = chi_eff*H-M
    sq = residual*residual if residual.ndim == 1 else np.sum(residual*residual, axis=-1)
    return MU0*sq/(chi_eff*tau)


def linear_metrics(B0, omega, tau, chi_eff=3.0, Ms=1.7e6, n=4096):
    """Loop metrics for the unsaturated sinusoidal prescribed-field case."""
    _, H, M = linear_steady_state(B0, omega, tau, chi_eff, n=n)
    q = omega*tau; H0 = B0/MU0
    area_exact = np.pi*MU0*chi_eff*H0**2*q/(1+q*q)
    apparent_Hc = H0*q/np.sqrt(1+q*q)
    apparent_Mr = chi_eff*H0*q/(1+q*q)
    peak_M = chi_eff*H0/np.sqrt(1+q*q)
    slope_origin = chi_eff/(1+q*q)
    power = area_exact*omega/(2*np.pi)
    return dict(H=H, M=M, loop_area=loop_area(H, M),
                loop_area_exact=area_exact, apparent_Hc=apparent_Hc,
                apparent_Mr=apparent_Mr, remanence_ratio=apparent_Mr/Ms,
                peak_M=peak_M, differential_susceptibility=slope_origin,
                average_loss_power_density=power)


def rotating_metrics(B0, omega, tau, volume, chi_eff=3.0):
    """Exact circular-field torque and loss for the current lag closure."""
    H0 = B0/MU0; q = omega*tau
    torque_density = MU0*chi_eff*H0**2*q/(1+q*q)
    torque = volume*torque_density
    power = omega*torque
    energy_cycle = 2*np.pi*torque
    return dict(torque=torque, torque_density=torque_density, power=power,
                energy_per_cycle=energy_cycle,
                energy_density_per_cycle=energy_cycle/volume)

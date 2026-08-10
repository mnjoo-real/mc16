"""Magnetoquasistatic diffusion modes of a conducting permeable sphere.

Convention and derivation
-------------------------
All phasors multiply exp(+i*omega*t).  Neglecting displacement current,

    curl H = sigma E,       curl E = -i omega B,

so, for homogeneous real mu and div(B)=0,

    laplacian(B) = i omega mu sigma B.

Define kappa**2=i*omega*mu*sigma with the attenuation branch
kappa=(1+i)/delta and delta=sqrt(2/(omega*mu*sigma)).  A regular spherical
solution is j_l(z*r/a), where z=i*kappa*a; thus z**2=-i*Pi and the ordinary
spherical Bessel equation is equivalent to the diffusion equation.

Use a toroidal vector potential A=f_l(r) X_lm.  Its poloidal magnetic field has

    B_r proportional to l(l+1) f/r,
    B_t proportional to d(r f)/dr / r.

Outside, f=u*(r/a)**l + v*(a/r)**(l+1).  Inside, f=C*j_l(z*r/a).
Continuity of normal B and tangential H at r=a gives

    C*j_l(z) = u+v,
    (u+v)*g/mu_r = (l+1)u-l*v,
    g = 1 + z*j_l'(z)/j_l(z).

Therefore the scattered/applied surface coefficient is

    t_l = v/u = [mu_r(l+1)-g] / [mu_r*l+g].

Its static limit is (l+1)(mu_r-1)/(l*mu_r+l+1), not a universal 1/3
demagnetizing factor.  For l=1, m=2*pi*a**3*H0*t_1, which recovers the exact
static permeable-sphere moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import least_squares

MU0 = 4.0e-7*np.pi


@dataclass(frozen=True)
class SphereModeResponse:
    l: int
    omega: float
    radius: float
    sigma: float
    mu_r: float
    kappa: complex
    z: complex
    delta: float
    Pi1: float
    log_derivative: complex
    scattered: complex
    scattered_static: float
    internal_boundary: complex
    internal_boundary_static: float
    normalized_scattered: complex
    normalized_internal_boundary: complex
    center_field: complex
    center_field_static: float
    normalized_center_field: complex
    dipole_moment_per_H0: complex
    joule_loss_per_H0_sq: float
    joule_loss_metric: float


@dataclass(frozen=True)
class ReducedResponseModel:
    """Stable real-pole model of normalized scattered response.

    G(s) = direct + sum_j residue_j/(1+s*tau_j), with s=i*omega for the
    exp(+i*omega*t) convention.  Its continuous-time poles are -1/tau_j.
    """
    l: int
    direct: float
    tau: np.ndarray
    residues: np.ndarray
    max_magnitude_error: float
    max_phase_error_deg: float

    @property
    def poles(self):
        return -1.0/self.tau

    def evaluate(self, omega):
        omega = np.asarray(omega, dtype=float)
        return self.direct+np.sum(
            self.residues/(1.0+1.0j*omega[..., None]*self.tau), axis=-1)


def _bessel_log_derivative(l: int, z: complex) -> complex:
    """Return g=1+z*j_l'(z)/j_l(z), stably as z approaches zero.

    For |z|<0.25, evaluate the convergent normalized power series

      j_l(z) proportional to z^l sum_k c_k z^(2k),
      c_k/c_(k-1) = -1/[2 k (2l+2k+1)],

    and differentiate that series analytically.  This avoids dividing two
    separately underflowing spherical-Bessel values.
    """
    z = complex(z)
    if abs(z) < 0.25:
        term = 1.0+0.0j
        series = term
        z_dseries = 0.0+0.0j
        for k in range(1, 30):
            term *= -z*z/(2.0*k*(2*l+2*k+1))
            series += term
            z_dseries += 2.0*k*term
            if abs(term) < 2e-16*abs(series):
                break
        return l+1.0+z_dseries/series
    jl = spherical_jn(l, z)
    djl = spherical_jn(l, z, derivative=True)
    return 1.0+z*djl/jl


def static_scattered_l(l: int, mu_r: float) -> float:
    """Exact magnetostatic scattered/applied coefficient for spatial mode l."""
    return (l+1.0)*(mu_r-1.0)/(l*mu_r+l+1.0)


def static_internal_boundary_l(l: int, mu_r: float) -> float:
    """Exact static total/applied f_l coefficient at the sphere surface."""
    return mu_r*(2*l+1.0)/(l*mu_r+l+1.0)


def response_l(l: int, omega: float, radius: float, sigma: float,
               mu_r: float, H0: float = 1.0) -> SphereModeResponse:
    """Complex response of one spherical spatial mode.

    ``H0`` is used only to dimensionalize the l=1 dipole moment and Joule
    power.  For l>1, ``dipole_moment_per_H0`` is NaN because the response is a
    higher multipole rather than a magnetic dipole.
    """
    if l < 1 or int(l) != l:
        raise ValueError("l must be a positive integer")
    if omega < 0 or radius <= 0 or sigma < 0 or mu_r <= 0:
        raise ValueError("require omega>=0, radius>0, sigma>=0, mu_r>0")
    l = int(l)
    t0 = static_scattered_l(l, mu_r)
    c0 = static_internal_boundary_l(l, mu_r)
    if omega == 0.0 or sigma == 0.0:
        kappa = 0.0+0.0j; z = 0.0+0.0j; delta = np.inf; Pi1 = 0.0
        g = complex(l+1.0); t = complex(t0); c = complex(c0)
    else:
        mu = MU0*mu_r
        delta = np.sqrt(2.0/(omega*mu*sigma))
        kappa = (1.0+1.0j)/delta
        z = 1.0j*kappa*radius
        Pi1 = mu*sigma*omega*radius*radius
        g = _bessel_log_derivative(l, z)
        t = (mu_r*(l+1.0)-g)/(mu_r*l+g)
        c = mu_r*(2*l+1.0)/(mu_r*l+g)
    nt = t/t0 if abs(t0) > 1e-15 else complex(np.nan, np.nan)
    nc = c/c0
    center_static = 3.0*mu_r/(mu_r+2.0) if l == 1 else 0.0
    if l == 1:
        if abs(z) < 0.25:
            # z/(3*j1(z)) = 1/[1-z^2/10+z^4/280-...]
            zz = z*z
            radial_center = 1.0/(1.0-zz/10.0+zz*zz/280.0-zz**3/15120.0)
        else:
            radial_center = z/(3.0*spherical_jn(1, z))
        center = c*radial_center
        ncenter = center/center_static
        moment_per_H0 = 2.0*np.pi*radius**3*t
        # exp(+iwt): passive magnetic polarizability has Im(m/H0)<=0.
        joule = -0.5*omega*MU0*moment_per_H0.imag*H0*H0
    else:
        center = 0.0+0.0j; ncenter = complex(np.nan, np.nan)
        moment_per_H0 = complex(np.nan, np.nan); joule = np.nan
    loss_metric = -omega*t.imag
    return SphereModeResponse(l, omega, radius, sigma, mu_r, kappa, z, delta,
                              Pi1, g, t, t0, c, c0, nt, nc, center,
                              center_static, ncenter, moment_per_H0, joule,
                              loss_metric)


def response_from_kappa_a(l: int, kappa_a: float, mu_r: float):
    """Dimensionless helper for stability sweeps with real |kappa*a|."""
    if kappa_a < 0:
        raise ValueError("kappa_a must be nonnegative")
    # Pi=|kappa*a|^2 because |1+i|/delta gives |kappa*a|^2=Pi.
    Pi1 = kappa_a*kappa_a
    z = (-1.0+1.0j)*kappa_a/np.sqrt(2.0)
    g = _bessel_log_derivative(l, z)
    t = (mu_r*(l+1.0)-g)/(mu_r*l+g)
    return t, g, Pi1


def fit_reduced_response(l: int, radius: float, sigma: float, mu_r: float,
                         omega_max: float = 400.0, order: int = 2,
                         n_sample: int = 400) -> ReducedResponseModel:
    """Fit a stable causal real-pole model over 0<=omega<=omega_max.

    The exact perfect-shielding limit t_l -> -1 fixes the direct term after
    normalization by the nonzero static coefficient.  Positive time constants
    are enforced through logarithmic parameters, and the final residue is
    constrained so G(0)=1 exactly.  No such normalization exists for mu_r=1.
    """
    t0 = static_scattered_l(l, mu_r)
    if abs(t0) < 1e-14:
        raise ValueError("normalized reduced response is undefined when static response is zero")
    if order < 1 or omega_max <= 0:
        raise ValueError("order and omega_max must be positive")
    omega = np.unique(np.r_[0.0, np.geomspace(1e-4, omega_max, n_sample//2),
                            np.linspace(0.0, omega_max, n_sample//2)])
    exact = np.array([response_l(l, w, radius, sigma, mu_r).normalized_scattered
                      for w in omega])
    direct = -1.0/t0

    def unpack(x):
        tau = np.exp(x[:order])
        residues = np.r_[x[order:], 1.0-direct-np.sum(x[order:])]
        return tau, residues

    def evaluate(x):
        tau, residues = unpack(x)
        return direct+np.sum(residues[None, :]/
                             (1.0+1.0j*omega[:, None]*tau[None, :]), axis=1)

    def residual(x):
        e = (evaluate(x)-exact)/np.maximum(np.abs(exact), 1e-12)
        return np.r_[e.real, e.imag]

    tau0 = np.geomspace(1e-5, 2e-3, order)
    residues0 = np.full(max(0, order-1), (1.0-direct)/order)
    fit = least_squares(residual, np.r_[np.log(tau0), residues0],
                        max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    tau, residues = unpack(fit.x)
    approx = evaluate(fit.x)
    mag_error = float(np.max(np.abs(np.abs(approx)/np.abs(exact)-1.0)))
    phase_error = float(np.max(np.abs(np.angle(approx/exact)))*180.0/np.pi)
    order_idx = np.argsort(tau)
    return ReducedResponseModel(l, direct, tau[order_idx], residues[order_idx],
                                mag_error, phase_error)

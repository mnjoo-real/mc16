"""Maxwell-stress force/torque for spherical magnetic-diffusion modes.

Harmonic and spatial conventions
--------------------------------
Phasors multiply exp(+i*omega*t).  ``scipy.special.sph_harm_y`` supplies
orthonormal complex Y_lm(theta,phi), including the Condon--Shortley phase.
The applied vacuum magnetic scalar potential is

    Phi_app = A_lm r**l Y_lm,

and H=-grad(Phi), B=mu0*H.  The diffusion kernel uses a toroidal-vector-
potential scattered coefficient t_l.  Matching radial B gives the equivalent
scalar-potential coefficient

    Phi_sc = A_lm a**(2l+1) s_l r**(-l-1) Y_lm,
    s_l = -l*t_l/(l+1).

For B(t)=Re[Bhat exp(i*omega*t)], direct averaging of the instantaneous vacuum
tensor T=(BB-I B^2/2)/mu0 gives

  <T>.n = [ Re(Bhat (Bhat*.n)) - n |Bhat|^2/2 ]/(2 mu0).

The factor 1/2 is absent for a genuinely static real field.  Applied-field-only
stress is subtracted numerically; its exact closed-surface integral is zero in
a source-free region, and subtraction reduces cancellation error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import sph_harm_y

from magnetic_diffusion_sphere import MU0, SphereModeResponse, response_l


@dataclass(frozen=True)
class AppliedMode:
    l: int
    m: int
    coefficient: complex
    response: SphereModeResponse

    @property
    def scalar_scattered_ratio(self):
        return -self.l*self.response.scattered/(self.l+1.0)


def surface_quadrature(radius: float, n_mu: int, n_phi: int):
    """Gauss-Legendre polar and uniform-azimuth surface quadrature."""
    mu, wmu = np.polynomial.legendre.leggauss(n_mu)
    phi = 2*np.pi*np.arange(n_phi)/n_phi
    pts, normals, weights = [], [], []
    for u, wu in zip(mu, wmu):
        st = np.sqrt(max(0.0, 1-u*u))
        for ph in phi:
            n = np.array((st*np.cos(ph), st*np.sin(ph), u))
            normals.append(n); pts.append(radius*n)
            weights.append(radius*radius*wu*2*np.pi/n_phi)
    return np.asarray(pts), np.asarray(normals), np.asarray(weights)


def _solid_harmonic(points, l, m, radial_power):
    points = np.asarray(points)
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(points[:, 2]/r, -1, 1))
    phi = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2*np.pi)
    return r**radial_power*sph_harm_y(l, m, theta, phi)


def scalar_potential(points, modes, radius, include_scattered=True):
    points = np.asarray(points)
    out = np.zeros(len(points), dtype=complex)
    for mode in modes:
        out += mode.coefficient*_solid_harmonic(points, mode.l, mode.m, mode.l)
        if include_scattered:
            out += (mode.coefficient*radius**(2*mode.l+1)
                    * mode.scalar_scattered_ratio
                    * _solid_harmonic(points, mode.l, mode.m, -mode.l-1))
    return out


def magnetic_field(points, modes, radius, include_scattered=True):
    """Complex external vacuum B from a centered modal expansion."""
    points = np.asarray(points, dtype=float)
    # Relative step balances roundoff and the O(h^2) central difference while
    # remaining far below the shortest l<=6 angular/radial scale.
    h = radius*2e-5
    H = np.empty((len(points), 3), dtype=complex)
    for axis in range(3):
        d = np.zeros(3); d[axis] = h
        H[:, axis] = -(scalar_potential(points+d, modes, radius, include_scattered)
                       - scalar_potential(points-d, modes, radius, include_scattered))/(2*h)
    return MU0*H


def _traction(B, normals, cycle_average):
    if cycle_average:
        bn = np.einsum("ij,ij->i", np.conjugate(B), normals)
        return (np.real(B*bn[:, None])-0.5*np.sum(np.abs(B)**2, axis=1)[:, None]*normals)/(2*MU0)
    Br = np.real(B)
    bn = np.einsum("ij,ij->i", Br, normals)
    return (Br*bn[:, None]-0.5*np.sum(Br*Br, axis=1)[:, None]*normals)/MU0


def maxwell_wrench(modes, radius, r_eval=None, n_mu=24, n_phi=48,
                   cycle_average=True, subtract_background=True):
    """Return net force and torque from external vacuum Maxwell stress."""
    if r_eval is None:
        r_eval = 1.2*radius
    if r_eval <= radius:
        raise ValueError("stress surface must lie outside the sphere")
    points, normals, weights = surface_quadrature(r_eval, n_mu, n_phi)
    B = magnetic_field(points, modes, radius, True)
    traction = _traction(B, normals, cycle_average)
    if subtract_background:
        B0 = magnetic_field(points, modes, radius, False)
        traction -= _traction(B0, normals, cycle_average)
    force = np.sum(traction*weights[:, None], axis=0)
    torque = np.sum(np.cross(points, traction)*weights[:, None], axis=0)
    return force, torque


def uniform_field_modes(H_complex, omega, radius, sigma, mu_r):
    """l=1 modes for a spatially uniform complex H phasor."""
    hx, hy, hz = np.asarray(H_complex, dtype=complex)
    c = np.sqrt(3/(8*np.pi))
    c0 = np.sqrt(3/(4*np.pi))
    # rY_1,-1=c(x-iy), rY_1,1=-c(x+iy), rY_1,0=c0*z.
    # Phi=-H.r.
    A_minus = -(hx+1j*hy)/(2*c)
    A_plus = (hx-1j*hy)/(2*c)
    A_zero = -hz/c0
    response = response_l(1, omega, radius, sigma, mu_r)
    return [AppliedMode(1, -1, A_minus, response),
            AppliedMode(1, 0, A_zero, response),
            AppliedMode(1, 1, A_plus, response)]


def axisymmetric_linear_gradient_modes(H0, gradient, omega, radius, sigma, mu_r):
    """Uniform z field plus source-free axial gradient dHz/dz=gradient."""
    modes = uniform_field_modes((0, 0, H0), omega, radius, sigma, mu_r)
    # Phi_quad=-(G/4)(2z^2-x^2-y^2) and
    # r^2 Y20=sqrt(5/(16pi))(2z^2-x^2-y^2).
    A20 = -gradient*np.sqrt(np.pi/5)
    modes.append(AppliedMode(2, 0, A20,
                             response_l(2, omega, radius, sigma, mu_r)))
    return modes


def rotating_uniform_analytic(B0, omega, radius, sigma, mu_r):
    """Analytic mean torque and Joule power for B0(x cos wt+y sin wt)."""
    r = response_l(1, omega, radius, sigma, mu_r)
    alpha = r.dipole_moment_per_H0
    H0 = B0/MU0
    torque_z = -MU0*alpha.imag*H0*H0
    joule = omega*torque_z
    return torque_z, joule


def tau_lag_rotating_torque(B0, omega, radius, mu_r, tau_lag):
    """Same rotating field under the point model's first-order lag closure."""
    alpha_static = 4*np.pi*radius**3*(mu_r-1)/(mu_r+2)
    alpha = alpha_static/(1+1j*omega*tau_lag)
    H0 = B0/MU0
    return -MU0*alpha.imag*H0*H0


def project_radial_field(sample_B, projection_radius, lmax, n_mu=24, n_phi=48):
    """Project a source-free applied B field onto Phi=A_lm r^l Y_lm.

    ``sample_B(points)`` returns lab-frame B at offsets from the sphere center.
    Since H_r=-sum_lm l A_lm r^(l-1)Y_lm, orthonormality gives A_lm directly.
    """
    points, normals, weights = surface_quadrature(projection_radius, n_mu, n_phi)
    B = np.asarray(sample_B(points))
    Hr = np.einsum("ij,ij->i", B, normals)/MU0
    dOmega = weights/projection_radius**2
    theta = np.arccos(np.clip(normals[:, 2], -1, 1))
    phi = np.mod(np.arctan2(normals[:, 1], normals[:, 0]), 2*np.pi)
    coefficients = {}
    for l in range(1, lmax+1):
        for m in range(-l, l+1):
            Y = sph_harm_y(l, m, theta, phi)
            coefficients[(l, m)] = (-projection_radius**(1-l)/l
                                     * np.sum(Hr*np.conjugate(Y)*dOmega))
    return coefficients


def modes_from_coefficients(coefficients, omega, radius, sigma, mu_r,
                            keep_l=None):
    modes = []
    for (l, m), coefficient in coefficients.items():
        if keep_l is None or l in keep_l:
            modes.append(AppliedMode(l, m, coefficient,
                                     response_l(l, omega, radius, sigma, mu_r)))
    return modes

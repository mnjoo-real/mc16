"""
Magnetic carousel  --  numerical model
======================================

A ring of N permanent magnets with alternating polarity is glued to a horizontal
disc that spins about its vertical axis with angular velocity `omega`.  A fixed
non-magnetic plate sits above the disc and a steel ball rests on the plate.

The ball is subject to two competing magnetic actions:

  (A) a translational force  F_i = sum_j m_j d(B_j)/d(x_i)   which pulls the ball
      towards maxima of |B| (i.e. towards a magnet)  ->  drags it PROGRADE;

  (B) a torque  tau = m x B .  Because the travelling magnetic wave makes the
      field vector rotate about a horizontal (radial) axis, and because the
      induced magnetisation lags behind B (eddy currents / hysteresis), the ball
      is spun so that its contact point is dragged forwards; friction then rolls
      it BACKWARDS  ->  RETROGRADE.

Field model
-----------
Every magnet is a uniformly magnetised cuboid.  Its scalar potential is obtained
in closed form from the equivalent magnetic surface charge (Coulombian model);
B = -mu0 * grad(psi) is evaluated on a Cartesian grid *in the frame co-rotating
with the disc* (where the field is static), so the expensive field evaluation is
done once.  Above the magnets the field is both curl- and divergence-free, which
is used to reconstruct the full gradient tensor from in-plane derivatives only:

    dBx/dz = dBz/dx ,  dBy/dz = dBz/dy ,  dBz/dz = -(dBx/dx + dBy/dy)

Ball model
----------
Soft-magnetic sphere, treated as a point dipole at its centre with a relaxing
(lagging) moment written in the *body* frame:

    dm/dt = omega_ball x m + (m_eq - m)/tau_lag ,
    m_eq  = min(3*V*|B|/mu0, V*Ms) * Bhat

The `omega_ball x m` term is what makes the torque vanish when the ball spins
synchronously with the field -- exactly the induction/hysteresis-motor physics.

Contact
-------
Regularised Coulomb friction (tanh) at the contact point plus rolling
resistance, so that stick (rolling) and slip are handled by one smooth law.

Units: SI throughout.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, asdict, field as dc_field

import numpy as np

MU0 = 4.0e-7 * np.pi
G_ACC = 9.81

# ----------------------------------------------------------------------------
# optional numba
# ----------------------------------------------------------------------------
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                             # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):                                # type: ignore
        def deco(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco


# ============================================================================
# Parameters
# ============================================================================
@dataclass
class Params:
    # --- magnet ring -------------------------------------------------------
    n_mag: int = 12            # number of magnets
    r_mag: float = 0.060       # radius of the magnet ring            [m]
    mag_lx: float = 0.010      # magnet size (radial)                 [m]
    mag_ly: float = 0.010      # magnet size (tangential)             [m]
    mag_lz: float = 0.005      # magnet height                        [m]
    Br: float = 1.30           # remanence of NdFeB                   [T]
    alternating: bool = True   # alternate N-up / S-up

    # --- geometry ----------------------------------------------------------
    gap: float = 0.008         # magnet top face -> top of plate      [m]
    ball_R: float = 0.006      # ball radius                          [m]
    rho_ball: float = 7800.0   # steel density                    [kg/m^3]

    # --- ball magnetics ----------------------------------------------------
    chi_eff: float = 3.0       # 3*(mu_r-1)/(mu_r+2) -> 3 for mu_r>>1
    Ms: float = 1.7e6          # saturation magnetisation             [A/m]
    tau_lag: float = 4.0e-3    # magnetisation relaxation time        [s]

    # --- contact -----------------------------------------------------------
    mu_k: float = 0.15         # sliding friction coefficient
    mu_roll: float = 0.004     # rolling resistance coefficient
    mu_spin: float = 0.004     # drilling (spin) friction coefficient
    u_reg: float = 3.0e-3      # friction regularisation velocity     [m/s]

    # --- confining rim (a lip on the plate; set r_rim=0 to disable) --------
    r_rim: float = 0.082       # radius of the retaining rim          [m]
    k_rim: float = 2000.0      # rim stiffness                        [N/m]
    c_rim: float = 0.6         # rim damping                        [N s/m]

    # --- drive -------------------------------------------------------------
    omega: float = 25.0        # disc angular velocity                [rad/s]

    # --- numerics ----------------------------------------------------------
    grid_ds: float = 5.0e-4    # field grid spacing                   [m]
    grid_pad: float = 0.030    # grid extends r_mag + pad             [m]
    dz_fd: float = 2.5e-4      # z step used for the z-derivative     [m]

    # ------------------------------------------------------------------
    @property
    def z_ball(self) -> float:
        """height of the ball centre above the magnet top face"""
        return self.gap + self.ball_R

    @property
    def mass(self) -> float:
        return self.rho_ball * 4.0 / 3.0 * np.pi * self.ball_R ** 3

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.ball_R ** 2

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.ball_R ** 3

    @property
    def alpha(self) -> float:
        """m = alpha * B  in the linear (unsaturated) regime"""
        return self.chi_eff * self.volume / MU0

    @property
    def m_sat(self) -> float:
        return self.volume * self.Ms

    @property
    def wavelength(self) -> float:
        """spatial period of the travelling wave at the magnet radius"""
        return 2.0 * np.pi * self.r_mag / self.n_mag

    @property
    def k_wave(self) -> float:
        return 2.0 * np.pi / self.wavelength

    @property
    def ka(self) -> float:
        """key dimensionless number: ball radius / (wavelength/2pi)"""
        return self.k_wave * self.ball_R

    def field_key(self) -> str:
        """hash of every parameter the *field* depends on (for caching)"""
        keys = ("n_mag", "r_mag", "mag_lx", "mag_ly", "mag_lz", "Br",
                "alternating", "gap", "ball_R", "grid_ds", "grid_pad", "dz_fd")
        s = "|".join(f"{k}={getattr(self, k)!r}" for k in keys)
        return hashlib.md5(s.encode()).hexdigest()[:16]


# ============================================================================
# Analytic potential of a uniformly magnetised cuboid
# ============================================================================
def _Phi(u, v, w):
    """Antiderivative with d2Phi/du dv = 1/sqrt(u^2+v^2+w^2)."""
    R = np.sqrt(u * u + v * v + w * w)
    eps = 1e-30
    return (u * np.arcsinh(v / np.sqrt(u * u + w * w + eps))
            + v * np.arcsinh(u / np.sqrt(v * v + w * w + eps))
            - w * np.arctan2(u * v, w * R + eps))


def _rect_potential(x, y, w, hx, hy):
    """Potential (1/4pi omitted) of a unit surface charge on the rectangle
    |x'|<hx, |y'|<hy at vertical offset w."""
    u1, u2 = x + hx, x - hx
    v1, v2 = y + hy, y - hy
    return (_Phi(u1, v1, w) - _Phi(u2, v1, w)
            - _Phi(u1, v2, w) + _Phi(u2, v2, w))


def magnet_potential(x, y, z, cx, cy, cz, hx, hy, hz, M):
    """Magnetic scalar potential psi of one z-magnetised cuboid.
    H = -grad(psi).  Charge +M on the top face, -M on the bottom face."""
    dx, dy = x - cx, y - cy
    top = _rect_potential(dx, dy, z - (cz + hz), hx, hy)
    bot = _rect_potential(dx, dy, z - (cz - hz), hx, hy)
    return M / (4.0 * np.pi) * (top - bot)


# ============================================================================
# Field grid in the co-rotating frame
# ============================================================================
class FieldGrid:
    """B and its full gradient tensor, sampled on a Cartesian grid at the
    height of the ball centre, in the frame that co-rotates with the disc."""

    N_COMP = 8   # Bx, By, Bz, dxBx, dyBy, dxBy(=dyBx), dxBz(=dzBx), dyBz(=dzBy)

    def __init__(self, p: Params, cache_dir: str | None = "fieldcache",
                 verbose: bool = True):
        self.p = p
        self.cache_dir = cache_dir
        fn = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            fn = os.path.join(cache_dir, f"field_{p.field_key()}.npz")
        if fn and os.path.exists(fn):
            d = np.load(fn)
            self.xs, self.ys, self.F = d["xs"], d["ys"], d["F"]
        else:
            if verbose:
                print(f"  [field] building grid (gap={p.gap*1e3:.1f} mm) ...",
                      flush=True)
            self.xs, self.ys, self.F = self._build()
            if fn:
                np.savez_compressed(fn, xs=self.xs, ys=self.ys, F=self.F)
        self.x0, self.y0 = self.xs[0], self.ys[0]
        self.dx = self.xs[1] - self.xs[0]
        self.dy = self.ys[1] - self.ys[0]

    # ------------------------------------------------------------------
    def _magnet_positions(self):
        p = self.p
        ang = 2.0 * np.pi * np.arange(p.n_mag) / p.n_mag
        cx, cy = p.r_mag * np.cos(ang), p.r_mag * np.sin(ang)
        sgn = np.ones(p.n_mag)
        if p.alternating:
            sgn = (-1.0) ** np.arange(p.n_mag)
            if p.n_mag % 2:                       # odd count cannot alternate
                sgn[-1] = 0.5 * (sgn[-1] + sgn[0])
        return cx, cy, sgn, ang

    def _build(self):
        p = self.p
        L = p.r_mag + p.grid_pad
        n = int(2 * L / p.grid_ds) + 1
        xs = np.linspace(-L, L, n)
        ys = xs.copy()
        X, Y = np.meshgrid(xs, ys, indexing="xy")     # rows = y, cols = x

        cx, cy, sgn, ang = self._magnet_positions()
        M = p.Br / MU0
        zc = -p.mag_lz / 2.0                          # top face at z = 0
        hx, hy, hz = p.mag_lx / 2.0, p.mag_ly / 2.0, p.mag_lz / 2.0

        zb, dz = p.z_ball, p.dz_fd
        psi = [np.zeros_like(X) for _ in range(3)]
        for zi, zz in enumerate((zb - dz, zb, zb + dz)):
            for j in range(p.n_mag):
                # magnet is rotated in the plane -> rotate the field point
                ca, sa = np.cos(ang[j]), np.sin(ang[j])
                # coordinates in the magnet's own frame (radial = x)
                xl = ca * X + sa * Y
                yl = -sa * X + ca * Y
                psi[zi] += magnet_potential(xl, yl, zz, p.r_mag, 0.0, zc,
                                            hx, hy, hz, sgn[j] * M)

        ds = p.grid_ds
        Bx = -MU0 * np.gradient(psi[1], ds, axis=1)
        By = -MU0 * np.gradient(psi[1], ds, axis=0)
        Bz = -MU0 * (psi[2] - psi[0]) / (2.0 * dz)

        dxBx = np.gradient(Bx, ds, axis=1)
        dyBy = np.gradient(By, ds, axis=0)
        dxBy = 0.5 * (np.gradient(By, ds, axis=1) + np.gradient(Bx, ds, axis=0))
        dxBz = np.gradient(Bz, ds, axis=1)
        dyBz = np.gradient(Bz, ds, axis=0)

        F = np.empty(X.shape + (self.N_COMP,))
        for i, a in enumerate((Bx, By, Bz, dxBx, dyBy, dxBy, dxBz, dyBz)):
            F[:, :, i] = a
        return xs, ys, F

    # ------------------------------------------------------------------
    def Bmag(self):
        return np.linalg.norm(self.F[:, :, :3], axis=2)

    def sample_circle(self, radius, n_phi=1440):
        """B and |B| on a circle of given radius (co-rotating frame)."""
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        x, y = radius * np.cos(phi), radius * np.sin(phi)
        out = np.empty((n_phi, self.N_COMP))
        for i in range(n_phi):
            out[i] = _bilinear(self.F, self.x0, self.y0, self.dx, self.dy,
                               x[i], y[i])
        return phi, out

    def potential_on_circle(self, radius, n_phi=1440):
        """U(phi) = -1/2 alpha |B|^2  (quasi-static magnetic trap potential)."""
        phi, out = self.sample_circle(radius, n_phi)
        B2 = out[:, 0] ** 2 + out[:, 1] ** 2 + out[:, 2] ** 2
        return phi, -0.5 * self.p.alpha * B2, np.sqrt(B2)


# ============================================================================
# Fast kernels
# ============================================================================
@njit(cache=True, fastmath=True)
def _bilinear(F, x0, y0, dx, dy, x, y):
    ny, nx, nc = F.shape
    fx = (x - x0) / dx
    fy = (y - y0) / dy
    i = int(np.floor(fx))
    j = int(np.floor(fy))
    out = np.zeros(nc)
    if i < 0 or j < 0 or i >= nx - 1 or j >= ny - 1:
        return out
    tx = fx - i
    ty = fy - j
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    for c in range(nc):
        out[c] = (w00 * F[j, i, c] + w10 * F[j, i + 1, c]
                  + w01 * F[j + 1, i, c] + w11 * F[j + 1, i + 1, c])
    return out


@njit(cache=True, fastmath=True)
def _field_lab(F, x0, y0, dx, dy, x, y, ang):
    """B (3) and gradient tensor J[i,j] = dB_j/dx_i (3x3), in the lab frame."""
    c = np.cos(ang)
    s = np.sin(ang)
    xr = c * x + s * y          # rotate lab -> disc frame  (by -ang)
    yr = -s * x + c * y
    q = _bilinear(F, x0, y0, dx, dy, xr, yr)

    Bp = np.empty(3)
    Bp[0], Bp[1], Bp[2] = q[0], q[1], q[2]
    Jp = np.empty((3, 3))
    dxBx, dyBy, dxBy, dxBz, dyBz = q[3], q[4], q[5], q[6], q[7]
    Jp[0, 0] = dxBx
    Jp[0, 1] = dxBy
    Jp[0, 2] = dxBz
    Jp[1, 0] = dxBy
    Jp[1, 1] = dyBy
    Jp[1, 2] = dyBz
    Jp[2, 0] = dxBz          # dBx/dz = dBz/dx   (curl-free)
    Jp[2, 1] = dyBz          # dBy/dz = dBz/dy
    Jp[2, 2] = -(dxBx + dyBy)   # div-free

    R = np.zeros((3, 3))
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    R[2, 2] = 1.0
    B = R @ Bp
    J = R @ Jp @ R.T
    return B, J


@njit(cache=True, fastmath=True)
def _deriv(t, y, F, x0, y0, dx, dy, omega, alpha, m_sat, tau_lag,
           mass, inertia, ball_R, mu_k, mu_roll, mu_spin, u_reg, gravity,
           r_rim, k_rim, c_rim):
    """RHS of the 10-dimensional state
       y = [x, y, vx, vy, wx, wy, wz, mx, my, mz]"""
    px, py = y[0], y[1]
    vx, vy = y[2], y[3]
    wx, wy, wz = y[4], y[5], y[6]
    mx, my, mz = y[7], y[8], y[9]

    B, J = _field_lab(F, x0, y0, dx, dy, px, py, omega * t)

    # --- equilibrium moment with saturation ------------------------------
    Bn = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2) + 1e-18
    meq_mag = alpha * Bn
    if meq_mag > m_sat:
        meq_mag = m_sat
    mex = meq_mag * B[0] / Bn
    mey = meq_mag * B[1] / Bn
    mez = meq_mag * B[2] / Bn

    # --- magnetic force  F_i = sum_j m_j dB_j/dx_i ------------------------
    Fx = J[0, 0] * mx + J[0, 1] * my + J[0, 2] * mz
    Fy = J[1, 0] * mx + J[1, 1] * my + J[1, 2] * mz
    Fz = J[2, 0] * mx + J[2, 1] * my + J[2, 2] * mz

    # --- magnetic torque  tau = m x B ------------------------------------
    Tmx = my * B[2] - mz * B[1]
    Tmy = mz * B[0] - mx * B[2]
    Tmz = mx * B[1] - my * B[0]

    # --- normal load ------------------------------------------------------
    Nf = mass * gravity - Fz
    if Nf < 0.0:
        Nf = 0.0

    # --- contact point velocity  u = v + w x (-a z^) ----------------------
    ux = vx - ball_R * wy
    uy = vy + ball_R * wx
    un = np.sqrt(ux * ux + uy * uy) + 1e-18
    fmag = mu_k * Nf * np.tanh(un / u_reg)
    fx = -fmag * ux / un
    fy = -fmag * uy / un

    # torque of friction about the centre: (0,0,-a) x f
    Tfx = ball_R * fy
    Tfy = -ball_R * fx

    # --- rolling resistance & drilling friction ---------------------------
    wh = np.sqrt(wx * wx + wy * wy) + 1e-18
    tr = mu_roll * Nf * ball_R * np.tanh(wh * ball_R / u_reg)
    Trx = -tr * wx / wh
    Try = -tr * wy / wh
    Trz = -mu_spin * Nf * ball_R * np.tanh(wz * ball_R / u_reg)

    # --- retaining rim ----------------------------------------------------
    Wx = 0.0
    Wy = 0.0
    if r_rim > 0.0:
        rr = np.sqrt(px * px + py * py) + 1e-18
        if rr > r_rim:
            ex, ey = px / rr, py / rr
            vr = vx * ex + vy * ey
            fw = -k_rim * (rr - r_rim) - c_rim * vr
            Wx = fw * ex
            Wy = fw * ey

    out = np.empty(10)
    out[0] = vx
    out[1] = vy
    out[2] = (Fx + fx + Wx) / mass
    out[3] = (Fy + fy + Wy) / mass
    out[4] = (Tmx + Tfx + Trx) / inertia
    out[5] = (Tmy + Tfy + Try) / inertia
    out[6] = (Tmz + Trz) / inertia
    # dm/dt = w x m + (m_eq - m)/tau      (relaxation in the body frame)
    out[7] = (wy * mz - wz * my) + (mex - mx) / tau_lag
    out[8] = (wz * mx - wx * mz) + (mey - my) / tau_lag
    out[9] = (wx * my - wy * mx) + (mez - mz) / tau_lag
    return out


@njit(cache=True, fastmath=True)
def _integrate(y0, t_end, dt, stride, F, x0, y0g, dx, dy, omega, alpha, m_sat,
               tau_lag, mass, inertia, ball_R, mu_k, mu_roll, mu_spin, u_reg,
               gravity, r_rim, k_rim, c_rim):
    nstep = int(t_end / dt)
    nout = nstep // stride + 1
    T = np.empty(nout)
    Y = np.empty((nout, 10))
    D = np.empty((nout, 8))     # Fx,Fy,Fz, Tx,Ty,Tz, |B|, N
    y = y0.copy()
    t = 0.0
    io = 0
    for step in range(nstep + 1):
        if step % stride == 0 and io < nout:
            T[io] = t
            for c in range(10):
                Y[io, c] = y[c]
            B, J = _field_lab(F, x0, y0g, dx, dy, y[0], y[1], omega * t)
            mx, my, mz = y[7], y[8], y[9]
            D[io, 0] = J[0, 0] * mx + J[0, 1] * my + J[0, 2] * mz
            D[io, 1] = J[1, 0] * mx + J[1, 1] * my + J[1, 2] * mz
            D[io, 2] = J[2, 0] * mx + J[2, 1] * my + J[2, 2] * mz
            D[io, 3] = my * B[2] - mz * B[1]
            D[io, 4] = mz * B[0] - mx * B[2]
            D[io, 5] = mx * B[1] - my * B[0]
            D[io, 6] = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
            Nf = mass * gravity - D[io, 2]
            D[io, 7] = Nf if Nf > 0.0 else 0.0
            io += 1
        if step == nstep:
            break
        k1 = _deriv(t, y, F, x0, y0g, dx, dy, omega, alpha, m_sat, tau_lag,
                    mass, inertia, ball_R, mu_k, mu_roll, mu_spin, u_reg,
                    gravity, r_rim, k_rim, c_rim)
        k2 = _deriv(t + 0.5 * dt, y + 0.5 * dt * k1, F, x0, y0g, dx, dy, omega,
                    alpha, m_sat, tau_lag, mass, inertia, ball_R, mu_k, mu_roll,
                    mu_spin, u_reg, gravity, r_rim, k_rim, c_rim)
        k3 = _deriv(t + 0.5 * dt, y + 0.5 * dt * k2, F, x0, y0g, dx, dy, omega,
                    alpha, m_sat, tau_lag, mass, inertia, ball_R, mu_k, mu_roll,
                    mu_spin, u_reg, gravity, r_rim, k_rim, c_rim)
        k4 = _deriv(t + dt, y + dt * k3, F, x0, y0g, dx, dy, omega, alpha,
                    m_sat, tau_lag, mass, inertia, ball_R, mu_k, mu_roll,
                    mu_spin, u_reg, gravity, r_rim, k_rim, c_rim)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = (step + 1) * dt
    return T[:io], Y[:io], D[:io]


# ============================================================================
# High level driver
# ============================================================================
class Result:
    def __init__(self, p, T, Y, D):
        self.p = p
        self.t = T
        self.state = Y            # raw 10-dim state history
        self.x, self.y = Y[:, 0], Y[:, 1]
        self.vx, self.vy = Y[:, 2], Y[:, 3]
        self.w = Y[:, 4:7]
        self.m = Y[:, 7:10]
        self.Fmag = D[:, 0:3]
        self.Tmag = D[:, 3:6]
        self.Bnorm = D[:, 6]
        self.Nload = D[:, 7]
        self.r = np.hypot(self.x, self.y)
        self.phi = np.unwrap(np.arctan2(self.y, self.x))
        self.speed = np.hypot(self.vx, self.vy)
        # tangential velocity (positive = prograde, i.e. same sense as disc)
        sgn = np.sign(p.omega) if p.omega != 0 else 1.0
        that_x, that_y = -self.y / np.maximum(self.r, 1e-9), self.x / np.maximum(self.r, 1e-9)
        self.v_tan = (self.vx * that_x + self.vy * that_y) * sgn
        # contact point slip
        self.slip = np.hypot(self.vx - p.ball_R * self.w[:, 1],
                             self.vy + p.ball_R * self.w[:, 0])

    def orbital_rate(self, frac=0.5):
        """mean orbital angular velocity over the last `frac` of the run"""
        i0 = int(len(self.t) * (1 - frac))
        if len(self.t) - i0 < 3:
            return 0.0
        A = np.polyfit(self.t[i0:], self.phi[i0:], 1)
        return A[0]

    def spin_rate(self, frac=0.5):
        """mean spin about the radial (rolling) axis, sign convention: a
        positive value rolls the ball prograde"""
        i0 = int(len(self.t) * (1 - frac))
        # unit vector: radial direction
        rx, ry = self.x / self.r, self.y / self.r
        w_rad = self.w[:, 0] * rx + self.w[:, 1] * ry
        return float(np.mean(w_rad[i0:]))

    def summary(self):
        p = self.p
        ratio = self.orbital_rate() / p.omega if p.omega else np.nan
        return dict(omega=p.omega, gap=p.gap, ball_R=p.ball_R, n_mag=p.n_mag,
                    Omega_ball=self.orbital_rate(), ratio=ratio,
                    r_mean=float(np.mean(self.r[len(self.r) // 2:])),
                    spin=self.spin_rate(), ka=p.ka,
                    slip=float(np.mean(self.slip[len(self.t) // 2:])))


def simulate(p: Params, t_end=1.2, dt=2.0e-5, stride=25, grid: FieldGrid | None = None,
             r0=None, phi0=0.0, v0=(0.0, 0.0), y_init=None, verbose=False) -> Result:
    """`y_init` (10-vector) overrides r0/phi0/v0 and lets a run continue
    seamlessly from the final state of a previous one -- needed for ramp
    experiments, otherwise every step restarts with zero spin and zero
    magnetisation and the transient swamps the result."""
    g = grid if grid is not None else FieldGrid(p, verbose=verbose)
    if y_init is not None:
        y0 = np.asarray(y_init, dtype=float).copy()
    else:
        if r0 is None:
            r0 = p.r_mag
        y0 = np.zeros(10)
        y0[0] = r0 * np.cos(phi0)
        y0[1] = r0 * np.sin(phi0)
        y0[2], y0[3] = v0
    T, Y, D = _integrate(y0, t_end, dt, stride, g.F, g.x0, g.y0, g.dx, g.dy,
                         p.omega, p.alpha, p.m_sat, p.tau_lag, p.mass,
                         p.inertia, p.ball_R, p.mu_k, p.mu_roll, p.mu_spin,
                         p.u_reg, G_ACC, p.r_rim, p.k_rim, p.c_rim)
    return Result(p, T, Y, D)

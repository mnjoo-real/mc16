"""Standalone scalar and isotropic-vector Jiles--Atherton reference kernels.

The frozen equations and citations are in ``docs/vector_ja_formulation.md``.
This module is rate independent: paths are integrated with magnetic-field
increment as the independent variable, never physical time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnetic_carousel import MU0


@dataclass(frozen=True)
class JilesAthertonParams:
    Ms: float
    a: float
    k: float
    c: float
    alpha: float
    label: str = "synthetic"

    def __post_init__(self):
        if self.Ms <= 0 or self.a <= 0 or self.k <= 0:
            raise ValueError("Ms, a, and k must be positive")
        if not 0 <= self.c <= 1:
            raise ValueError("c must lie in [0,1]")
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if self.alpha*self.Ms/(3*self.a) >= 0.95:
            raise ValueError("near-singular initial susceptibility: alpha*Ms/(3a) >= .95")


SYNTHETIC_PARAMETER_SETS = {
    # These span pinning/reversible scales only; they are not material fits.
    "synthetic_soft": JilesAthertonParams(1.2e6, 4000., 25., .35, 5e-5,
                                           "synthetic_soft"),
    "synthetic_medium": JilesAthertonParams(1.35e6, 6000., 400., .20, 5e-5,
                                             "synthetic_medium"),
    "synthetic_hard": JilesAthertonParams(1.5e6, 9000., 1600., .10, 3e-5,
                                           "synthetic_hard"),
}


def langevin(x):
    """L(x)=coth(x)-1/x with its analytic odd series near zero."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = xs/3-xs**3/45+2*xs**5/945-xs**7/4725
    xb = x[~small]
    out[~small] = 1/np.tanh(xb)-1/xb
    return float(out) if out.ndim == 0 else out


def langevin_prime(x):
    """Derivative of the Langevin function with stable even series."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = 1/3-xs**2/15+2*xs**4/189-xs**6/675
    xb = x[~small]
    out[~small] = 1/xb**2-1/np.sinh(xb)**2
    return float(out) if out.ndim == 0 else out


def anhysteretic_scalar(He, p: JilesAthertonParams):
    x = He/p.a
    return p.Ms*langevin(x), p.Ms/p.a*langevin_prime(x)


def anhysteretic_vector(He, p: JilesAthertonParams):
    """Return isotropic vector M_an and its exact 3x3 Jacobian xi."""
    He = np.asarray(He, dtype=float)
    r = float(np.linalg.norm(He))
    x = r/p.a
    if r == 0.0:
        return np.zeros(3), np.eye(3)*p.Ms/(3*p.a)
    L = langevin(x); radial = p.Ms/p.a*langevin_prime(x)
    tangent = p.Ms*L/r
    u = He/r
    J = tangent*np.eye(3)+(radial-tangent)*np.outer(u, u)
    return p.Ms*L*u, J


def _scalar_derivative(H, state, dH, p):
    Mirr, Mrev = state; M = Mirr+Mrev
    Man, xi = anhysteretic_scalar(H+p.alpha*M, p)
    chif = (Man-M)/p.k
    girr = abs(chif)
    G = girr+p.c*xi
    den = 1-p.alpha*G
    if abs(den) < 1e-10:
        raise FloatingPointError("singular scalar JA differential susceptibility")
    dM = G*dH/den; dHe = dH+p.alpha*dM
    if chif*dHe <= 0 or girr == 0:
        G = p.c*xi; den = 1-p.alpha*G
        if abs(den) < 1e-10:
            raise FloatingPointError("singular reversible scalar JA susceptibility")
        dM = G*dH/den; dHe = dH+p.alpha*dM
        dMirr = 0.0
    else:
        dMirr = np.sign(chif)*(chif*dHe)
    dMrev = p.c*xi*dHe
    return np.array((dMirr, dMrev))


def _vector_derivative(H, state, dH, p):
    Mirr, Mrev = state[:3], state[3:]; M = Mirr+Mrev
    Man, xi = anhysteretic_vector(H+p.alpha*M, p)
    chif = (Man-M)/p.k
    cf = float(np.linalg.norm(chif))
    I = np.eye(3)
    Girr = np.zeros((3, 3)) if cf == 0 else np.outer(chif/cf, chif)
    G = Girr+p.c*xi
    A = np.linalg.solve(I-p.alpha*G, G)
    dM = A@dH; dHe = dH+p.alpha*dM
    drive = float(chif@dHe)
    if drive <= 0 or cf == 0:
        G = p.c*xi
        A = np.linalg.solve(I-p.alpha*G, G)
        dM = A@dH; dHe = dH+p.alpha*dM
        dMirr = np.zeros(3)
    else:
        dMirr = chif/cf*drive
    dMrev = p.c*xi@dHe
    return np.r_[dMirr, dMrev]


def _rk4_segment(H0, state, dH, derivative, p):
    k1 = derivative(H0, state, dH, p)
    k2 = derivative(H0+.5*dH, state+.5*k1, dH, p)
    k3 = derivative(H0+.5*dH, state+.5*k2, dH, p)
    k4 = derivative(H0+dH, state+k3, dH, p)
    return state+(k1+2*k2+2*k3+k4)/6


def _advance_segment(H0, state, dH, derivative, p):
    """Substep large prescribed increments relative to the pinning scale."""
    size = float(abs(dH) if np.ndim(dH) == 0 else np.linalg.norm(dH))
    nsub = max(1, int(np.ceil(size/(2*p.k))))
    dh = dH/nsub
    h = np.asarray(H0).copy() if np.ndim(H0) else float(H0)
    for _ in range(nsub):
        state = _rk4_segment(h, state, dh, derivative, p)
        h = h+dh
    return state


def integrate_scalar_path(H, p: JilesAthertonParams, initial_state=None):
    H = np.asarray(H, dtype=float)
    state = (np.zeros(2) if initial_state is None
             else np.asarray(initial_state, dtype=float).copy())
    states = np.empty((len(H), 2)); states[0] = state
    for i in range(len(H)-1):
        state = _advance_segment(H[i], state, H[i+1]-H[i],
                                 _scalar_derivative, p)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("nonfinite scalar JA state")
        states[i+1] = state
    return dict(H=H, M_irr=states[:, 0], M_rev=states[:, 1],
                M=states.sum(axis=1), final_state=state)


def integrate_vector_path(H, p: JilesAthertonParams, initial_state=None):
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[1] != 3:
        raise ValueError("vector field path must have shape (n,3)")
    state = (np.zeros(6) if initial_state is None
             else np.asarray(initial_state, dtype=float).copy())
    states = np.empty((len(H), 6)); states[0] = state
    for i in range(len(H)-1):
        state = _advance_segment(H[i], state, H[i+1]-H[i],
                                 _vector_derivative, p)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("nonfinite vector JA state")
        states[i+1] = state
    return dict(H=H, M_irr=states[:, :3], M_rev=states[:, 3:],
                M=states[:, :3]+states[:, 3:], final_state=state)


def advance_vector_state(H0, H1, state, p: JilesAthertonParams):
    """Advance one physical JA increment without mutating ``state``.

    This small public wrapper is used by nonlinear field solvers.  Repeated
    trial calls always start from the same committed state, so constitutive
    history is not accumulated by nonlinear iterations.
    """
    state = np.asarray(state, dtype=float).copy()
    out = _advance_segment(np.asarray(H0, dtype=float), state,
                           np.asarray(H1, dtype=float)-np.asarray(H0, dtype=float),
                           _vector_derivative, p)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("nonfinite vector JA state")
    return out


def _anhysteretic_vector_batch(He, p):
    He = np.asarray(He, dtype=float)
    r = np.linalg.norm(He, axis=1)
    x = r/p.a
    L = langevin(x)
    radial = p.Ms/p.a*langevin_prime(x)
    tangent = np.full_like(r, p.Ms/(3*p.a))
    nz = r > 0
    tangent[nz] = p.Ms*L[nz]/r[nz]
    u = np.zeros_like(He); u[nz] = He[nz]/r[nz, None]
    Man = p.Ms*L[:, None]*u
    outer = np.einsum("ni,nj->nij", u, u)
    J = tangent[:, None, None]*np.eye(3)+(radial-tangent)[:, None, None]*outer
    return Man, J


def _vector_derivative_batch(H, state, dH, p):
    Mirr, Mrev = state[:, :3], state[:, 3:]
    M = Mirr+Mrev
    Man, xi = _anhysteretic_vector_batch(H+p.alpha*M, p)
    chif = (Man-M)/p.k
    cf = np.linalg.norm(chif, axis=1)
    u = np.zeros_like(chif); nz = cf > 0; u[nz] = chif[nz]/cf[nz, None]
    Girr = np.einsum("ni,nj->nij", u, chif)
    I = np.broadcast_to(np.eye(3), xi.shape)
    G = Girr+p.c*xi
    A = np.linalg.solve(I-p.alpha*G, G)
    dM = np.einsum("nij,nj->ni", A, dH)
    dHe = dH+p.alpha*dM
    active = nz & (np.einsum("ni,ni->n", chif, dHe) > 0)
    if np.any(~active):
        Gr = p.c*xi[~active]
        Ar = np.linalg.solve(I[~active]-p.alpha*Gr, Gr)
        dM[~active] = np.einsum("nij,nj->ni", Ar, dH[~active])
        dHe[~active] = dH[~active]+p.alpha*dM[~active]
    drive = np.einsum("ni,ni->n", chif, dHe)
    dMirr = np.zeros_like(M); dMirr[active] = u[active]*drive[active, None]
    dMrev = p.c*np.einsum("nij,nj->ni", xi, dHe)
    return np.column_stack((dMirr, dMrev))


def advance_vector_states(H0, H1, states, p: JilesAthertonParams,
                          max_increment_factor=2.0):
    """Vectorized commit-free JA advance for many independent material cells."""
    H0, H1 = np.asarray(H0, dtype=float), np.asarray(H1, dtype=float)
    state = np.asarray(states, dtype=float).copy()
    dH = H1-H0
    nsub = max(1, int(np.ceil(np.max(np.linalg.norm(dH, axis=1))
                                  /(max_increment_factor*p.k))))
    dh = dH/nsub; H = H0.copy()
    for _ in range(nsub):
        k1 = _vector_derivative_batch(H, state, dh, p)
        k2 = _vector_derivative_batch(H+.5*dh, state+.5*k1, dh, p)
        k3 = _vector_derivative_batch(H+.5*dh, state+.5*k2, dh, p)
        k4 = _vector_derivative_batch(H+dh, state+k3, dh, p)
        state += (k1+2*k2+2*k3+k4)/6
        H += dh
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("nonfinite batched vector JA state")
    return state


def alternating_path(H0, points_per_cycle=1000, cycles=4):
    """Demagnetized start, ramp to +H0, then complete closed cycles."""
    ramp = np.linspace(0, H0, points_per_cycle//4+1)
    theta = np.linspace(0, 2*np.pi*cycles, points_per_cycle*cycles+1)
    cyc = H0*np.cos(theta)
    return np.r_[ramp, cyc[1:]]


def circular_path(H0, points_per_cycle=1000, cycles=4):
    ramp = np.linspace(0, H0, points_per_cycle//4+1)
    Hr = np.column_stack((ramp, np.zeros_like(ramp), np.zeros_like(ramp)))
    theta = np.linspace(0, 2*np.pi*cycles, points_per_cycle*cycles+1)
    Hc = H0*np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    return np.vstack((Hr, Hc[1:]))


def vector_cycle_work(H, M):
    """mu0 integral H.dM; positive orientation denotes dissipated work."""
    H, M = np.asarray(H), np.asarray(M)
    return float(MU0*np.sum(np.einsum("ij,ij->i", .5*(H[:-1]+H[1:]),
                                                  np.diff(M, axis=0))))


def scalar_loop_metrics(H, M):
    H, M = np.asarray(H), np.asarray(M)
    area = float(MU0*np.sum(.5*(H[:-1]+H[1:])*np.diff(M)))
    # Last descending half-cycle: +Hmax to -Hmax.
    imax = int(np.argmax(H)); starts = np.where((H[:-1] >= H.max()-.01) &
                                                (np.diff(H) < 0))[0]
    start = starts[-1] if len(starts) else imax
    end_candidates = np.where((np.arange(len(H)) > start) & (H <= H.min()+.01))[0]
    end = end_candidates[0] if len(end_candidates) else len(H)-1
    sl = slice(start, end+1); Hd, Md = H[sl], M[sl]
    def crossing(x, y, target=0.0):
        z = x-target; ids = np.where(z[:-1]*z[1:] <= 0)[0]
        if not len(ids): return np.nan
        i = ids[0]; f = -z[i]/(z[i+1]-z[i]+1e-300)
        return y[i]+f*(y[i+1]-y[i])
    Mr = crossing(Hd, Md, 0.0)
    Hcross = crossing(Md, Hd, 0.0)
    return dict(loop_area=area, Mr=Mr, Hc=abs(Hcross))

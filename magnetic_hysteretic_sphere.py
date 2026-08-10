"""Offline arbitrary-M demagnetization and distributed vector-JA sphere.

This module is deliberately independent of the production Carousel RK4.  A
voxel is a persistent material region in the body frame, not a quadrature
point.  The magnetostatic operator is linear in cell magnetization; the JA
constitutive update coupled through that operator is nonlinear and iterative.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares, root

from magnetic_carousel import MU0, _Phi
from magnetic_hysteresis import advance_vector_states, JilesAthertonParams


MATERIAL_MESH_LEVELS = {"coarse": 4, "medium": 7, "fine": 9}


@dataclass(frozen=True)
class SphereMaterialMesh:
    radius: float
    level: str
    points: np.ndarray
    volumes: np.ndarray
    cell_size: float

    @classmethod
    def build(cls, radius: float, level: str = "medium"):
        if level not in MATERIAL_MESH_LEVELS:
            raise ValueError(f"unknown material mesh {level!r}")
        n = MATERIAL_MESH_LEVELS[level]
        h = 2.0*radius/n
        x = -radius+(np.arange(n)+0.5)*h
        points = np.array(np.meshgrid(x, x, x, indexing="ij")).reshape(3, -1).T
        points = points[np.linalg.norm(points, axis=1) <= radius]
        volumes = np.full(len(points), h**3)
        return cls(radius, level, points, volumes, h)

    @property
    def n_cells(self):
        return len(self.points)

    @property
    def represented_volume(self):
        return float(self.volumes.sum())

    @property
    def exact_volume(self):
        return 4*np.pi*self.radius**3/3

    @property
    def volume_error(self):
        return abs(self.represented_volume-self.exact_volume)/self.exact_volume

    @property
    def surface_error_bound(self):
        """Half a voxel diagonal divided by R (conservative geometry scale)."""
        return np.sqrt(3)*self.cell_size/(2*self.radius)

    def rotated(self, rotation):
        rotation = np.asarray(rotation, dtype=float)
        return SphereMaterialMesh(self.radius, self.level, self.points@rotation.T,
                                  self.volumes.copy(), self.cell_size)


def _rect_potential(x, y, w, hx, hy):
    return (_Phi(x+hx, y+hy, w)-_Phi(x-hx, y+hy, w)
            -_Phi(x+hx, y-hy, w)+_Phi(x-hx, y-hy, w))


def _cuboid_potential(r, magnetization, half_width):
    """Closed-form scalar potential of one uniformly magnetized cube."""
    x, y, z = np.asarray(r, dtype=float)
    mx, my, mz = np.asarray(magnetization, dtype=float)
    q = half_width
    # Positive charge is on the outward face whose normal follows M.
    return (mx*(_rect_potential(y, z, x-q, q, q)
                -_rect_potential(y, z, x+q, q, q))
            + my*(_rect_potential(x, z, y-q, q, q)
                  -_rect_potential(x, z, y+q, q, q))
            + mz*(_rect_potential(x, y, z-q, q, q)
                  -_rect_potential(x, y, z+q, q, q)))/(4*np.pi)


def cuboid_demag_tensor(displacement, cell_size):
    """Cell-centre H/M tensor for a finite uniformly magnetized cube.

    Off-diagonal cells use the analytic rectangular-face potential.  Its
    gradient is evaluated once during operator construction with a centered
    difference at 1e-5 cell widths.  The coincident cube self term is the
    exact ``-I/3`` tensor, so no singular-distance cutoff is present.
    """
    d = np.asarray(displacement, dtype=float)
    if np.linalg.norm(d) < 1e-14*cell_size:
        return -np.eye(3)/3
    eps = 1e-5*cell_size
    tensor = np.empty((3, 3))
    for a in range(3):
        dp = np.zeros(3); dp[a] = eps
        for b in range(3):
            m = np.zeros(3); m[b] = 1.0
            tensor[a, b] = -(_cuboid_potential(d+dp, m, cell_size/2)
                              -_cuboid_potential(d-dp, m, cell_size/2))/(2*eps)
    return tensor


class CuboidDemagOperator:
    """Precomputed H_demag = D M operator for arbitrary voxel M.

    Centre-in-sphere voxelization leaves an unresolved stair-step boundary.
    The symmetric finite-cuboid matrix is therefore split orthogonally into
    the three uniform macro modes and zero-mean micro modes.  The exact sphere
    tensor ``-I/3`` is assigned to the macro subspace; the resolved cuboid
    operator is retained on the micro subspace.  This consistency projection
    is symmetric and negative semidefinite, unlike a row-by-row correction,
    so it preserves the nonnegative magnetostatic-energy property.  Nonuniform
    modes remain mesh approximations and must pass refinement tests.
    """

    def __init__(self, mesh: SphereMaterialMesh, consistency_correction=True):
        self.mesh = mesh
        n = mesh.n_cells
        t0 = time.perf_counter()
        blocks = np.empty((n, 3, n, 3))
        lattice = np.rint(mesh.points/mesh.cell_size).astype(int)
        cache = {}
        for i in range(n):
            for j in range(n):
                key = tuple(lattice[i]-lattice[j])
                if key not in cache:
                    cache[key] = cuboid_demag_tensor(
                        mesh.points[i]-mesh.points[j], mesh.cell_size)
                blocks[i, :, j, :] = cache[key]
        # Suppress only roundoff asymmetry from numerical differentiation.
        matrix = blocks.reshape(3*n, 3*n)
        matrix = .5*(matrix+matrix.T)
        blocks = matrix.reshape(n, 3, n, 3)
        unit = np.tile((0., 0., 1.), (n, 1))
        self.uncorrected_uniform_rms = float(np.sqrt(np.mean(np.sum(
            ((matrix@unit.reshape(-1)).reshape(n, 3)+unit/3)**2, axis=1))))
        if consistency_correction:
            U = np.zeros((3*n, 3))
            for i in range(n):
                U[3*i:3*i+3] = np.eye(3)/np.sqrt(n)
            P = U@U.T
            Q = np.eye(3*n)-P
            matrix = -.3333333333333333*P+Q@matrix@Q
            matrix = .5*(matrix+matrix.T)
            # A continuum demagnetizing operator has spectrum in [-1, 0].
            # Cell-centre collocation can overshoot that interval for voxel-
            # scale modes, so project those unresolved eigenvalues onto the
            # physical energy bounds while retaining their eigenvectors.
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            self.raw_eigenvalue_range = (float(eigenvalues[0]),
                                         float(eigenvalues[-1]))
            matrix = (eigenvectors*np.clip(eigenvalues, -1.0, 0.0))@eigenvectors.T
        else:
            eigenvalues = np.linalg.eigvalsh(matrix)
            self.raw_eigenvalue_range = (float(eigenvalues[0]),
                                         float(eigenvalues[-1]))
        self.matrix = matrix
        self.blocks = matrix.reshape(n, 3, n, 3)
        self.build_seconds = time.perf_counter()-t0
        self.memory_bytes = self.matrix.nbytes

    def field(self, magnetization):
        M = np.asarray(magnetization, dtype=float)
        return (self.matrix@M.reshape(-1)).reshape(self.mesh.n_cells, 3)

    def uniform_validation(self, direction=(0, 0, 1), magnitude=1.0):
        u = np.asarray(direction, dtype=float); u /= np.linalg.norm(u)
        M = np.tile(magnitude*u, (self.mesh.n_cells, 1))
        H = self.field(M); target = -M/3
        mean = np.average(H, axis=0, weights=self.mesh.volumes)
        rms = np.sqrt(np.average(np.sum((H-target)**2, axis=1),
                                 weights=self.mesh.volumes))
        return mean, float(rms), float(np.linalg.norm(mean+magnitude*u/3)/(magnitude/3))

    def solve_linear(self, H_external, mu_r):
        if mu_r <= 1:
            raise ValueError("mu_r must exceed one")
        H = np.asarray(H_external, dtype=float)
        if H.shape == (3,):
            H = np.tile(H, (self.mesh.n_cells, 1))
        chi = mu_r-1
        A = np.eye(3*self.mesh.n_cells)-chi*self.matrix
        return np.linalg.solve(A, (chi*H).reshape(-1)).reshape(-1, 3)

    def rotated(self, rotation):
        R = np.asarray(rotation, dtype=float)
        out = object.__new__(CuboidDemagOperator)
        out.mesh = self.mesh.rotated(R)
        out.blocks = np.einsum("pa,iajb,qb->ipjq", R, self.blocks, R)
        out.matrix = out.blocks.reshape(self.matrix.shape)
        out.build_seconds = 0.0
        out.memory_bytes = out.matrix.nbytes
        out.uncorrected_uniform_rms = self.uncorrected_uniform_rms
        return out


def field_from_cells(observation_points, mesh, magnetization):
    """Magnetization field in vacuum from finite cuboid material cells.

    A 3x3x3 Gauss volume integral of the nonsingular dipole kernel is used.
    This avoids branch-cut cancellation in numerical derivatives of the face
    potential at arbitrary observation points while retaining finite sources.
    The Maxwell surface must lie outside every voxel.
    """
    obs = np.asarray(observation_points, dtype=float)
    M = np.asarray(magnetization, dtype=float)
    H = np.zeros((len(obs), 3))
    x, w = np.polynomial.legendre.leggauss(3)
    offsets = (mesh.cell_size/2*np.array(np.meshgrid(x, x, x, indexing="ij"))
               .reshape(3, -1).T)
    weights = ((mesh.cell_size/2)**3*np.array(np.meshgrid(w, w, w, indexing="ij"))
               .prod(axis=0).reshape(-1))
    for centre, m in zip(mesh.points, M):
        for off, weight in zip(offsets, weights):
            d = obs-(centre+off)
            r = np.linalg.norm(d, axis=1)
            u = d/r[:, None]
            H += weight/(4*np.pi*r**3)[:, None]*(3*u*(u@m)[:, None]-m)
    return H


def maxwell_stress_wrench(surface_points, surface_weights, B_external,
                          H_magnetization, subtract_external=True):
    """Vacuum Maxwell-stress force and centre torque on a closed sphere."""
    r = np.asarray(surface_points, dtype=float)
    n = r/np.linalg.norm(r, axis=1)[:, None]
    B = np.asarray(B_external)+MU0*np.asarray(H_magnetization)
    traction = (B*np.einsum("ij,ij->i", B, n)[:, None]
                -.5*np.einsum("ij,ij->i", B, B)[:, None]*n)/MU0
    if subtract_external:
        Be = np.asarray(B_external)
        traction -= (Be*np.einsum("ij,ij->i", Be, n)[:, None]
                     -.5*np.einsum("ij,ij->i", Be, Be)[:, None]*n)/MU0
    dF = traction*np.asarray(surface_weights)[:, None]
    return dF.sum(axis=0), np.cross(r, dF).sum(axis=0)


class HystereticDemagSphere:
    """Self-consistent distributed vector-JA state on a fixed material mesh."""

    def __init__(self, operator: CuboidDemagOperator, params: JilesAthertonParams,
                 tolerance=1e-6, relaxation=.25, max_iterations=120,
                 constitutive_increment_factor=4.0,
                 use_least_squares_fallback=True):
        self.operator, self.params = operator, params
        self.tolerance, self.relaxation = tolerance, relaxation
        self.max_iterations = max_iterations
        self.constitutive_increment_factor = constitutive_increment_factor
        self.use_least_squares_fallback = use_least_squares_fallback
        n = operator.mesh.n_cells
        self.state = np.zeros((n, 6))
        self.H_internal = np.zeros((n, 3))
        self.H_external = np.zeros((n, 3))
        self.last_info = None

    @property
    def magnetization(self):
        return self.state[:, :3]+self.state[:, 3:]

    def reset(self):
        self.state.fill(0); self.H_internal.fill(0); self.H_external.fill(0)
        self.last_info = None

    def _trial_states(self, target_H):
        # Every trial starts from exactly the same committed physical state.
        return advance_vector_states(self.H_internal, target_H, self.state,
                                     self.params, self.constitutive_increment_factor)

    def trial_increment(self, H_external_new, tolerance=None, relaxation=None,
                        max_iterations=None):
        Hext = np.asarray(H_external_new, dtype=float)
        if Hext.shape == (3,):
            Hext = np.tile(Hext, (self.operator.mesh.n_cells, 1))
        tol = self.tolerance if tolerance is None else tolerance
        relax = self.relaxation if relaxation is None else relaxation
        maxit = self.max_iterations if max_iterations is None else max_iterations
        uniform_external = np.max(np.linalg.norm(Hext-Hext[0], axis=1)) < 1e-12
        uniform_committed = (np.max(np.linalg.norm(self.H_internal-self.H_internal[0], axis=1))
                             < 1e-10 and
                             np.max(np.linalg.norm(self.state-self.state[0], axis=1)) < 1e-7)
        if uniform_external and uniform_committed:
            return self._trial_uniform(Hext[0], tol, relax, maxit)
        guess = self.H_internal+(Hext-self.H_external)
        scale = max(self.params.a, float(np.sqrt(np.mean(np.sum(Hext**2, axis=1)))))
        history = []
        # An under-relaxed Picard predictor reduces the first Anderson residual.
        trial = self._trial_states(guess)
        M = trial[:, :3]+trial[:, 3:]
        mapped = Hext+self.operator.field(M)
        guess += relax*(mapped-guess)

        last, best = {}, {"residual": np.inf}
        trust_radius = 3*(float(np.max(np.linalg.norm(Hext-self.H_external, axis=1)))
                          +self.params.a)
        def closure(flat):
            target = flat.reshape(-1, 3)
            delta = target-self.H_internal
            dn = np.linalg.norm(delta, axis=1)
            factor = np.minimum(1., trust_radius/np.maximum(dn, 1e-300))
            evaluated = self.H_internal+delta*factor[:, None]
            trial_state = self._trial_states(evaluated)
            trial_M = trial_state[:, :3]+trial_state[:, 3:]
            residual_field = target-Hext-self.operator.field(trial_M)
            last.update(target=target, state=trial_state, M=trial_M)
            normalized = float(np.sqrt(np.mean(np.sum(residual_field**2, axis=1)))/scale)
            if normalized < best["residual"]:
                best.update(residual=normalized, target=evaluated.copy(),
                            state=trial_state.copy(), M=trial_M.copy(),
                            field=residual_field.copy())
            return residual_field.reshape(-1)

        def callback(x, f):
            history.append(float(np.sqrt(np.mean(f.reshape(-1, 3)**2))/scale))

        # Broyden's second method is substantially more robust than plain
        # Picard/Anderson for spatial modes crossing different JA branches.
        result = root(closure, guess.reshape(-1), method="broyden2", callback=callback,
                      options={"fatol": tol*scale/np.sqrt(3), "maxiter": maxit,
                               "line_search": "armijo"})
        closure(result.x)
        fallback_evaluations = 0
        if best["residual"] > tol and self.use_least_squares_fallback:
            # Independent bounded-residual fallback.  It is expensive because
            # its numerical Jacobian samples every coupled field component,
            # but it distinguishes a Broyden failure from absence of a root.
            max_nfev = 10 if self.params.k < 100 else 25
            ls = least_squares(closure, result.x, xtol=tol/10, ftol=tol/10,
                               gtol=tol/10, max_nfev=max_nfev, x_scale="jac")
            closure(ls.x)
            fallback_evaluations = int(ls.nfev)
        residual = best["residual"]
        iteration = int(getattr(result, "nit", getattr(result, "nfev", len(history))))
        converged = residual <= tol
        guess = best["target"]
        trial = best["state"]
        M = best["M"]
        info = dict(iterations=iteration, residual=residual,
                    converged=converged,
                    residual_history=np.asarray(history), H_internal=guess,
                    H_external=Hext, state=trial, M=M,
                    fallback_evaluations=fallback_evaluations)
        return info

    def _trial_uniform(self, Hext, tol, relax, maxit):
        """Solve the exact uniform invariant subspace of a spherical mesh."""
        H0, state0 = self.H_internal[0], self.state[0]
        scale = max(self.params.a, float(np.linalg.norm(Hext)))
        guess = H0+(Hext-self.H_external[0])
        first = advance_vector_states(H0[None], guess[None], state0[None],
                                      self.params, self.constitutive_increment_factor)[0]
        Mfirst = first[:3]+first[3:]
        guess += relax*(Hext-Mfirst/3-guess)
        history, best = [], {"residual": np.inf}
        trust_radius = 3*(float(np.linalg.norm(Hext-self.H_external[0]))
                          +self.params.a)

        def closure(h):
            h = np.asarray(h)
            delta = h-H0
            evaluated = H0+delta*min(1., trust_radius/max(np.linalg.norm(delta), 1e-300))
            state = advance_vector_states(H0[None], evaluated[None],
                                          state0[None], self.params,
                                          self.constitutive_increment_factor)[0]
            M = state[:3]+state[3:]
            f = h-Hext+M/3
            residual = float(np.linalg.norm(f)/scale)
            history.append(residual)
            if residual < best["residual"]:
                best.update(residual=residual, H=evaluated.copy(),
                            state=state.copy(), M=M.copy())
            return f

        # The uniform invariant problem has only three unknowns.  Powell's
        # safeguarded hybrid Newton method avoids the runaway Anderson trial
        # fields that can occur at a sharp JA reversal.
        result = root(closure, guess, method="hybr",
                      options={"xtol": tol, "maxfev": maxit})
        closure(result.x)
        n = self.operator.mesh.n_cells
        return dict(iterations=int(getattr(result, "nit", result.nfev)),
                    residual=best["residual"], converged=best["residual"] <= tol,
                    residual_history=np.asarray(history),
                    H_internal=np.tile(best["H"], (n, 1)),
                    H_external=np.tile(Hext, (n, 1)),
                    state=np.tile(best["state"], (n, 1)),
                    M=np.tile(best["M"], (n, 1)), uniform_mode=True)

    def advance(self, H_external_new, **kwargs):
        info = self.trial_increment(H_external_new, **kwargs)
        if not info["converged"]:
            raise RuntimeError(f"JA+demag iteration failed: residual={info['residual']:.3e}")
        # The only state mutation: exactly once after physical convergence.
        self.H_internal = info["H_internal"].copy()
        self.H_external = info["H_external"].copy()
        self.state = info["state"].copy()
        self.last_info = info
        return info

    def advance_adaptive(self, H_external_new, max_bisections=5, **kwargs):
        """Commit one requested increment, bisecting its physical path if needed.

        Intermediate commits are real points on the linearly interpolated
        external-field history, not nonlinear trial iterations.  If the full
        recursively refined increment fails, the entry state is restored.
        """
        target = np.asarray(H_external_new, dtype=float)
        if target.shape == (3,):
            target = np.tile(target, (self.operator.mesh.n_cells, 1))
        saved = (self.state.copy(), self.H_internal.copy(),
                 self.H_external.copy(), self.last_info)
        try:
            try:
                info = self.advance(target, **kwargs)
                info["physical_substeps"] = 1
                return info
            except RuntimeError:
                if max_bisections <= 0:
                    raise
                middle = .5*(self.H_external+target)
                first = self.advance_adaptive(middle, max_bisections-1, **kwargs)
                second = self.advance_adaptive(target, max_bisections-1, **kwargs)
                second["physical_substeps"] = (first.get("physical_substeps", 1)
                                                +second.get("physical_substeps", 1))
                return second
        except Exception:
            self.state, self.H_internal, self.H_external, self.last_info = saved
            raise

    def run_history(self, external_history, **kwargs):
        Hext = np.asarray(external_history, dtype=float)
        if Hext.ndim == 2 and Hext.shape[1] == 3:
            Hext = np.repeat(Hext[:, None, :], self.operator.mesh.n_cells, axis=1)
        records = []
        for h in Hext:
            records.append(self.advance(h, **kwargs))
        mean_M = np.array([np.average(x["M"], axis=0,
                                      weights=self.operator.mesh.volumes)
                           for x in records])
        mean_H = np.array([np.average(x["H_internal"], axis=0,
                                      weights=self.operator.mesh.volumes)
                           for x in records])
        return dict(records=records, mean_M=mean_M, mean_H_internal=mean_H,
                    final_state=self.state.copy())

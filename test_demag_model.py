"""Fast analytic checks for the linear self-consistent sphere solver."""

import numpy as np

from magnetic_carousel import DemagSphereSolver, Params, sphere_quadrature


def main():
    H0 = np.array([0.0, 0.0, 1000.0])
    previous = np.inf
    for level in ("coarse", "medium", "fine"):
        p = Params(volume_quadrature=level)
        points, weights = sphere_quadrature(p.ball_R, level)
        solver = DemagSphereSolver(p.ball_R, 1000.0, level)
        sol = solver.solve(np.tile(H0, (solver.n_surface, 1)),
                           np.tile(H0, (len(points), 1)), points, weights)
        exact = 3.0*999.0/1002.0*H0*p.volume
        rel = np.linalg.norm(sol["moment"]-exact)/np.linalg.norm(exact)
        direction = np.linalg.norm(np.cross(sol["moment"], exact)) / (
            np.linalg.norm(sol["moment"])*np.linalg.norm(exact))
        print(f"{level:6s} panels={solver.n_surface:3d} volume={len(points):3d} "
              f"moment_rel={rel:.3e} direction_sin={direction:.3e}")
        assert rel < previous
        assert direction < 1e-12
        previous = rel
    assert previous < 5e-4


if __name__ == "__main__":
    main()

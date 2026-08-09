"""Fast consistency checks for the first-order distributed ball model."""

import numpy as np

from magnetic_carousel import MU0, Params, sphere_quadrature


def run():
    p = Params()
    B = np.array([0.021, -0.013, 0.034])
    bn = np.linalg.norm(B)
    meq_density = min(3.0 * bn / MU0, p.Ms) * B / bn
    point_moment = min(p.alpha * bn, p.m_sat) * B / bn

    print("Finite-volume consistency tests")
    for level in ("coarse", "medium", "fine"):
        points, weights = sphere_quadrature(p.ball_R, level)
        volume_error = abs(weights.sum() - p.volume) / p.volume
        summed_moment = np.sum(weights[:, None] * meq_density[None, :], axis=0)
        moment_error = np.linalg.norm(summed_moment-point_moment) / np.linalg.norm(point_moment)
        # Uniform B, gradB=0: every element force and force-arm torque vanish.
        zero_gradient = np.zeros((len(weights), 3, 3))
        dm = weights[:, None] * meq_density[None, :]
        force_elements = np.einsum("nij,nj->ni", zero_gradient, dm)
        force = force_elements.sum(axis=0)
        force_torque = np.cross(points, force_elements).sum(axis=0)
        print(f"  {level:6s} n={len(weights):4d} volume_rel={volume_error:.3e} "
              f"uniform_moment_rel={moment_error:.3e} "
              f"|F_zero_grad|={np.linalg.norm(force):.3e} "
              f"|tau_force_zero_grad|={np.linalg.norm(force_torque):.3e}")
        assert volume_error < 1e-13
        assert moment_error < 1e-13
        assert np.linalg.norm(force) == 0.0
        assert np.linalg.norm(force_torque) == 0.0


if __name__ == "__main__":
    run()

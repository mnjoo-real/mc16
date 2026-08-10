"""Checks for Step 5A orientation and diffusion scalings."""

import numpy as np

from diagnose_skin_effect import (quat_matrix, reconstruct_orientation,
                                  skin_values)


def main():
    t = np.linspace(0.0, 1.0, 1001)
    omega = np.tile((0.0, 0.0, 2.0), (len(t), 1))
    q = reconstruct_orientation(t, omega)
    R = quat_matrix(q[-1])
    angle = np.arctan2(R[1, 0], R[0, 0])
    assert abs(angle-2.0) < 1e-10
    assert np.linalg.norm(R.T@R-np.eye(3)) < 1e-12

    s = skin_values(50.0, 6e6, 100.0, .006, .004)
    assert abs(s["delta_over_a"]-np.sqrt(2.0/s["Pi1"])) < 1e-14
    assert abs(s["Omega_tau_lag"]-.2) < 1e-14
    print(f"orientation angle={angle:.12f} rad, orthogonality="
          f"{np.linalg.norm(R.T@R-np.eye(3)):.3e}, delta/a={s['delta_over_a']:.6f}")


if __name__ == "__main__":
    main()

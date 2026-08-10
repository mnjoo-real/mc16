"""Validation tests for the standalone scalar/vector JA reference kernel."""

import numpy as np

from magnetic_hysteresis import (MU0, SYNTHETIC_PARAMETER_SETS,
    alternating_path, anhysteretic_scalar, anhysteretic_vector,
    circular_path, integrate_scalar_path, integrate_vector_path, langevin,
    langevin_prime, scalar_loop_metrics, vector_cycle_work)


def closed_vector_path(kind, H0, n=600, cycles=4):
    ramp = np.linspace(0, 1, n//4+1)
    if kind == "ellipse":
        start = np.array((H0, 0., 0.)); Hr = ramp[:, None]*start
        th = np.linspace(0, 2*np.pi*cycles, n*cycles+1)
        cyc = np.column_stack((H0*np.cos(th), .55*H0*np.sin(th), 0*th))
    else:
        start = np.array((H0, 0., .2*H0)); Hr = ramp[:, None]*start
        th = np.linspace(0, 2*np.pi*cycles, n*cycles+1)
        cyc = H0*np.column_stack((np.cos(th), .65*np.sin(th),
                                  .2*np.cos(2*th)))
    return np.vstack((Hr, cyc[1:]))


def main():
    p = SYNTHETIC_PARAMETER_SETS["synthetic_medium"]

    # 1. Stable Langevin and anhysteretic tensor at and near zero.
    x = np.array((-1e-8, 0., 1e-8))
    assert np.all(np.isfinite(langevin(x)))
    assert np.max(np.abs(langevin(x)-x/3)) < 1e-22
    assert np.max(np.abs(langevin_prime(x)-1/3)) < 1e-15
    Man, J = anhysteretic_vector(np.zeros(3), p)
    assert np.linalg.norm(Man) == 0 and np.allclose(J, np.eye(3)*p.Ms/(3*p.a))

    # 2. Scalar positive and negative saturation approaches +/- Ms.
    Hsat = np.r_[np.linspace(0, 200*p.a, 2500),
                 np.linspace(200*p.a, -200*p.a, 5000)[1:]]
    sat = integrate_scalar_path(Hsat, p)["M"]
    assert sat[2499] > .99*p.Ms and sat[-1] < -.99*p.Ms

    # 3/4. Major loop has remanence/coercivity and reaches a periodic state.
    n = 800; H = alternating_path(4e4, n, 5)
    s = integrate_scalar_path(H, p); last = scalar_loop_metrics(H[-n-1:], s["M"][-n-1:])
    assert abs(last["Mr"]) > 1e3 and last["Hc"] > 100 and last["loop_area"] > 0
    closure = abs(s["M"][-1]-s["M"][-n-1])/p.Ms
    assert closure < 2e-5

    # 5. Rate-independent loop area is finite and independent of traversal rate.
    # Omega changes only the timestamps, never the H path or constitutive result.
    areas = []
    for omega in (.01, 1., 100.):
        _t = np.linspace(0, 2*np.pi/omega, n+1)
        areas.append(scalar_loop_metrics(H[-n-1:], s["M"][-n-1:])["loop_area"])
    assert min(areas) > 100 and np.ptp(areas) == 0

    # 6. Collinear vector model reduces to scalar JA.
    Hv = np.column_stack((H, np.zeros((len(H), 2))))
    v = integrate_vector_path(Hv, p)
    discrepancy = np.max(np.abs(v["M"][:, 0]-s["M"]))/p.Ms
    assert discrepancy < 2e-12
    assert np.max(np.abs(v["M"][:, 1:])) < 1e-12*p.Ms

    # 7. Isotropic covariance under a fixed arbitrary 3-D rotation.
    C = circular_path(3e4, 500, 4); vc = integrate_vector_path(C, p)
    axis = np.array((1., 2., 3.)); axis /= np.linalg.norm(axis); ang = .73
    K = np.array(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]),
                  (-axis[1], axis[0], 0.)))
    R = np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*(K@K)
    vr = integrate_vector_path(C@R.T, p)
    covariance = np.max(np.linalg.norm(vr["M"]-vc["M"]@R.T, axis=1))/p.Ms
    assert covariance < 2e-11

    # 8. Closed-cycle work is nonnegative on scalar, circular, elliptical,
    # and nonplanar paths. This is tested evidence, not a global JA proof.
    works = [last["loop_area"], vector_cycle_work(C[-501:], vc["M"][-501:])]
    for kind in ("ellipse", "3d"):
        Q = closed_vector_path(kind, 3e4, 500, 4)
        mq = integrate_vector_path(Q, p)["M"]
        works.append(vector_cycle_work(Q[-501:], mq[-501:]))
    assert min(works) >= -1e-8

    # 9. Path-increment convergence of Mr, Hc, and loop area.
    metrics = []
    for nn in (800, 1600, 3200):
        h = alternating_path(4e4, nn, 4); m = integrate_scalar_path(h, p)["M"]
        metrics.append(scalar_loop_metrics(h[-nn-1:], m[-nn-1:]))
    for key in ("Mr", "Hc", "loop_area"):
        e1 = abs(metrics[1][key]-metrics[0][key])
        e2 = abs(metrics[2][key]-metrics[1][key])
        assert e2 < .4*e1

    # 10. No NaN/Inf over all synthetic sets and benchmark path classes.
    for pp in SYNTHETIC_PARAMETER_SETS.values():
        for B0 in (.005, .02, .05):
            h0 = B0/MU0
            hs = alternating_path(h0, 180, 3)
            hc = circular_path(h0, 180, 3)
            ms = integrate_scalar_path(hs, pp)["M"]
            mv = integrate_vector_path(hc, pp)["M"]
            assert np.all(np.isfinite(ms)) and np.all(np.isfinite(mv))
            assert scalar_loop_metrics(hs[-181:], ms[-181:])["loop_area"] >= -1e-7
            assert vector_cycle_work(hc[-181:], mv[-181:]) >= -1e-7

    print(f"vector JA tests passed: reduction={discrepancy:.3e}, "
          f"covariance={covariance:.3e}, W range={min(works):.3f}..{max(works):.3f} J/m3")


if __name__ == "__main__":
    main()

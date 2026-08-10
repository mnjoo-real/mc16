"""Validation gates for the offline arbitrary-M hysteretic sphere solver."""

import numpy as np

from dataclasses import replace
from magnetic_carousel import (DemagSphereSolver, MU0, Params, PointSetFieldGrid,
    VolumeFieldGrid, distributed_force_torque, sphere_quadrature,
    static_magnetic_response)
from magnetic_hysteresis import (SYNTHETIC_PARAMETER_SETS,
                                 scalar_loop_metrics, vector_cycle_work)
from magnetic_hysteretic_sphere import (CuboidDemagOperator,
    HystereticDemagSphere, MATERIAL_MESH_LEVELS, SphereMaterialMesh)


def rotation(axis, angle):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    K = np.array(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]),
                  (-axis[1], axis[0], 0.)))
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def main():
    meshes = {q: SphereMaterialMesh.build(.006, q) for q in MATERIAL_MESH_LEVELS}
    ops = {q: CuboidDemagOperator(m) for q, m in meshes.items()}

    # 1. Uniform sphere mode is -M/3 at every cell and at every resolution.
    for q, op in ops.items():
        mean, rms, rel = op.uniform_validation((.2, -.3, .9), 2e5)
        assert rms/2e5 < 2e-12 and rel < 2e-12, (q, mean, rms, rel)

    # 2. Geometry and arbitrary-M operator rotate covariantly.
    op = ops["coarse"]; R = rotation((1, 2, 3), .61)
    rng = np.random.default_rng(8); M = rng.normal(size=(op.mesh.n_cells, 3))*1e4
    rotated = op.rotated(R)
    covariance = np.max(np.linalg.norm(rotated.field(M@R.T)-op.field(M)@R.T,
                                       axis=1))/np.max(np.linalg.norm(M, axis=1))
    assert covariance < 2e-12

    # 3. Linear chi sphere equals the analytic 3(mu_r-1)/(mu_r+2) response.
    H0 = np.array((0., 0., 1000.))
    for q, operator in ops.items():
        for mur in (10., 50., 100., 500.):
            Mlin = operator.solve_linear(H0, mur)
            exact = 3*(mur-1)/(mur+2)*H0
            assert np.linalg.norm(np.mean(Mlin, axis=0)-exact)/np.linalg.norm(exact) < 2e-10

    # 4. Same uniform linear limit as the independently validated Step-4B BEM.
    p0 = Params(); vp, vw = sphere_quadrature(p0.ball_R, "medium")
    bem = DemagSphereSolver(p0.ball_R, 100., "medium")
    b = bem.solve(np.tile(H0, (bem.n_surface, 1)), np.tile(H0, (len(vp), 1)), vp, vw)
    Mnew = ops["medium"].solve_linear(H0, 100.)
    mnew = np.sum(Mnew*meshes["medium"].volumes[:, None], axis=0)
    # Correct the reported moment for the voxelized represented volume.
    mnew *= p0.volume/meshes["medium"].represented_volume
    assert np.linalg.norm(mnew-b["moment"])/np.linalg.norm(b["moment"]) < 2e-3
    # Actual nonuniform FieldGrid force bridge (medium material mesh).
    pf = replace(Params(gap=.008), ball_model="point")
    centre = np.array((pf.r_mag, 0., 0.))
    fg = PointSetFieldGrid(pf, meshes["medium"].points,
                           cache_dir="volume_fieldcache",
                           cache_tag="hysteretic_cells", verbose=False)
    B, J = fg.sample(centre)
    Mf = ops["medium"].solve_linear(B/MU0, 100.)
    fnew, _ = distributed_force_torque(Mf, B, J, meshes["medium"].points,
                                        meshes["medium"].volumes)
    po = replace(pf, ball_model="volume_demag", volume_quadrature="medium",
                 demag_resolution="fine")
    vg = VolumeFieldGrid(po, verbose=False); obem = DemagSphereSolver(po.ball_R, 100., "fine")
    osg = PointSetFieldGrid(po, obem.surface_points, cache_dir="volume_fieldcache",
                            cache_tag="demag_surface", verbose=False)
    old = static_magnetic_response(po, centre, grid=vg, surface_grid=osg,
                                   demag_solver=obem)
    force_bridge = abs(fnew[2]-old["force"][2])/abs(old["force"][2])
    assert force_bridge < .12

    pja = SYNTHETIC_PARAMETER_SETS["synthetic_medium"]
    solver = HystereticDemagSphere(op, pja, tolerance=1e-7, max_iterations=80)
    state_before = solver.state.copy()
    a = solver.trial_increment((0, 0, 100), max_iterations=30)
    btrial = solver.trial_increment((0, 0, 100), max_iterations=80)
    # 5. Trials do not commit or accumulate state; iteration budget is irrelevant.
    assert np.array_equal(solver.state, state_before)
    assert np.max(np.abs(a["state"]-btrial["state"]))/pja.Ms < 2e-7

    # 6. Nonlinear residual reaches the requested tolerance.
    assert btrial["converged"] and btrial["residual"] < 1e-7
    assert btrial["residual_history"][-1] < btrial["residual_history"][0]

    # 7. Tolerance refinement changes M negligibly for a controlled increment.
    trial = [solver.trial_increment((0, 0, 100), tolerance=t, max_iterations=100)
             for t in (1e-4, 1e-6, 1e-8)]
    assert all(x["converged"] for x in trial)
    dm = np.max(np.abs(trial[-1]["M"]-trial[-2]["M"]))/pja.Ms
    assert dm < 2e-6

    # 8. The complete uniform JA+demag solve remains isotropic.
    s1 = HystereticDemagSphere(op, pja, tolerance=1e-6)
    s2 = HystereticDemagSphere(op.rotated(R), pja, tolerance=1e-6)
    path = np.array(((0, 0, 0), (0, 0, 100), (40, -30, 80), (-20, 10, -50.)))
    q1 = s1.run_history(path); q2 = s2.run_history(path@R.T)
    iso = np.max(np.linalg.norm(q2["mean_M"]-q1["mean_M"]@R.T, axis=1))/pja.Ms
    assert iso < 2e-9

    # 9/10. Periodic scalar excitation closes and dissipates nonnegative work.
    theta = np.linspace(0, 2*np.pi*3, 19)
    h = 1200*np.cos(theta); hp = np.column_stack((0*h, 0*h, h))
    cyc = HystereticDemagSphere(op, pja, tolerance=2e-6).run_history(hp)
    closure = np.linalg.norm(cyc["mean_M"][-1]-cyc["mean_M"][-7])/pja.Ms
    metrics = scalar_loop_metrics(h[-7:], cyc["mean_M"][-7:, 2])
    assert closure < 2e-3 and metrics["loop_area"] >= -1e-8

    # 11. A nonuniform zero-mean mode is stable under medium/fine refinement.
    energies = []
    for q in ("medium", "fine"):
        mesh, operator = meshes[q], ops[q]
        M = np.zeros((mesh.n_cells, 3)); M[:, 2] = 1e5*mesh.points[:, 2]/mesh.radius
        H = operator.field(M)
        energies.append(-.5*MU0*np.average(np.sum(M*H, axis=1), weights=mesh.volumes))
    assert abs(energies[1]-energies[0])/energies[1] < .03

    # 12. All synthetic sets remain finite and passive on a small rotating loop.
    th = np.linspace(0, 2*np.pi*2, 13)
    path = 500*np.column_stack((np.cos(th), np.sin(th), np.zeros_like(th)))
    losses = []
    for pp in SYNTHETIC_PARAMETER_SETS.values():
        ans = HystereticDemagSphere(op, pp, tolerance=2e-6).run_history(path)
        assert np.all(np.isfinite(ans["mean_M"]))
        losses.append(vector_cycle_work(path[-7:], ans["mean_M"][-7:]))
    assert min(losses) >= -1e-7

    print(f"hysteretic demag tests passed: covariance={covariance:.3e}, "
          f"isotropy={iso:.3e}, tolerance_dM={dm:.3e}, "
          f"force_bridge={force_bridge:.3e}, "
          f"W={min(losses):.6g}..{max(losses):.6g} J/m3")


if __name__ == "__main__":
    main()

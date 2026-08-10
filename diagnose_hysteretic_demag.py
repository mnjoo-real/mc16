"""Step 6B-2 offline diagnostics for the self-consistent hysteretic sphere."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hysteresis_scale import equilibrium_M
from magnetic_carousel import (MU0, DemagSphereSolver, Params, PointSetFieldGrid,
                               VolumeFieldGrid, distributed_force_torque,
                               sphere_quadrature, sphere_surface_quadrature,
                               static_magnetic_response)
from magnetic_hysteresis import (SYNTHETIC_PARAMETER_SETS, alternating_path,
    integrate_scalar_path, integrate_vector_path, scalar_loop_metrics,
    vector_cycle_work)
from magnetic_hysteretic_sphere import (CuboidDemagOperator,
    HystereticDemagSphere, MATERIAL_MESH_LEVELS, SphereMaterialMesh,
    field_from_cells, maxwell_stress_wrench)

OUT = Path("out")
CSV_PATH = OUT/"hysteretic_demag_diagnostic.csv"
PNG_PATH = OUT/"hysteretic_demag_diagnostic.png"
GAPS = (.003, .005, .008, .016)
OMEGA_BY_GAP = {.003: 2.0, .005: 7.8, .008: 16.0, .016: 10.0}


def base_row(section, **kw):
    row = dict(section=section, mesh="", n_cells=np.nan, parameter_set="",
               field_history="", gap_m=np.nan, phase_rad=np.nan,
               H_ext_A_m=np.nan, mean_H_int_A_m=np.nan, mean_M_A_m=np.nan,
               total_moment_Am2=np.nan, force_N=np.nan, Fz_N=np.nan,
               torque_Nm=np.nan, loop_loss_J_m3=np.nan, remanence_A_m=np.nan,
               coercivity_A_m=np.nan, angle_lag_deg=np.nan,
               spatial_H_rms_A_m=np.nan, spatial_M_rms_A_m=np.nan,
               iterations=np.nan, residual=np.nan, runtime_s=np.nan,
               represented_volume_m3=np.nan, volume_error=np.nan,
               surface_error_scale=np.nan, relative_error=np.nan,
               converged="", note="")
    row.update(kw)
    return row


def weighted_mean(x, weights):
    return np.average(x, axis=0, weights=weights)


def interpolate_keys(values, subdivisions=2):
    out = [values[0]]
    for a, b in zip(values[:-1], values[1:]):
        for j in range(1, subdivisions+1):
            out.append(a+(b-a)*j/subdivisions)
    return np.asarray(out)


def validation(rows):
    meshes, operators, uniform_curve, linear_curve, refinement = {}, {}, [], [], []
    H0 = np.array((0., 0., 1000.))
    for level in MATERIAL_MESH_LEVELS:
        mesh = SphereMaterialMesh.build(.006, level); meshes[level] = mesh
        op = CuboidDemagOperator(mesh); operators[level] = op
        mean, rms, rel = op.uniform_validation((0, 0, 1), 1e5)
        uniform_curve.append((mesh.n_cells, rms/1e5, op.uncorrected_uniform_rms))
        rows.append(base_row("uniform_demag", mesh=level, n_cells=mesh.n_cells,
            mean_H_int_A_m=mean[2], spatial_H_rms_A_m=rms,
            represented_volume_m3=mesh.represented_volume,
            volume_error=mesh.volume_error, surface_error_scale=mesh.surface_error_bound,
            relative_error=rel, runtime_s=op.build_seconds,
            note=f"target Hdemag_z=-33333.333 A/m; raw spectrum={op.raw_eigenvalue_range}"))
        for mur in (10., 50., 100., 500.):
            M = op.solve_linear(H0, mur); exact = 3*(mur-1)/(mur+2)*H0
            err = np.linalg.norm(weighted_mean(M, mesh.volumes)-exact)/np.linalg.norm(exact)
            linear_curve.append((mesh.n_cells, mur, err))
            rows.append(base_row("linear_sphere", mesh=level, n_cells=mesh.n_cells,
                parameter_set=f"mu_r={mur:g}", H_ext_A_m=np.linalg.norm(H0),
                mean_M_A_m=np.linalg.norm(weighted_mean(M, mesh.volumes)),
                relative_error=err, note="exact 3(mu_r-1)/(mu_r+2) Hext"))
        M = np.zeros((mesh.n_cells, 3)); M[:, 2] = 1e5*mesh.points[:, 2]/mesh.radius
        Hd = op.field(M)
        energy = -.5*MU0*np.average(np.sum(M*Hd, axis=1), weights=mesh.volumes)
        refinement.append((mesh.n_cells, energy))
        rows.append(base_row("nonuniform_refinement", mesh=level, n_cells=mesh.n_cells,
                             loop_loss_J_m3=energy,
                             spatial_H_rms_A_m=np.sqrt(np.average(np.sum(Hd**2, axis=1),
                                                                  weights=mesh.volumes)),
                             note="zero-mean Mz proportional to body z; energy diagnostic"))
    # Independent old BEM bridge at mu_r=100.
    p = Params(); vp, vw = sphere_quadrature(p.ball_R, "medium")
    bem = DemagSphereSolver(p.ball_R, 100., "medium")
    old = bem.solve(np.tile(H0, (bem.n_surface, 1)), np.tile(H0, (len(vp), 1)), vp, vw)
    rows.append(base_row("step4b_bridge", mesh="medium", n_cells=meshes["medium"].n_cells,
                         parameter_set="mu_r=100", total_moment_Am2=np.linalg.norm(old["moment"]),
                         relative_error=linear_curve[-6][2],
                         note="independent homogeneous linear surface-charge BEM"))
    return meshes, operators, uniform_curve, linear_curve, refinement


def uniform_hysteresis(rows, operator):
    p = SYNTHETIC_PARAMETER_SETS["synthetic_medium"]
    Hs = alternating_path(.015/MU0, 32, 3)
    bare = integrate_scalar_path(Hs, p)
    path = np.column_stack((0*Hs, 0*Hs, Hs))
    solver = HystereticDemagSphere(operator, p, tolerance=2e-6, max_iterations=100)
    t0 = time.perf_counter(); coupled = solver.run_history(path); runtime = time.perf_counter()-t0
    n = 33
    bare_metrics = scalar_loop_metrics(Hs[-n:], bare["M"][-n:])
    sphere_metrics = scalar_loop_metrics(Hs[-n:], coupled["mean_M"][-n:, 2])
    for name, metrics in (("intrinsic_bare_JA", bare_metrics),
                          ("whole_sphere_JA_demag", sphere_metrics)):
        rows.append(base_row("uniform_loop", mesh=operator.mesh.level,
            n_cells=operator.mesh.n_cells, parameter_set=p.label, field_history=name,
            H_ext_A_m=np.max(abs(Hs)), mean_M_A_m=np.max(abs(coupled["mean_M"][:, 2]))
            if "sphere" in name else np.max(abs(bare["M"])),
            remanence_A_m=metrics["Mr"], coercivity_A_m=metrics["Hc"],
            loop_loss_J_m3=metrics["loop_area"], runtime_s=runtime if "sphere" in name else 0,
            iterations=max(x["iterations"] for x in coupled["records"]) if "sphere" in name else 0,
            residual=max(x["residual"] for x in coupled["records"]) if "sphere" in name else 0,
            converged=True))
    return Hs[-n:], bare["M"][-n:], coupled["mean_M"][-n:, 2], coupled


def linear_carousel_bridge(rows, operator):
    data = []
    for gap in GAPS:
        p = replace(Params(gap=gap), ball_model="point")
        centre = np.array((p.r_mag, 0., 0.))
        grid = PointSetFieldGrid(p, operator.mesh.points,
                                 cache_dir="volume_fieldcache",
                                 cache_tag="hysteretic_cells", verbose=False)
        B, J = grid.sample(centre, 0.)
        M = operator.solve_linear(B/MU0, 100.)
        fvol, tvol = distributed_force_torque(M, B, J, operator.mesh.points,
                                               operator.mesh.volumes)
        sp, sw = sphere_surface_quadrature(1.35*p.ball_R, "coarse")
        sg = PointSetFieldGrid(p, sp, cache_dir="volume_fieldcache",
                               cache_tag="hysteretic_stress", verbose=False)
        Bs, _ = sg.sample(centre, 0.)
        fms, tms = maxwell_stress_wrench(sp, sw, Bs,
            field_from_cells(sp, operator.mesh, M))

        po = replace(p, ball_model="volume_demag", volume_quadrature="medium",
                     demag_resolution="fine")
        vg = VolumeFieldGrid(po, verbose=False)
        bem = DemagSphereSolver(po.ball_R, po.mu_r, "fine")
        bsg = PointSetFieldGrid(po, bem.surface_points,
                                cache_dir="volume_fieldcache",
                                cache_tag="demag_surface", verbose=False)
        old = static_magnetic_response(po, centre, grid=vg,
                                       surface_grid=bsg, demag_solver=bem)
        for model, force, torque in (("new_volume_force", fvol, tvol),
                                     ("new_Maxwell_stress", fms, tms),
                                     ("Step4B_surface_BEM", old["force"], old["torque"])):
            rows.append(base_row("linear_carousel_bridge", mesh=operator.mesh.level,
                n_cells=operator.mesh.n_cells, parameter_set="mu_r=100",
                field_history=model, gap_m=gap,
                total_moment_Am2=np.linalg.norm(np.sum(M*operator.mesh.volumes[:, None], axis=0))
                    if "new" in model else np.linalg.norm(old["moment"]),
                force_N=np.linalg.norm(force), Fz_N=force[2],
                torque_Nm=np.linalg.norm(torque), converged=True,
                relative_error=abs(force[2]-old["force"][2])/abs(old["force"][2])
                    if "new" in model else 0,
                note="zero-hysteresis actual FieldGrid bridge"))
        data.append((gap, fvol, fms, old["force"]))
    return data


def rotating_uniform(rows, operator):
    curves = {}
    # A controlled 2.5 mT-equivalent field keeps every synthetic set on a
    # tractable common benchmark; Carousel histories below use actual fields.
    H0 = 2000.; theta = np.linspace(0, 4*np.pi, 33)
    path = H0*np.column_stack((np.cos(theta), np.sin(theta), 0*theta))
    for name, p in SYNTHETIC_PARAMETER_SETS.items():
        solver = HystereticDemagSphere(operator, p, tolerance=3e-6, max_iterations=100)
        t0 = time.perf_counter(); ans = solver.run_history(path); runtime = time.perf_counter()-t0
        Hc, Mc = path[-17:], ans["mean_M"][-17:]
        loss = vector_cycle_work(Hc, Mc)
        cross = np.linalg.norm(np.cross(Hc, Mc), axis=1)
        dot = np.einsum("ij,ij->i", Hc, Mc)
        lag = np.degrees(np.max(np.arctan2(cross, dot)))
        torque = operator.mesh.represented_volume*MU0*np.mean(cross)
        curves[name] = (Mc[:, 0], Mc[:, 1])
        rows.append(base_row("rotating_uniform", mesh=operator.mesh.level,
            n_cells=operator.mesh.n_cells, parameter_set=name,
            field_history="two circular cycles", H_ext_A_m=H0,
            mean_M_A_m=np.mean(np.linalg.norm(Mc, axis=1)),
            total_moment_Am2=np.mean(np.linalg.norm(Mc, axis=1))*operator.mesh.represented_volume,
            torque_Nm=torque, loop_loss_J_m3=loss, angle_lag_deg=lag,
            iterations=max(x["iterations"] for x in ans["records"]),
            residual=max(x["residual"] for x in ans["records"]), runtime_s=runtime,
            converged=True))
    return curves


def tau_lag_history(H, dt, p=Params()):
    M = np.empty_like(H); M[0] = 0
    decay = np.exp(-dt/p.tau_lag)
    for i in range(len(H)-1):
        eq = equilibrium_M(H[i+1], p.chi_eff, p.Ms)
        M[i+1] = eq+(M[i]-eq)*decay
    return M


def carousel_case(gap, operator, parameter_name, rows):
    mesh = operator.mesh
    pfield = replace(Params(gap=gap), ball_model="point")
    centre = np.array((pfield.r_mag, 0., 0.))
    cell_grid = PointSetFieldGrid(pfield, mesh.points, cache_dir="volume_fieldcache",
                                  cache_tag="hysteretic_cells", verbose=False)
    stress_r = 1.35*mesh.radius
    sp, sw = sphere_surface_quadrature(stress_r, "coarse")
    surface_grid = PointSetFieldGrid(pfield, sp, cache_dir="volume_fieldcache",
                                     cache_tag="hysteretic_stress", verbose=False)
    center_grid = PointSetFieldGrid(pfield, np.zeros((1, 3)), cache_dir=None)
    phases = np.linspace(0, np.pi/3, 9)
    cell_B, center_B, center_J = [], [], []
    for ph in phases:
        b, _ = cell_grid.sample(centre, ph); cell_B.append(b)
        b0, j0 = center_grid.sample(centre, ph); center_B.append(b0[0]); center_J.append(j0[0])
    cell_B = np.asarray(cell_B); center_B = np.asarray(center_B); center_J = np.asarray(center_J)
    cycle_B = interpolate_keys(cell_B, 2); cycle_center_B = interpolate_keys(center_B, 2)
    cycle_center_J = interpolate_keys(center_J, 2)
    ramp = np.linspace(0, 1, 33)[:, None, None]*cycle_B[0]
    ramp_c = np.linspace(0, 1, 33)[:, None]*cycle_center_B[0]
    ramp_j = np.repeat(cycle_center_J[0][None], 33, axis=0)
    Bpath = np.concatenate((ramp, cycle_B[1:], cycle_B[1:]))
    Bcenter = np.concatenate((ramp_c, cycle_center_B[1:], cycle_center_B[1:]))
    Jcenter = np.concatenate((ramp_j, cycle_center_J[1:], cycle_center_J[1:]))
    pja = SYNTHETIC_PARAMETER_SETS[parameter_name]
    maxit = 60 if parameter_name == "synthetic_soft" else 120
    solver = HystereticDemagSphere(operator, pja, tolerance=2e-3,
                                   max_iterations=maxit,
                                   constitutive_increment_factor=8,
                                   use_least_squares_fallback=False)
    t0 = time.perf_counter(); failure = ""
    records = []
    for istep, B in enumerate(Bpath):
        try:
            records.append(solver.advance(B/MU0))
        except RuntimeError as exc:
            failure = f"step {istep}/{len(Bpath)-1}: {exc}"
            break
    runtime = time.perf_counter()-t0
    if failure:
        rows.append(base_row("carousel_history", mesh=mesh.level, n_cells=mesh.n_cells,
            parameter_set=parameter_name, field_history="ramp+two rotor field periods",
            gap_m=gap, runtime_s=runtime, iterations=solver.last_info["iterations"]
            if solver.last_info else 0, converged=False, note=failure))
        return None
    Mmean = np.array([weighted_mean(x["M"], mesh.volumes) for x in records])
    Hint = np.array([weighted_mean(x["H_internal"], mesh.volumes) for x in records])
    ncycle = len(cycle_B); last = slice(-ncycle, None)
    Hmean_ext = np.array([weighted_mean(x/MU0, mesh.volumes) for x in Bpath])
    loss = vector_cycle_work(Hmean_ext[last], Mmean[last])
    dt = (np.pi/3)/(len(cycle_B)-1)/OMEGA_BY_GAP[gap]
    Mtau = tau_lag_history(Bcenter/MU0, dt, pfield)
    bare = integrate_vector_path(Bcenter/MU0, pja)["M"]
    V = 4*np.pi*mesh.radius**3/3
    tau_force, bare_force, coupled_force = [], [], []
    tau_torque, bare_torque, coupled_torque = [], [], []
    key_indices = np.linspace(len(Bpath)-ncycle, len(Bpath)-1, 5, dtype=int)
    for idx in key_indices:
        mt = V*Mtau[idx]; mb = V*bare[idx]
        tau_force.append(Jcenter[idx]@mt); bare_force.append(Jcenter[idx]@mb)
        tau_torque.append(np.cross(mt, Bcenter[idx])); bare_torque.append(np.cross(mb, Bcenter[idx]))
        phase = (idx-(len(Bpath)-ncycle))/(ncycle-1)*np.pi/3
        Bs, _ = surface_grid.sample(centre, phase)
        Hm = field_from_cells(sp, mesh, records[idx]["M"])
        f, t = maxwell_stress_wrench(sp, sw, Bs, Hm)
        coupled_force.append(f); coupled_torque.append(t)
    series = {"tau_lag": (np.asarray(tau_force), np.asarray(tau_torque), Mtau),
              "bare_vector_JA": (np.asarray(bare_force), np.asarray(bare_torque), bare),
              "JA_demag_Maxwell": (np.asarray(coupled_force), np.asarray(coupled_torque), Mmean)}
    for model, (forces, torques, mags) in series.items():
        rows.append(base_row("carousel_history", mesh=mesh.level, n_cells=mesh.n_cells,
            parameter_set=parameter_name, field_history=model, gap_m=gap,
            H_ext_A_m=np.mean(np.linalg.norm(Hmean_ext[last], axis=1)),
            mean_H_int_A_m=np.mean(np.linalg.norm(Hint[last], axis=1)) if "demag" in model else np.nan,
            mean_M_A_m=np.mean(np.linalg.norm(mags[last], axis=1)),
            total_moment_Am2=V*np.mean(np.linalg.norm(mags[last], axis=1)),
            force_N=np.mean(np.linalg.norm(forces, axis=1)), Fz_N=np.mean(forces[:, 2]),
            torque_Nm=np.mean(np.linalg.norm(torques, axis=1)),
            loop_loss_J_m3=loss if "demag" in model else
                vector_cycle_work(Bcenter[last]/MU0, mags[last]),
            spatial_H_rms_A_m=np.mean([np.sqrt(np.mean(np.sum(
                (x["H_internal"]-weighted_mean(x["H_internal"], mesh.volumes))**2,
                axis=1))) for x in records[-ncycle:]]) if "demag" in model else np.nan,
            spatial_M_rms_A_m=np.mean([np.sqrt(np.mean(np.sum(
                (x["M"]-weighted_mean(x["M"], mesh.volumes))**2, axis=1)))
                for x in records[-ncycle:]]) if "demag" in model else np.nan,
            iterations=max(x["iterations"] for x in records) if "demag" in model else 0,
            residual=max(x["residual"] for x in records) if "demag" in model else 0,
            runtime_s=runtime if "demag" in model else 0, converged=True,
            note="actual FieldGrid cell history; two polarity periods after zero-field ramp"))
    return dict(gap=gap, parameter=parameter_name, Hext=Hmean_ext[last],
                M=Mmean[last], Hint=Hint[last], loss=loss,
                forces=series, runtime=runtime,
                iterations=max(x["iterations"] for x in records),
                residual=max(x["residual"] for x in records))


def performance_rows(rows, operators, uniform_runtime, carousel):
    for level, op in operators.items():
        n = op.mesh.n_cells
        rows.append(base_row("performance", mesh=level, n_cells=n,
            runtime_s=op.build_seconds, note=f"operator memory={op.memory_bytes/2**20:.3f} MiB"))
    uniform = next(x for x in rows if x["section"] == "uniform_loop"
                   and x["field_history"] == "whole_sphere_JA_demag")
    per_step = uniform["runtime_s"]/(32*3+32//4+1)
    estimate = 4*per_step/20e-6
    rows.append(base_row("performance", mesh="coarse",
        field_history="uniform coupled increment lower bound", runtime_s=per_step,
        note=f"naive RK4 estimate={estimate:.3e} wall-s/sim-s; nonuniform capped histories were slower and failed"))


def write_outputs(rows, validation_data, loop_data, rotating, carousel, linear_bridge):
    keys = list(rows[0])
    OUT.mkdir(exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    uniform_curve, linear_curve, refinement = validation_data
    H, bare, sphere = loop_data
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    a = ax.ravel()
    a[0].loglog([x[0] for x in uniform_curve],
                [max(x[1], 1e-18) for x in uniform_curve], "o-", label="projected")
    a[0].loglog([x[0] for x in uniform_curve], [x[2] for x in uniform_curve], "s--", label="raw cuboids")
    a[0].set(title="A  Uniform M: -M/3", xlabel="cells", ylabel="normalized RMS error"); a[0].legend(fontsize=7)
    for mur in (10., 50., 100., 500.):
        q = [x for x in linear_curve if x[1] == mur]
        a[1].semilogy([x[0] for x in q], [max(x[2], 1e-18) for x in q], "o-", label=f"mu_r={mur:g}")
    a[1].set(title="B  Linear sphere analytic limit", xlabel="cells", ylabel="moment relative error"); a[1].legend(fontsize=6)
    a[2].plot(H/1e3, bare/1e3, label="intrinsic JA")
    a[2].plot(H/1e3, sphere/1e3, label="whole sphere")
    a[2].set(title="C  Intrinsic vs external loop", xlabel="Hext [kA/m]", ylabel="M [kA/m]"); a[2].legend(fontsize=7)
    a[3].plot([x[0] for x in refinement], [x[1] for x in refinement], "o-")
    a[3].set(title="D  Nonuniform mesh refinement", xlabel="cells", ylabel="demag energy [J/m3]")
    for name, (mx, my) in rotating.items(): a[4].plot(mx/1e3, my/1e3, label=name.replace("synthetic_", ""))
    a[4].set(title="E  Rotating uniform field", xlabel="Mx [kA/m]", ylabel="My [kA/m]"); a[4].legend(fontsize=7)
    ok = [x for x in carousel if x]
    if ok:
        a[5].plot([x["gap"]*1e3 for x in ok], [x["loss"] for x in ok], "o-")
        a[5].set_yscale("log")
        gaps = sorted(set(x["gap"] for x in ok))
        for model in ("tau_lag", "bare_vector_JA", "JA_demag_Maxwell"):
            vals=[]; model_gaps=[]
            for g in gaps:
                candidates=[x for x in ok if x["gap"]==g and x["parameter"]=="synthetic_medium"]
                if not candidates: continue
                q=candidates[0]; model_gaps.append(g)
                vals.append(np.mean(np.linalg.norm(q["forces"][model][1], axis=1)))
            a[6].plot(np.array(model_gaps)*1e3, vals, "o-", label=model)
        a[6].set_yscale("log"); a[6].legend(fontsize=6)
        sens=[x for x in ok if x["gap"]==.008]
        a[7].bar([x["parameter"].replace("synthetic_", "") for x in sens], [x["loss"] for x in sens])
    a[5].set(title="F  Carousel-history loss", xlabel="gap [mm]", ylabel="J/m3/cycle")
    if not ok:
        for idx, label in ((1, "volume force"), (2, "Maxwell stress"), (3, "Step 4B")):
            a[6].plot([x[0]*1e3 for x in linear_bridge],
                      [-x[idx][2] for x in linear_bridge], "o-", label=label)
        a[6].set_yscale("log"); a[6].legend(fontsize=7)
    a[6].set(title="G  Force bridge / history torque", xlabel="gap [mm]", ylabel="force [N] or |tau| [N m]")
    a[7].set(title="H  8 mm parameter sensitivity", ylabel="J/m3/cycle")
    if not ok:
        a[5].text(.5, .5, "No converged histories\nunder diagnostic compute cap",
                  ha="center", va="center", transform=a[5].transAxes)
        a[7].text(.5, .5, "soft / medium / hard\nall nonconverged",
                  ha="center", va="center", transform=a[7].transAxes)
    for q in a: q.grid(alpha=.25)
    fig.suptitle("Step 6B-2: self-consistent distributed hysteretic sphere")
    fig.savefig(PNG_PATH, dpi=180); plt.close(fig)


def main():
    rows = []
    meshes, operators, uc, lc, refine = validation(rows)
    linear_bridge = linear_carousel_bridge(rows, operators["medium"])
    loop_data = uniform_hysteresis(rows, operators["coarse"])
    rotating = rotating_uniform(rows, operators["coarse"])
    carousel = []
    for gap in GAPS:
        print(f"Carousel history: gap={gap*1e3:g} mm medium", flush=True)
        carousel.append(carousel_case(gap, operators["coarse"], "synthetic_medium", rows))
    for name in ("synthetic_soft", "synthetic_hard"):
        print(f"Carousel sensitivity: gap=8 mm {name}", flush=True)
        carousel.append(carousel_case(.008, operators["coarse"], name, rows))
    performance_rows(rows, operators, 0, [x for x in carousel if x])
    write_outputs(rows, (uc, lc, refine), loop_data[:3], rotating, carousel,
                  linear_bridge)
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    print(f"wrote {PNG_PATH}")
    for x in carousel:
        if x:
            print(f"gap={x['gap']*1e3:g}mm {x['parameter']} W={x['loss']:.4g} "
                  f"itmax={x['iterations']} residual={x['residual']:.3g} runtime={x['runtime']:.2f}s")


if __name__ == "__main__":
    main()

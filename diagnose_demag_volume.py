"""Validate and compare the self-consistent linear permeable-sphere model."""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_carousel import (MU0, DemagSphereSolver, FieldGrid, Params,
                               PointSetFieldGrid, VolumeFieldGrid,
                               sphere_quadrature, static_magnetic_response)

OUT = Path("out")
LEVELS = ("coarse", "medium", "fine")
MU_VALUES = (10.0, 50.0, 100.0, 500.0, 1000.0)
BENCHMARKS = ((.003, 2.0), (.005, 7.8), (.008, 16.0), (.016, 10.0))


def uniform_validation(rows):
    H0 = np.array([0.0, 0.0, 1000.0])
    curves = {}
    demag_profile = None
    for level in LEVELS:
        p = Params(volume_quadrature=level)
        vp, vw = sphere_quadrature(p.ball_R, level)
        curves[level] = []
        for mur in MU_VALUES:
            t0 = time.perf_counter()
            solver = DemagSphereSolver(p.ball_R, mur, level)
            matrix_time = time.perf_counter()-t0
            Hs = np.tile(H0, (solver.n_surface, 1))
            Hv = np.tile(H0, (len(vp), 1))
            t0 = time.perf_counter()
            sol = solver.solve(Hs, Hv, vp, vw)
            solve_time = time.perf_counter()-t0
            exact_M = 3.0*(mur-1.0)/(mur+2.0)*H0
            exact_m = exact_M*p.volume
            err = np.linalg.norm(sol["moment"]-exact_m)/np.linalg.norm(exact_m)
            cosang = np.dot(sol["moment"], exact_m)/(
                np.linalg.norm(sol["moment"])*np.linalg.norm(exact_m))
            direction_deg = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            hd_mean = np.average(sol["Hdemag"], axis=0, weights=vw)
            target = -exact_M/3.0
            hd_rms = np.sqrt(np.average(np.sum((sol["Hdemag"]-target)**2, axis=1),
                                        weights=vw))
            curves[level].append(err)
            rows.append(dict(section="uniform", level=level, n_surface=solver.n_surface,
                             n_volume=len(vw), mu_r=mur, moment_rel_error=err,
                             direction_error_deg=direction_deg,
                             moment_per_volume_H0=sol["moment"][2]/p.volume/H0[2],
                             Hdemag_mean_x=hd_mean[0], Hdemag_mean_y=hd_mean[1],
                             Hdemag_mean_z=hd_mean[2], Hdemag_target_z=target[2],
                             Hdemag_rms_error=hd_rms,
                             matrix_build_s=matrix_time, solve_s=solve_time))
            if level == "fine" and mur == 100.0:
                demag_profile = (vp[:, 2]/p.ball_R, sol["Hdemag"][:, 2], target[2])
    return curves, demag_profile


def carousel_case(gap, omega, volume_level="medium", demag_level=None):
    demag_level = demag_level or volume_level
    base = Params(gap=gap, omega=omega, volume_quadrature=volume_level,
                  demag_resolution=demag_level)
    centre = np.array([base.r_mag, 0.0, 0.0])
    point_p = replace(base, ball_model="point")
    point = static_magnetic_response(point_p, centre,
                                     grid=FieldGrid(point_p, verbose=False))
    volume_p = replace(base, ball_model="volume_independent")
    vg = VolumeFieldGrid(volume_p, verbose=False)
    independent = static_magnetic_response(volume_p, centre, grid=vg)
    demag_p = replace(base, ball_model="volume_demag")
    solver = DemagSphereSolver(base.ball_R, base.mu_r, demag_level)
    sg = PointSetFieldGrid(demag_p, solver.surface_points,
                           cache_dir="volume_fieldcache", cache_tag="demag_surface",
                           verbose=True)
    demag = static_magnetic_response(demag_p, centre, grid=vg,
                                     surface_grid=sg, demag_solver=solver)
    return base, vg, point, independent, demag


def response_row(section, base, level, model, response, point, vg=None):
    out = dict(section=section, level=level, gap_m=base.gap, omega_rad_s=base.omega,
               model=model, moment_Am2=np.linalg.norm(response["moment"]),
               Fx_N=response["force"][0], Fy_N=response["force"][1],
               Fz_N=response["force"][2], Fnorm_N=np.linalg.norm(response["force"]),
               torque_Nm=np.linalg.norm(response["torque"]),
               moment_ratio_point=np.linalg.norm(response["moment"])/np.linalg.norm(point["moment"]),
               Fz_ratio_point=response["force"][2]/point["force"][2],
               force_ratio_point=np.linalg.norm(response["force"])/np.linalg.norm(point["force"]),
               torque_ratio_point=(np.linalg.norm(response["torque"])/
                                   max(np.linalg.norm(point["torque"]), 1e-300)))
    if "M" in response:
        sat = np.linalg.norm(response["M"], axis=1)/base.Ms
        out.update(max_M_over_Ms=np.max(sat),
                   mean_M_over_Ms=np.average(sat, weights=vg.weights),
                   volume_fraction_over_Ms=np.average(sat > 1.0, weights=vg.weights))
    return out


def benchmarks_and_convergence(rows):
    benchmark_data = []
    gap3_distribution = None
    for gap, omega in BENCHMARKS:
        base, vg, point, independent, demag = carousel_case(gap, omega, "medium", "fine")
        for model, response in (("point", point),
                                ("volume_independent", independent),
                                ("volume_demag", demag)):
            rows.append(response_row("benchmark", base, "medium/fine", model,
                                     response, point, vg))
        benchmark_data.append((gap, point, independent, demag))
        if gap == .003:
            gap3_distribution = np.linalg.norm(demag["M"], axis=1)/base.Ms

    convergence = []
    for level in ("medium", "fine", "very_fine"):
        base, vg, point, independent, demag = carousel_case(.003, 2.0, "fine", level)
        row = response_row("convergence", base, f"fine/{level}", "volume_demag",
                           demag, point, vg)
        row.update(n_surface=DemagSphereSolver(base.ball_R, base.mu_r, level).n_surface,
                   n_volume=vg.n_elem)
        rows.append(row)
        convergence.append((level, DemagSphereSolver(base.ball_R, base.mu_r,
                                                       level).n_surface, demag))
    return benchmark_data, convergence, gap3_distribution


def performance(rows):
    estimates = []
    H0 = np.array([0.0, 0.0, 1000.0])
    for level in LEVELS:
        p = Params(volume_quadrature=level)
        vp, vw = sphere_quadrature(p.ball_R, level)
        t0 = time.perf_counter()
        solver = DemagSphereSolver(p.ball_R, p.mu_r, level)
        build = time.perf_counter()-t0
        Hs = np.tile(H0, (solver.n_surface, 1)); Hv = np.tile(H0, (len(vp), 1))
        solver.solve(Hs, Hv, vp, vw)
        repeats = 5 if level != "fine" else 3
        t0 = time.perf_counter()
        for _ in range(repeats):
            solver.solve(Hs, Hv, vp, vw)
        solve = (time.perf_counter()-t0)/repeats
        rk4_cost = solve*4.0/2.0e-5
        rows.append(dict(section="performance", level=level,
                         n_surface=solver.n_surface, n_volume=len(vw),
                         matrix_build_s=build, solve_s=solve,
                         estimated_RK4_s_per_sim_s=rk4_cost))
        estimates.append((level, build, solve, rk4_cost))
    return estimates


def write_csv(rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    OUT.mkdir(exist_ok=True)
    with (OUT/"demag_volume_diagnostic.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def plot(curves, demag_profile, benchmark_data, convergence, distribution):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for level in LEVELS:
        ax[0, 0].loglog(MU_VALUES, curves[level], "o-", label=level)
    ax[0, 0].set(xlabel=r"$\mu_r$", ylabel="relative total-moment error",
                 title="A. Uniform-field analytic validation")
    ax[0, 0].grid(True, which="both", alpha=.3); ax[0, 0].legend()

    z, hd, target = demag_profile
    ax[0, 1].scatter(z, hd, s=10, alpha=.55, label="numeric")
    ax[0, 1].axhline(target, color="k", ls="--", label=r"exact $-M/3$")
    ax[0, 1].set(xlabel="z/R", ylabel=r"$H_{demag,z}$ [A/m]",
                 title=r"B. Internal demag field ($\mu_r=100$)")
    ax[0, 1].grid(alpha=.3); ax[0, 1].legend()

    gaps = np.array([x[0] for x in benchmark_data])*1e3
    for idx, label in ((1, "point"), (2, "independent"), (3, "demag")):
        ax[0, 2].plot(gaps, [-x[idx]["force"][2] for x in benchmark_data],
                      "o-", label=label)
        ax[1, 0].plot(gaps, [np.linalg.norm(x[idx]["moment"]) for x in benchmark_data],
                      "o-", label=label)
    ax[0, 2].set(xlabel="gap [mm]", ylabel=r"downward force $-F_z$ [N]",
                 title="C. Fixed-position force comparison")
    ax[0, 2].set_yscale("log"); ax[0, 2].grid(True, which="both", alpha=.3); ax[0, 2].legend()
    ax[1, 0].set(xlabel="gap [mm]", ylabel=r"$|m|$ [A m$^2$]",
                 title="D. Total moment comparison")
    ax[1, 0].set_yscale("log"); ax[1, 0].grid(True, which="both", alpha=.3); ax[1, 0].legend()

    ax[1, 1].hist(distribution, bins=18, color="#4C78A8", alpha=.85)
    ax[1, 1].axvline(1, color="r", ls="--")
    ax[1, 1].set(xlabel=r"$|M|/M_s$", ylabel="element count",
                 title="E. 3 mm demag local magnetization")
    n = [x[1] for x in convergence]
    fz = [-x[2]["force"][2] for x in convergence]
    moment = [np.linalg.norm(x[2]["moment"]) for x in convergence]
    a2 = ax[1, 2].twinx()
    ax[1, 2].plot(n, fz, "o-", color="#E45756", label=r"$-F_z$")
    a2.plot(n, moment, "s--", color="#54A24B", label=r"$|m|$")
    ax[1, 2].set(xlabel="surface panels (768 volume elements)", ylabel=r"$-F_z$ [N]",
                 title="F. 3 mm discretization convergence")
    a2.set_ylabel(r"$|m|$ [A m$^2$]")
    ax[1, 2].grid(alpha=.3)
    lines = ax[1, 2].lines+a2.lines
    ax[1, 2].legend(lines, [x.get_label() for x in lines])
    fig.suptitle("Self-consistent linear demagnetization reference", fontsize=15)
    fig.savefig(OUT/"demag_volume_diagnostic.png", dpi=180)


def main():
    rows = []
    curves, demag_profile = uniform_validation(rows)
    benchmark_data, convergence, distribution = benchmarks_and_convergence(rows)
    estimates = performance(rows)
    write_csv(rows)
    plot(curves, demag_profile, benchmark_data, convergence, distribution)
    print(f"wrote {OUT/'demag_volume_diagnostic.csv'}")
    print(f"wrote {OUT/'demag_volume_diagnostic.png'}")
    for gap, point, independent, demag in benchmark_data:
        print(f"gap={gap*1e3:4.0f} mm  Fz point={point['force'][2]:+.6f}  "
              f"ind={independent['force'][2]:+.6f}  demag={demag['force'][2]:+.6f}")
    for level, build, solve, rk4 in estimates:
        print(f"{level:6s}: build={build:.4g}s solve={solve:.4g}s "
              f"estimated RK4={rk4:.1f}s/sim-s")


if __name__ == "__main__":
    main()

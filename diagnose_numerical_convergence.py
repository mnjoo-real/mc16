"""Numerical convergence audit for the existing Magnetic Carousel model.

No production defaults or physical equations are changed.  Field grids bypass
the cache, and all dynamical comparisons use the existing fixed-step RK4
integrator with identical initial conditions and averaging windows.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_carousel import G_ACC, FieldGrid, Params, simulate


DEFAULT_DS = Params().grid_ds
GRID_LEVELS = (("coarse", 2 * DEFAULT_DS),
               ("default", DEFAULT_DS),
               ("fine", 0.5 * DEFAULT_DS))
GAPS = (0.003, 0.005, 0.008, 0.016)
N_PHI = 3072

DEFAULT_DT = 2.0e-5
DT_LEVELS = (("2dt", 2 * DEFAULT_DT),
             ("dt", DEFAULT_DT),
             ("dt/2", DEFAULT_DT / 2),
             ("dt/4", DEFAULT_DT / 4))
T_END = 1.3
OUTPUT_INTERVAL = 1.0e-3
CASES = (
    ("A_locked", 0.003, 2.0),
    ("B_transition", 0.005, 7.0),
    ("C_retrograde", 0.008, 16.0),
    ("D_weak_field", 0.016, 10.0),
)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def rel_errors(value, reference):
    diff = np.asarray(value) - np.asarray(reference)
    return (rms(diff) / max(rms(reference), 1e-300),
            float(np.max(np.abs(diff))) / max(float(np.max(np.abs(reference))), 1e-300))


def full_jacobian(q):
    """Reconstruct J[i,j]=dB_j/dx_i exactly as _field_lab does."""
    j = np.empty((len(q), 3, 3))
    dx_bx, dy_by, dx_by, dx_bz, dy_bz = (q[:, i] for i in range(3, 8))
    j[:, 0, 0], j[:, 0, 1], j[:, 0, 2] = dx_bx, dx_by, dx_bz
    j[:, 1, 0], j[:, 1, 1], j[:, 1, 2] = dx_by, dy_by, dy_bz
    j[:, 2, 0], j[:, 2, 1], j[:, 2, 2] = dx_bz, dy_bz, -(dx_bx + dy_by)
    return j


def cylindrical_b(phi, q):
    c, s = np.cos(phi), np.sin(phi)
    bx, by, bz = q[:, 0], q[:, 1], q[:, 2]
    br = c * bx + s * by
    bp = -s * bx + c * by
    b = np.column_stack((br, bp, bz))
    bm = np.linalg.norm(b, axis=1)
    return b, bm, bm**2


def representative_moment(phi, q_ref, p):
    """A reproducible lagged moment at phi=pi/(2N), based on the fine field."""
    target = np.pi / (2 * p.n_mag)
    i = int(round(target / (2 * np.pi) * len(phi))) % len(phi)
    b = q_ref[i, :3]
    bn = np.linalg.norm(b)
    m0 = min(p.alpha * bn, p.m_sat) * b / bn
    axis = np.array([np.cos(phi[i]), np.sin(phi[i]), 0.0])
    angle = np.deg2rad(15.0)
    # Rodrigues rotation about the local radial direction.
    m = (m0 * np.cos(angle) + np.cross(axis, m0) * np.sin(angle)
         + axis * np.dot(axis, m0) * (1 - np.cos(angle)))
    return i, m


def field_convergence(rows):
    grids = {}
    summaries = []
    print("\nFIELD-GRID CONVERGENCE")
    print("gap level       ds[mm]    n     dx[mm]  Bvec_RMS Bvec_max "
          "Bmag_RMS B2_RMS   grad_RMS grad_max  F_rel    tau_rel")
    for gap in GAPS:
        samples = {}
        for level, ds in GRID_LEVELS:
            p = Params(gap=gap, grid_ds=ds)
            grid = FieldGrid(p, cache_dir=None, verbose=True)
            grids[(gap, level)] = grid
            phi, q = grid.sample_circle(p.r_mag, N_PHI)
            b, bm, b2 = cylindrical_b(phi, q)
            samples[level] = (p, grid, phi, q, b, bm, b2, full_jacobian(q))

        p_ref, _, phi, q_ref, b_ref, bm_ref, b2_ref, j_ref = samples["fine"]
        state_i, moment = representative_moment(phi, q_ref, p_ref)
        force_ref = j_ref[state_i] @ moment
        torque_ref = np.cross(moment, q_ref[state_i, :3])

        for level, ds in GRID_LEVELS:
            p, grid, _, q, b, bm, b2, jac = samples[level]
            be_rms, be_max = rel_errors(b, b_ref)
            bm_rms, bm_max = rel_errors(bm, bm_ref)
            b2_rms, b2_max = rel_errors(b2, b2_ref)
            ge_rms, ge_max = rel_errors(jac, j_ref)
            force = jac[state_i] @ moment
            torque = np.cross(moment, q[state_i, :3])
            f_rel = np.linalg.norm(force-force_ref) / max(np.linalg.norm(force_ref), 1e-300)
            t_rel = np.linalg.norm(torque-torque_ref) / max(np.linalg.norm(torque_ref), 1e-300)
            item = dict(record="grid", gap_mm=gap*1e3, grid_level=level,
                        grid_ds=ds, grid_n=len(grid.xs), grid_dx=grid.dx,
                        Bvec_rms_rel=be_rms, Bvec_max_rel=be_max,
                        Bmag_rms_rel=bm_rms, Bmag_max_rel=bm_max,
                        B2_rms_rel=b2_rms, B2_max_rel=b2_max,
                        grad_rms_rel=ge_rms, grad_max_rel=ge_max,
                        F_rel=f_rel, tau_rel=t_rel,
                        F_norm=float(np.linalg.norm(force)),
                        tau_norm=float(np.linalg.norm(torque)))
            rows.append(item)
            summaries.append(item)
            print(f"{gap*1e3:3.0f} {level:<8} {ds*1e3:8.3f} {len(grid.xs):5d} "
                  f"{grid.dx*1e3:8.3f} {be_rms:9.2e} {be_max:9.2e} "
                  f"{bm_rms:8.2e} {b2_rms:8.2e} {ge_rms:9.2e} {ge_max:9.2e} "
                  f"{f_rel:8.2e} {t_rel:8.2e}")
    return grids, summaries


def classify(ratio):
    if not np.isfinite(ratio):
        return "other/slipping"
    return "locked/prograde" if ratio >= 0.0 else "retrograde"


def run_observables(p, grid, dt):
    stride = max(1, int(round(OUTPUT_INTERVAL / dt)))
    r = simulate(p, t_end=T_END, dt=dt, stride=stride, grid=grid)
    sl = slice(len(r.t)//2, None)
    omega_ball = r.orbital_rate(frac=0.5)
    ratio = omega_ball / p.omega
    radial = (r.w[:, 0]*r.x + r.w[:, 1]*r.y) / np.maximum(r.r, 1e-12)
    return dict(ratio=float(ratio), vtan_mean=float(np.mean(r.v_tan[sl])),
                omega_ball=float(omega_ball), spin_radial=float(np.mean(radial[sl])),
                slip_mean=float(np.mean(r.slip[sl])),
                Fz_mean=float(np.mean(r.Fmag[sl, 2])),
                N_mean=float(np.mean(r.Nload[sl])),
                tau_mean=float(np.mean(np.linalg.norm(r.Tmag[sl], axis=1))),
                final_x=float(r.x[-1]), final_y=float(r.y[-1]),
                accumulated_angle=float(r.phi[-1]-r.phi[0]),
                regime=classify(ratio))


def timestep_convergence(rows, grids):
    results = {}
    print("\nTIMESTEP CONVERGENCE")
    print("case level dt[us] ratio       vtan[mm/s] Omega_ball spin_rad "
          "slip[mm/s] Fz[N]       N[N]        |tau|[Nm]   angle[rad] regime")
    for case, gap, omega in CASES:
        grid = grids[(gap, "default")]
        p = Params(gap=gap, omega=omega)
        for level, dt in DT_LEVELS:
            obs = run_observables(p, grid, dt)
            item = dict(record="timestep", case=case, gap_mm=gap*1e3,
                        omega=omega, dt_level=level, dt=dt, **obs)
            rows.append(item)
            results[(case, level)] = item
            print(f"{case:<12} {level:<4} {dt*1e6:6.1f} {obs['ratio']:+10.6f} "
                  f"{obs['vtan_mean']*1e3:+11.4f} {obs['omega_ball']:+10.5f} "
                  f"{obs['spin_radial']:+9.4f} {obs['slip_mean']*1e3:10.4f} "
                  f"{obs['Fz_mean']:+10.3e} {obs['N_mean']:10.3e} "
                  f"{obs['tau_mean']:10.3e} {obs['accumulated_angle']:+10.4f} "
                  f"{obs['regime']}")

    print("\nTIMESTEP ERRORS RELATIVE TO dt/4")
    print("case level ratio_abs ratio_rel vtan_rel Omega_rel spin_rel slip_rel "
          "Fz_rel N_rel tau_rel angle_rel regime_change")
    keys = ("vtan_mean", "omega_ball", "spin_radial", "slip_mean",
            "Fz_mean", "N_mean", "tau_mean", "accumulated_angle")
    for case, _, _ in CASES:
        ref = results[(case, "dt/4")]
        for level, _ in DT_LEVELS:
            item = results[(case, level)]
            item["ratio_abs_error"] = abs(item["ratio"]-ref["ratio"])
            item["ratio_rel_error"] = item["ratio_abs_error"] / max(abs(ref["ratio"]), 1e-12)
            for key in keys:
                item[key+"_rel_error"] = abs(item[key]-ref[key]) / max(abs(ref[key]), 1e-12)
            item["regime_change"] = item["regime"] != ref["regime"]
            print(f"{case:<12} {level:<4} {item['ratio_abs_error']:9.2e} "
                  f"{item['ratio_rel_error']:9.2e} {item['vtan_mean_rel_error']:8.2e} "
                  f"{item['omega_ball_rel_error']:9.2e} {item['spin_radial_rel_error']:8.2e} "
                  f"{item['slip_mean_rel_error']:8.2e} {item['Fz_mean_rel_error']:8.2e} "
                  f"{item['N_mean_rel_error']:8.2e} {item['tau_mean_rel_error']:8.2e} "
                  f"{item['accumulated_angle_rel_error']:9.2e} {item['regime_change']}")
    return results


def critical_speed(grid, gap, dt, low, high, iterations=7):
    """From-rest sign boundary, maintaining a prograde low and retrograde high."""
    history = []
    for omega in (low, high):
        obs = run_observables(Params(gap=gap, omega=omega, grid_ds=grid.p.grid_ds), grid, dt)
        history.append((omega, obs["ratio"]))
    if history[0][1] < 0 or history[1][1] >= 0:
        return np.nan, history, "invalid bracket"
    for _ in range(iterations):
        mid = 0.5 * (low+high)
        obs = run_observables(Params(gap=gap, omega=mid, grid_ds=grid.p.grid_ds), grid, dt)
        history.append((mid, obs["ratio"]))
        if obs["ratio"] >= 0:
            low = mid
        else:
            high = mid
    return 0.5*(low+high), history, "ok"


def critical_convergence(rows, grids):
    output = []
    print("\nFROM-REST CRITICAL-SPEED CONVERGENCE")
    print("gap grid    dt[us] omega_crit bracket_width status")
    for gap, bracket in ((0.005, (6.0, 8.0)), (0.008, (3.0, 4.0))):
        for level, dt in DT_LEVELS[1:]:
            crit, history, status = critical_speed(grids[(gap, "default")], gap, dt, *bracket)
            item = dict(record="critical", gap_mm=gap*1e3, grid_level="default",
                        dt_level=level, dt=dt, omega_crit=crit,
                        bracket_width=(bracket[1]-bracket[0])/2**7, status=status)
            rows.append(item); output.append(item)
            print(f"{gap*1e3:3.0f} default {dt*1e6:7.1f} {crit:10.5f} "
                  f"{item['bracket_width']:13.5f} {status}")
        if np.isclose(gap, 0.005):
            crit, history, status = critical_speed(grids[(gap, "fine")], gap,
                                                    DEFAULT_DT, *bracket)
            item = dict(record="critical", gap_mm=gap*1e3, grid_level="fine",
                        dt_level="dt", dt=DEFAULT_DT, omega_crit=crit,
                        bracket_width=(bracket[1]-bracket[0])/2**7, status=status)
            rows.append(item); output.append(item)
            print(f"{gap*1e3:3.0f} fine    {DEFAULT_DT*1e6:7.1f} {crit:10.5f} "
                  f"{item['bracket_width']:13.5f} {status}")
    for gap in (0.005, 0.008):
        fine = next(x for x in output if x["gap_mm"] == gap*1e3
                    and x["grid_level"] == "default" and x["dt_level"] == "dt/4")
        for item in output:
            if item["gap_mm"] == gap*1e3 and item["grid_level"] == "default":
                item["omega_crit_shift"] = item["omega_crit"]-fine["omega_crit"]
    return output


def timescale_rows(rows):
    output = []
    print("\nTIMESCALE RESOLUTION")
    print("omega dt[us] dt/tau dt*Omega6 dt*Omega18 steps/tau steps/cycle6 steps/cycle18")
    for _, _, omega in CASES:
        for level, dt in DT_LEVELS:
            p = Params(omega=omega)
            item = dict(record="timescale", omega=omega, dt_level=level, dt=dt,
                        dt_tau=dt/p.tau_lag, dt_Omega6=dt*6*omega,
                        dt_Omega18=dt*18*omega, steps_tau=p.tau_lag/dt,
                        steps_cycle6=2*np.pi/(6*omega*dt),
                        steps_cycle18=2*np.pi/(18*omega*dt))
            rows.append(item); output.append(item)
            print(f"{omega:5.1f} {dt*1e6:6.1f} {item['dt_tau']:8.3e} "
                  f"{item['dt_Omega6']:10.3e} {item['dt_Omega18']:11.3e} "
                  f"{item['steps_tau']:9.1f} {item['steps_cycle6']:12.1f} "
                  f"{item['steps_cycle18']:13.1f}")
    return output


def write_csv(rows, path):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def make_figure(grid_rows, time_results, critical_rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes[0, 0]
    for gap in GAPS:
        data = [x for x in grid_rows if x["gap_mm"] == gap*1e3]
        ax.loglog([x["grid_ds"]*1e3 for x in data], [x["Bvec_rms_rel"] for x in data],
                  "o-", label=f"B, {gap*1e3:g} mm")
        ax.loglog([x["grid_ds"]*1e3 for x in data], [x["grad_rms_rel"] for x in data],
                  "s--", label=f"grad, {gap*1e3:g} mm")
    ax.set(xlabel="grid spacing [mm]", ylabel="relative RMS error", title="A. Field convergence")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0, 1]
    for gap in GAPS:
        data = [x for x in grid_rows if x["gap_mm"] == gap*1e3]
        ax.loglog([x["grid_ds"]*1e3 for x in data], [x["F_rel"] for x in data], "o-",
                  label=f"F, {gap*1e3:g} mm")
        ax.loglog([x["grid_ds"]*1e3 for x in data], [x["tau_rel"] for x in data], "s--",
                  label=f"tau, {gap*1e3:g} mm")
    ax.set(xlabel="grid spacing [mm]", ylabel="relative error", title="B. Force/torque convergence")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[0, 2]
    for case, _, _ in CASES:
        data = [time_results[(case, level)] for level, _ in DT_LEVELS]
        ax.semilogx([x["dt"]*1e6 for x in data], [x["ratio"] for x in data], "o-", label=case)
    ax.axhline(0, color="k", lw=0.7)
    ax.set(xlabel=r"$dt$ [$\mu$s]", ylabel="motion ratio", title="C. Motion-ratio convergence")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for gap in (5.0, 8.0):
        data = [x for x in critical_rows if x["gap_mm"] == gap and x["grid_level"] == "default"]
        ax.semilogx([x["dt"]*1e6 for x in data], [x["omega_crit"] for x in data], "o-",
                    label=f"gap {gap:g} mm")
    ax.set(xlabel=r"$dt$ [$\mu$s]", ylabel=r"$\omega_{crit}$ [rad/s]",
           title="D. From-rest critical speed")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for case, _, _ in CASES:
        data = [time_results[(case, level)] for level, _ in DT_LEVELS]
        ax.loglog([x["dt"]*1e6 for x in data],
                  [max(x["ratio_abs_error"], 1e-12) for x in data], "o-", label=case)
    ax.set(xlabel=r"$dt$ [$\mu$s]", ylabel="absolute ratio error", title="E. Ratio error vs dt")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    dts = np.array([x[1] for x in DT_LEVELS])
    for omega in (2, 7, 10, 16):
        ax.loglog(dts*1e6, 2*np.pi/(18*omega*dts), "o-", label=f"n=18, omega={omega}")
    ax.set(xlabel=r"$dt$ [$\mu$s]", ylabel="steps per n=18 cycle",
           title="F. Shortest field-cycle resolution")
    ax.legend(fontsize=7)

    for ax in axes.flat:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("out", exist_ok=True)
    rows = []
    p = Params()
    L = p.r_mag+p.grid_pad
    n = int(2*L/p.grid_ds)+1
    print("NUMERICAL CONVERGENCE DIAGNOSTIC (model/defaults unchanged)")
    print(f"default field domain=[{-L:.6f},{L:.6f}] m in x,y")
    print(f"default requested ds={p.grid_ds:.6e} m n={n}")
    print(f"default simulation dt={DEFAULT_DT:.6e} s tau_lag={p.tau_lag:.6e} s")
    print("field cache bypassed for all grid comparisons")
    grids, grid_rows = field_convergence(rows)
    time_results = timestep_convergence(rows, grids)
    critical_rows = critical_convergence(rows, grids)
    timescale_rows(rows)
    csv_path = "out/numerical_convergence.csv"
    fig_path = "out/numerical_convergence.png"
    write_csv(rows, csv_path)
    make_figure(grid_rows, time_results, critical_rows, fig_path)
    print(f"\nsaved {csv_path}")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()

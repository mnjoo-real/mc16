"""Diagnostics for the first-order finite-volume Magnetic Carousel ball model."""

from __future__ import annotations

import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_carousel import (G_ACC, MU0, FieldGrid, Params, VolumeFieldGrid,
                               _field_lab, simulate, sphere_quadrature)


GAPS = (0.003, 0.005, 0.008, 0.010, 0.016)
OMEGAS = (2.0, 5.0, 10.0, 16.0)
LEVELS = ("coarse", "medium", "fine")
DT, T_END, STRIDE = 2.0e-5, 1.3, 50
VOLUME_DT, VOLUME_T_END = 5.0e-5, 1.0


def weighted_rms(values, weights):
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def jacobian_from_field(BJ):
    q = BJ
    return np.array([[q[3], q[5], q[6]],
                     [q[5], q[4], q[7]],
                     [q[6], q[7], -(q[3]+q[4])]])


def sample_volume(vg, x, y, angle=0.0):
    B = np.empty((vg.n_elem, 3))
    J = np.empty((vg.n_elem, 3, 3))
    for i, r in enumerate(vg.points):
        bi, ji = _field_lab(vg.F[vg.z_index[i]], vg.x0, vg.y0, vg.dx, vg.dy,
                            x+r[0], y+r[1], angle)
        B[i], J[i] = bi, ji
    return B, J


def point_sample(grid, x, y, angle=0.0):
    return _field_lab(grid.F, grid.x0, grid.y0, grid.dx, grid.dy, x, y, angle)


def rotate_about_axis(vectors, axis, angle):
    vectors = np.asarray(vectors)
    axis = np.asarray(axis)/np.linalg.norm(axis)
    return (vectors*np.cos(angle) + np.cross(axis, vectors)*np.sin(angle)
            + np.outer(vectors@axis, axis)*(1-np.cos(angle)))


def static_point_volume(p, pg, vg, lag_deg=15.0):
    Bc, Jc = point_sample(pg, p.r_mag, 0.0)
    bc = np.linalg.norm(Bc)
    Mpc = min(3*bc/MU0, p.Ms)*Bc/bc
    Mpc = rotate_about_axis(Mpc[None, :], [1, 0, 0], np.deg2rad(lag_deg))[0]
    mp = Mpc*p.volume
    Fp = Jc@mp
    Tp = np.cross(mp, Bc)

    B, J = sample_volume(vg, p.r_mag, 0.0)
    bn = np.linalg.norm(B, axis=1)
    Meq = np.minimum(3*bn/MU0, p.Ms)[:, None]*B/bn[:, None]
    M = rotate_about_axis(Meq, [1, 0, 0], np.deg2rad(lag_deg))
    dm = M*vg.weights[:, None]
    dF = np.einsum("nij,nj->ni", J, dm)
    local_torque = np.cross(dm, B)
    force_torque = np.cross(vg.points, dF)
    return dict(Bc=Bc, Jc=Jc, B=B, J=J, Fp=Fp, Tp=Tp,
                Fv=dF.sum(axis=0), Tv=(local_torque+force_torque).sum(axis=0),
                Tv_local=local_torque.sum(axis=0), Tv_force=force_torque.sum(axis=0))


def dynamic_summary(p, grid, dt=DT, t_end=T_END):
    t0 = time.perf_counter()
    stride = max(1, int(round(1.0e-3/dt)))
    r = simulate(p, t_end=t_end, dt=dt, stride=stride, grid=grid)
    elapsed = time.perf_counter()-t0
    sl = slice(len(r.t)//2, None)
    ratio = r.orbital_rate()/p.omega
    return dict(ratio=float(ratio), Fz=float(np.mean(r.Fmag[sl, 2])),
                Fmag=float(np.mean(np.linalg.norm(r.Fmag[sl], axis=1))),
                torque=float(np.mean(np.linalg.norm(r.Tmag[sl], axis=1))),
                Nload=float(np.mean(r.Nload[sl])),
                vtan=float(np.mean(r.v_tan[sl])),
                regime="locked/prograde" if ratio >= 0 else "retrograde",
                elapsed=elapsed, result=r)


def field_nonuniformity(rows):
    data = []
    print("\nFIELD NONUNIFORMITY THROUGH DEFAULT 6 mm BALL (medium quadrature)")
    print("gap |Bc|[mT] mean[mT] min[mT] max[mT] std[mT] Bvec_rms_rel grad_rms_rel")
    for gap in GAPS:
        p = Params(gap=gap, ball_model="volume", volume_quadrature="medium")
        vg = VolumeFieldGrid(p, verbose=True)
        pg = FieldGrid(Params(gap=gap), verbose=False)
        s = static_point_volume(p, pg, vg)
        bm = np.linalg.norm(s["B"], axis=1)
        bc = np.linalg.norm(s["Bc"])
        mean = np.sum(vg.weights*bm)/p.volume
        std = weighted_rms(bm-mean, vg.weights)
        bdiff = np.linalg.norm(s["B"]-s["Bc"], axis=1)
        jdiff = np.linalg.norm(s["J"]-s["Jc"], axis=(1, 2))
        item = dict(record="field_variation", gap_mm=gap*1e3,
                    B_center=bc, B_mean=mean, B_min=float(bm.min()),
                    B_max=float(bm.max()), B_std=std,
                    B_vector_rms_rel=weighted_rms(bdiff, vg.weights)/bc,
                    grad_rms_rel=weighted_rms(jdiff, vg.weights)/np.linalg.norm(s["Jc"]))
        rows.append(item); data.append(item)
        print(f"{gap*1e3:3.0f} {bc*1e3:9.3f} {mean*1e3:8.3f} {bm.min()*1e3:7.3f} "
              f"{bm.max()*1e3:7.3f} {std*1e3:7.3f} "
              f"{item['B_vector_rms_rel']:12.4f} {item['grad_rms_rel']:12.4f}")
    return data


def point_baselines(rows):
    print("\nREQUESTED POINT-MODEL BASELINES")
    print("case gap omega ratio Fz[N] |F|[N] |tau|[Nm] N[N] vtan[m/s] regime")
    for case, gap, omega in (("A", .003, 2.0), ("B", .005, 7.8),
                             ("C", .008, 16.0), ("D", .016, 10.0)):
        p = Params(gap=gap, omega=omega)
        d = dynamic_summary(p, FieldGrid(p, verbose=False))
        item = dict(record="point_baseline", case=case, gap_mm=gap*1e3,
                    omega=omega, ratio=d["ratio"], Fz=d["Fz"], Fmag=d["Fmag"],
                    torque=d["torque"], Nload=d["Nload"], vtan=d["vtan"],
                    regime=d["regime"])
        rows.append(item)
        print(f"{case} {gap*1e3:3.0f} {omega:5.1f} {d['ratio']:+8.4f} {d['Fz']:+9.3e} "
              f"{d['Fmag']:9.3e} {d['torque']:10.3e} {d['Nload']:9.3e} "
              f"{d['vtan']:+10.4e} {d['regime']}")


def small_radius_limit(rows):
    data = []
    center_z = Params().z_ball
    print("\nSMALL-RADIUS LIMIT (fixed center height 14 mm, medium quadrature)")
    print("radius[mm] Fv/Fp tauv/taup |Fv-Fp|/|Fp| |Tv-Tp|/|Tp|")
    for radius in (0.006, 0.003, 0.0015, 0.00075):
        gap = center_z-radius
        p = Params(ball_R=radius, gap=gap, ball_model="volume",
                   volume_quadrature="medium")
        vg = VolumeFieldGrid(p, verbose=False)
        pg = FieldGrid(replace_point(p), verbose=False)
        s = static_point_volume(p, pg, vg)
        fnp, fnv = np.linalg.norm(s["Fp"]), np.linalg.norm(s["Fv"])
        tnp, tnv = np.linalg.norm(s["Tp"]), np.linalg.norm(s["Tv"])
        item = dict(record="small_radius", radius_mm=radius*1e3,
                    F_ratio=fnv/fnp, tau_ratio=tnv/tnp,
                    F_vector_rel=np.linalg.norm(s["Fv"]-s["Fp"])/fnp,
                    tau_vector_rel=np.linalg.norm(s["Tv"]-s["Tp"])/tnp)
        rows.append(item); data.append(item)
        print(f"{radius*1e3:9.3f} {item['F_ratio']:8.5f} {item['tau_ratio']:10.5f} "
              f"{item['F_vector_rel']:13.5e} {item['tau_vector_rel']:13.5e}")
    return data


def replace_point(p):
    from dataclasses import replace
    return replace(p, ball_model="point")


def quadrature_convergence(rows):
    data = []
    print("\nQUADRATURE CONVERGENCE")
    print("gap level n Fnorm[N] taunorm[Nm] F_rel_fine tau_rel_fine ratio")
    for gap, omega in ((0.003, 2.0), (0.008, 16.0)):
        level_data = []
        for level in LEVELS:
            p = Params(gap=gap, omega=omega, ball_model="volume", volume_quadrature=level)
            vg = VolumeFieldGrid(p, verbose=True)
            pg = FieldGrid(Params(gap=gap), verbose=False)
            s = static_point_volume(p, pg, vg)
            # A common reduced-cost trajectory is sufficient to test element-count
            # convergence; production-resolution cases are compared separately.
            dyn = dynamic_summary(p, vg, dt=1.0e-4, t_end=.8)
            level_data.append((level, vg.n_elem, s, dyn))
        sf = level_data[-1][2]
        for level, n, s, dyn in level_data:
            fr = np.linalg.norm(s["Fv"]-sf["Fv"])/np.linalg.norm(sf["Fv"])
            tr = np.linalg.norm(s["Tv"]-sf["Tv"])/np.linalg.norm(sf["Tv"])
            item = dict(record="quadrature", gap_mm=gap*1e3, omega=omega,
                        quadrature=level, n_elem=n,
                        F_norm=float(np.linalg.norm(s["Fv"])),
                        tau_norm=float(np.linalg.norm(s["Tv"])),
                        F_rel_fine=fr, tau_rel_fine=tr, ratio=dyn["ratio"])
            rows.append(item); data.append(item)
            print(f"{gap*1e3:3.0f} {level:6s} {n:4d} {item['F_norm']:9.4e} "
                  f"{item['tau_norm']:11.4e} {fr:10.3e} {tr:12.3e} {dyn['ratio']:+9.5f}")
    return data


def point_volume_matrix(rows):
    data = []
    print("\nPOINT VS MEDIUM-VOLUME DYNAMICS")
    print("gap omega ratio_p ratio_v Fz_p Fz_v |F|p |F|v taup tauv N_v/N_p regime_p regime_v")
    for gap in GAPS:
        pp0 = Params(gap=gap)
        pg = FieldGrid(pp0, verbose=False)
        pv0 = Params(gap=gap, ball_model="volume", volume_quadrature="medium")
        vg = VolumeFieldGrid(pv0, verbose=False)
        for omega in OMEGAS:
            pp = Params(gap=gap, omega=omega)
            pv = Params(gap=gap, omega=omega, ball_model="volume", volume_quadrature="medium")
            dp = dynamic_summary(pp, pg, dt=VOLUME_DT, t_end=VOLUME_T_END)
            dv = dynamic_summary(pv, vg, dt=VOLUME_DT, t_end=VOLUME_T_END)
            item = dict(record="comparison", gap_mm=gap*1e3, omega=omega,
                        ratio_point=dp["ratio"], ratio_volume=dv["ratio"],
                        Fz_point=dp["Fz"], Fz_volume=dv["Fz"],
                        Fmag_point=dp["Fmag"], Fmag_volume=dv["Fmag"],
                        tau_point=dp["torque"], tau_volume=dv["torque"],
                        normal_ratio=dv["Nload"]/dp["Nload"],
                        regime_point=dp["regime"], regime_volume=dv["regime"],
                        time_point=dp["elapsed"], time_volume=dv["elapsed"])
            rows.append(item); data.append(item)
            print(f"{gap*1e3:3.0f} {omega:5.1f} {dp['ratio']:+7.3f} {dv['ratio']:+7.3f} "
                  f"{dp['Fz']:+8.3e} {dv['Fz']:+8.3e} {dp['Fmag']:7.3e} {dv['Fmag']:7.3e} "
                  f"{dp['torque']:7.3e} {dv['torque']:7.3e} {item['normal_ratio']:8.3f} "
                  f"{dp['regime']} {dv['regime']}")
    return data


def critical_speed(pbase, grid, low, high, iterations=6):
    def ratio(omega):
        from dataclasses import replace
        p = replace(pbase, omega=omega)
        return dynamic_summary(p, grid, dt=VOLUME_DT, t_end=VOLUME_T_END)["ratio"]
    rlo, rhi = ratio(low), ratio(high)
    if rlo < 0 or rhi >= 0:
        return np.nan
    for _ in range(iterations):
        mid = 0.5*(low+high)
        if ratio(mid) >= 0:
            low = mid
        else:
            high = mid
    return 0.5*(low+high)


def critical_comparison(rows):
    data = []
    print("\nFROM-REST CRITICAL SPEED: POINT VS MEDIUM VOLUME")
    print("gap omega_point omega_volume shift")
    for gap, point_bracket, volume_bracket in (
            (0.005, (6.0, 10.0), (10.0, 16.0)),
            (0.008, (2.0, 6.0), (2.0, 6.0))):
        pp = Params(gap=gap)
        pv = Params(gap=gap, ball_model="volume", volume_quadrature="medium")
        cp = critical_speed(pp, FieldGrid(pp, verbose=False), *point_bracket)
        cv = critical_speed(pv, VolumeFieldGrid(pv, verbose=False), *volume_bracket)
        item = dict(record="critical", gap_mm=gap*1e3,
                    omega_crit_point=cp, omega_crit_volume=cv, shift=cv-cp)
        rows.append(item); data.append(item)
        print(f"{gap*1e3:3.0f} {cp:11.5f} {cv:12.5f} {cv-cp:+9.5f}")
    return data


def performance(rows):
    data = []
    print("\nPERFORMANCE (warm grid/JIT, wall seconds per simulated second)")
    print("model quadrature n cost")
    pp = Params(gap=.008, omega=16)
    pg = FieldGrid(pp, verbose=False)
    cases = [("point", "point", 1, pp, pg)]
    for level in LEVELS:
        pv = Params(gap=.008, omega=16, ball_model="volume", volume_quadrature=level)
        cases.append(("volume", level, len(sphere_quadrature(pv.ball_R, level)[1]),
                      pv, VolumeFieldGrid(pv, verbose=False)))
    for model, level, n, p, g in cases:
        simulate(p, t_end=.002, dt=DT, stride=100, grid=g)  # JIT/warm-up
        t0 = time.perf_counter()
        simulate(p, t_end=.1, dt=DT, stride=100, grid=g)
        cost = (time.perf_counter()-t0)/.1
        item = dict(record="performance", model=model, quadrature=level,
                    n_elem=n, seconds_per_sim_second=cost)
        rows.append(item); data.append(item)
        print(f"{model:6s} {level:9s} {n:4d} {cost:10.4f}")
    return data


def write_csv(rows, path):
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def make_figure(field, quad, comp, critical, output):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    gaps = [x["gap_mm"] for x in field]
    axes[0, 0].plot(gaps, [x["B_vector_rms_rel"] for x in field], "o-")
    axes[0, 0].set(title="A. B nonuniformity", xlabel="gap [mm]", ylabel="weighted RMS / center")
    axes[0, 1].plot(gaps, [x["grad_rms_rel"] for x in field], "o-")
    axes[0, 1].set(title="B. grad B nonuniformity", xlabel="gap [mm]", ylabel="weighted RMS / center")
    for omega in OMEGAS:
        d = [x for x in comp if x["omega"] == omega]
        axes[0, 2].plot([x["gap_mm"] for x in d],
                        [x["Fmag_volume"]/x["Fmag_point"] for x in d], "o-", label=f"omega={omega:g}")
        axes[0, 3].plot([x["gap_mm"] for x in d],
                        [x["tau_volume"]/x["tau_point"] for x in d], "o-", label=f"omega={omega:g}")
    axes[0, 2].set(title="C. Dynamic |F| volume/point", xlabel="gap [mm]", ylabel="ratio")
    axes[0, 3].set(title="D. Dynamic |tau| volume/point", xlabel="gap [mm]", ylabel="ratio")
    axes[0, 2].set_yscale("log"); axes[0, 3].set_yscale("log")
    axes[0, 2].legend(fontsize=7); axes[0, 3].legend(fontsize=7)
    for omega in OMEGAS:
        d = [x for x in comp if x["omega"] == omega]
        axes[1, 0].plot([x["gap_mm"] for x in d], [x["ratio_point"] for x in d], "o--", alpha=.6)
        axes[1, 0].plot([x["gap_mm"] for x in d], [x["ratio_volume"] for x in d], "s-", label=f"omega={omega:g}")
    axes[1, 0].axhline(0, color="k", lw=.7)
    axes[1, 0].set(title="E. Motion ratio (dashed point, solid volume)", xlabel="gap [mm]", ylabel="motion ratio")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].bar([str(x["gap_mm"]) for x in critical], [x["shift"] for x in critical])
    axes[1, 1].set(title="F. Critical-speed shift", xlabel="gap [mm]", ylabel="Delta omega [rad/s]")
    for gap in (3.0, 8.0):
        d = [x for x in quad if x["gap_mm"] == gap]
        axes[1, 2].plot([x["n_elem"] for x in d], [x["ratio"] for x in d], "o-", label=f"gap={gap:g}")
    axes[1, 2].set_xscale("log")
    axes[1, 2].set(title="G. Quadrature motion convergence", xlabel="elements", ylabel="motion ratio")
    axes[1, 2].legend(fontsize=8)
    # Cross-section proxy: min/mean/max B at each gap.
    axes[1, 3].fill_between(gaps, [x["B_min"]*1e3 for x in field],
                            [x["B_max"]*1e3 for x in field], alpha=.25, label="min-max")
    axes[1, 3].plot(gaps, [x["B_mean"]*1e3 for x in field], "o-", label="volume mean")
    axes[1, 3].plot(gaps, [x["B_center"]*1e3 for x in field], "s--", label="center")
    axes[1, 3].set(title="H. Field range through sphere", xlabel="gap [mm]", ylabel="|B| [mT]")
    axes[1, 3].legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs("out", exist_ok=True)
    rows = []
    print("FINITE-VOLUME BALL DIAGNOSTIC")
    p = Params()
    print(f"a={p.ball_R*1e3:g} mm V={p.volume:.9e} m^3 Ms={p.Ms:.6e} A/m")
    for n in (6, 12, 18):
        print(f"k{n}*a = {n/p.r_mag*p.ball_R:.6f}")
    point_baselines(rows)
    field = field_nonuniformity(rows)
    small_radius_limit(rows)
    quad = quadrature_convergence(rows)
    comp = point_volume_matrix(rows)
    critical = critical_comparison(rows)
    performance(rows)
    csv_path = "out/volume_integration_diagnostic.csv"
    fig_path = "out/volume_integration_diagnostic.png"
    write_csv(rows, csv_path); make_figure(field, quad, comp, critical, fig_path)
    print(f"\nsaved {csv_path}\nsaved {fig_path}")


if __name__ == "__main__":
    main()

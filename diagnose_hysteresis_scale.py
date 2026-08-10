"""Step 6A: quantify the material-loss scale represented by ``tau_lag``.

Diagnostic only.  No hysteresis model is connected to production dynamics.
Apparent Hc and Mr below are dynamic loop intercepts, not true quasistatic
coercivity or remanence.  Illustrative rectangular loops are scale comparisons,
not material fits.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_skin_effect import (load_trajectory, quat_matrix,
                                  reconstruct_orientation)
from hysteresis_scale import linear_metrics, rotating_metrics
from magnetic_carousel import MU0, FieldGrid, Params, _field_lab
from magnetic_diffusion_sphere import response_l
from magnetic_diffusion_wrench import rotating_uniform_analytic


OUT = Path("out")
CSV_PATH = OUT/"hysteresis_scale_diagnostic.csv"
PNG_PATH = OUT/"hysteresis_scale_diagnostic.png"
B_VALUES = (.005, .010, .020, .030, .050)
OMEGAS = (.01, 20.42, 26.70, 34.56, 53.41, 72.26, 103.67, 117.81)
LOW_OMEGAS = np.logspace(-3, 1, 17)
CASES = (
    ("A_locked_3mm_2", .003, 2.0),
    ("B_transition_5mm_7p8", .005, 7.8),
    ("C_retrograde_8mm_16", .008, 16.0),
    ("D_weak_16mm_10", .016, 10.0),
)


def blank_row(benchmark, **kw):
    d = dict(benchmark=benchmark, case="", B0_T=np.nan, mean_B_T=np.nan,
             Omega_rad_s=np.nan, omega_disc_rad_s=np.nan, tau_lag_s=np.nan,
             loop_area_J_m3_cycle=np.nan, apparent_Hc_A_m=np.nan,
             apparent_Mr_A_m=np.nan, apparent_Mr_over_Ms=np.nan,
             peak_M_A_m=np.nan, differential_susceptibility=np.nan,
             mean_torque_Nm=np.nan, mean_radial_torque_Nm=np.nan,
             mechanical_power_W=np.nan, loss_power_W=np.nan,
             energy_per_cycle_J=np.nan, energy_density_J_m3_cycle=np.nan,
             eddy_loss_reference_J_m3_cycle=np.nan, ratio_to_eddy=np.nan,
             Hc_illustrative_A_m=np.nan, Mr_over_Ms_illustrative=np.nan,
             note="")
    d.update(kw)
    return d


def dominant_omega(case):
    with (OUT/"skin_effect_diagnostic.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if (r["row_type"] == "spectral_peak" and r["case"] == case and
                    r["material_point"] == "center" and r["peak_rank"] == "1" and
                    float(r["sigma_S_m"]) == 6e6 and float(r["mu_r_ac"]) == 100):
                return float(r["Omega_rad_s"])
    raise RuntimeError(f"no Step-5A dominant frequency for {case}")


def carousel_lag_metrics(name, gap, omega_disc):
    p, t, state = load_trajectory(name, gap, omega_disc)
    grid = FieldGrid(p, verbose=False)
    B_all = np.empty((len(t), 3))
    for i, (ti, yi) in enumerate(zip(t, state)):
        B_all[i], _ = _field_lab(grid.F, grid.x0, grid.y0, grid.dx, grid.dy,
                                 yi[0], yi[1], p.omega*ti)
    orientation = reconstruct_orientation(t, state[:, 4:7])
    keep = t >= 4.0
    tk, y, B = t[keep], state[keep], B_all[keep]
    M = y[:, 7:10]/p.volume
    H = B/MU0
    meq = p.chi_eff*H
    mn = np.linalg.norm(meq, axis=1)
    sat = mn > p.Ms
    meq[sat] *= (p.Ms/mn[sat])[:, None]
    loss_density = MU0*np.sum((meq-M)**2, axis=1)/(p.chi_eff*p.tau_lag)
    torque = np.cross(y[:, 7:10], B)
    omega_body = y[:, 4:7]
    radius = np.hypot(y[:, 0], y[:, 1])
    er = np.column_stack((y[:, 0]/radius, y[:, 1]/radius, np.zeros(len(y))))
    trad = np.einsum("ij,ij->i", torque, er)
    mech_power = np.einsum("ij,ij->i", torque, omega_body)
    Om = dominant_omega(name)
    ploss = float(np.mean(loss_density)*p.volume)
    Wdensity = float(np.mean(loss_density)*2*np.pi/Om)
    # Exact l=1 homogeneous-sphere loss summed over the measured material-frame
    # spectrum.  Unequal-frequency quadratic cross terms vanish in the record
    # average; all Cartesian phasors in one bin are combined before loss.
    Bbody = np.array([quat_matrix(q).T@b for q, b in zip(orientation[keep], B)])
    Bbody -= Bbody.mean(axis=0)
    n = len(tk); dt = float(np.median(np.diff(tk)))
    X = np.fft.rfft(Bbody, axis=0)/n
    freq = np.fft.rfftfreq(n, dt)
    Peddy = 0.0
    for k in range(1, len(freq)):
        if freq[k] > 100.0:
            break
        omega_k = 2*np.pi*freq[k]
        Hhat_sq = np.sum(np.abs(2*X[k]/MU0)**2)
        alpha = response_l(1, omega_k, p.ball_R, p.sigma,
                           p.mu_r_ac).dipole_moment_per_H0
        Peddy += -.5*omega_k*MU0*alpha.imag*Hhat_sq
    Weddy = Peddy*2*np.pi/Om/p.volume
    return dict(p=p, mean_B=float(np.mean(np.linalg.norm(B, axis=1))), Omega=Om,
                mean_torque=float(np.mean(np.linalg.norm(torque, axis=1))),
                mean_radial_torque=float(np.mean(trad)),
                mechanical_power=float(np.mean(mech_power)), loss_power=ploss,
                energy_density=Wdensity, energy_cycle=Wdensity*p.volume,
                eddy_density=Weddy, eddy_power=Peddy,
                ratio=Wdensity/Weddy)


def run():
    OUT.mkdir(exist_ok=True)
    p = Params(); V = p.volume
    rows = []
    loops = {}
    rotating = []
    for B0 in B_VALUES:
        for Om in OMEGAS:
            lm = linear_metrics(B0, Om, p.tau_lag, p.chi_eff, p.Ms)
            eddy = response_l(1, Om, p.ball_R, p.sigma, p.mu_r_ac,
                              H0=B0/MU0)
            Weddy = (eddy.joule_loss_per_H0_sq*2*np.pi/Om/V
                     if Om > 0 else 0.0)
            rows.append(blank_row(
                "linear_tau_lag", B0_T=B0, Omega_rad_s=Om,
                tau_lag_s=p.tau_lag, loop_area_J_m3_cycle=lm["loop_area_exact"],
                apparent_Hc_A_m=lm["apparent_Hc"],
                apparent_Mr_A_m=lm["apparent_Mr"],
                apparent_Mr_over_Ms=lm["remanence_ratio"],
                peak_M_A_m=lm["peak_M"],
                differential_susceptibility=lm["differential_susceptibility"],
                loss_power_W=lm["average_loss_power_density"]*V,
                energy_per_cycle_J=lm["loop_area_exact"]*V,
                energy_density_J_m3_cycle=lm["loop_area_exact"],
                eddy_loss_reference_J_m3_cycle=Weddy,
                ratio_to_eddy=lm["loop_area_exact"]/Weddy,
                note="Hc and Mr are dynamic intercepts, not quasistatic material properties"))
            rm = rotating_metrics(B0, Om, p.tau_lag, V, p.chi_eff)
            tau_e, _ = rotating_uniform_analytic(B0, Om, p.ball_R,
                                                  p.sigma, p.mu_r_ac)
            Weddy_rot = 2*np.pi*tau_e/V
            rotating.append((B0, Om, rm, Weddy_rot))
            rows.append(blank_row(
                "circular_tau_lag", B0_T=B0, Omega_rad_s=Om,
                tau_lag_s=p.tau_lag, mean_torque_Nm=rm["torque"],
                mechanical_power_W=rm["power"], loss_power_W=rm["power"],
                energy_per_cycle_J=rm["energy_per_cycle"],
                energy_density_J_m3_cycle=rm["energy_density_per_cycle"],
                eddy_loss_reference_J_m3_cycle=Weddy_rot,
                ratio_to_eddy=rm["energy_density_per_cycle"]/Weddy_rot,
                note="circular field: P_loss=Omega*tau and E_cycle=2pi*tau"))
            if B0 == .020 and Om in (.01, 20.42, 53.41, 117.81):
                loops[(B0, Om)] = (lm["H"], lm["M"])

    low_area = []
    for Om in LOW_OMEGAS:
        lm = linear_metrics(.020, Om, p.tau_lag, p.chi_eff, p.Ms)
        low_area.append((Om, lm["loop_area_exact"], lm["apparent_Hc"],
                         lm["apparent_Mr"]))
        rows.append(blank_row("low_frequency_asymptotic", B0_T=.020,
                              Omega_rad_s=Om, tau_lag_s=p.tau_lag,
                              loop_area_J_m3_cycle=lm["loop_area_exact"],
                              apparent_Hc_A_m=lm["apparent_Hc"],
                              apparent_Mr_A_m=lm["apparent_Mr"],
                              apparent_Mr_over_Ms=lm["remanence_ratio"],
                              note="exact Debye relaxation asymptotic"))

    carousel = []
    for name, gap, od in CASES:
        q = carousel_lag_metrics(name, gap, od)
        carousel.append((name, gap, od, q))
        rows.append(blank_row(
            "carousel_tau_lag", case=name, mean_B_T=q["mean_B"],
            Omega_rad_s=q["Omega"], omega_disc_rad_s=od,
            tau_lag_s=q["p"].tau_lag, mean_torque_Nm=q["mean_torque"],
            mean_radial_torque_Nm=q["mean_radial_torque"],
            mechanical_power_W=q["mechanical_power"], loss_power_W=q["loss_power"],
            energy_per_cycle_J=q["energy_cycle"],
            energy_density_J_m3_cycle=q["energy_density"],
            eddy_loss_reference_J_m3_cycle=q["eddy_density"],
            ratio_to_eddy=q["ratio"],
            note="trajectory average after 4 s; eddy reference sums exact l=1 loss over body-frame spectrum <=100 Hz"))

    # Purely illustrative rectangular-loop scale: W=4 mu0 Hc Mr.
    illustrative = []
    for Hc in (100, 300, 1000, 3000, 10000):
        for rr in (.05, .2, .5, .8):
            W = 4*MU0*Hc*rr*p.Ms
            illustrative.append((Hc, rr, W))
            rows.append(blank_row("illustrative_rectangular_loop",
                                  energy_density_J_m3_cycle=W,
                                  Hc_illustrative_A_m=Hc,
                                  Mr_over_Ms_illustrative=rr,
                                  note="scale only; not a fitted ball material"))

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, axs = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    ax = axs.ravel()
    for (_, Om), (H, M) in loops.items():
        ax[0].plot(H/1e3, M/1e3, label=f"{Om:g} rad/s")
    ax[0].set(xlabel="Hext [kA/m]", ylabel="M [kA/m]", title="A  tau_lag M-H loops"); ax[0].legend(fontsize=7)
    q20 = [r for r in rows if r["benchmark"] == "linear_tau_lag" and r["B0_T"] == .020]
    ax[1].loglog([r["Omega_rad_s"] for r in q20], [r["loop_area_J_m3_cycle"] for r in q20], "o-")
    ax[1].set(xlabel="Omega [rad/s]", ylabel="loop area [J/m3/cycle]", title="B  Dynamic loop area")
    ax[2].loglog([x[0] for x in low_area], [x[2] for x in low_area], "o-", label="apparent Hc")
    ax[2].loglog([x[0] for x in low_area], [x[3] for x in low_area], "s-", label="apparent Mr")
    ax[2].set(xlabel="Omega [rad/s]", ylabel="dynamic intercept", title="C  Hc and Mr vanish"); ax[2].legend(fontsize=7)
    qrot = [x for x in rotating if x[0] == .020]
    ax[3].plot([x[1] for x in qrot], [x[2]["torque"] for x in qrot], "o-", label="torque")
    ax[3].set(xlabel="Omega [rad/s]", ylabel="torque [N m]", title="D  Circular-field lag torque")
    ax[4].semilogy([x[1] for x in qrot], [x[2]["energy_density_per_cycle"]/x[3] for x in qrot], "o-")
    ax[4].set(xlabel="Omega [rad/s]", ylabel="Wlag / Weddy", title="E  Lag versus exact eddy loss")
    names = [x[0].split("_")[0] for x in carousel]
    ax[5].bar(names, [x[3]["energy_density"] for x in carousel], label="required lag loss")
    ax[5].bar(names, [x[3]["eddy_density"] for x in carousel], label="eddy reference")
    ax[5].set(ylabel="J/m3/cycle", title="F  Carousel loss-density scale"); ax[5].legend(fontsize=7)
    for rr in (.05, .2, .5, .8):
        q = [x for x in illustrative if x[1] == rr]
        ax[6].loglog([x[0] for x in q], [x[2] for x in q], "o-", label=f"Mr/Ms={rr:g}")
    req = [x[3]["energy_density"] for x in carousel]
    ax[6].axhspan(min(req), max(req), color="k", alpha=.12, label="Carousel-required")
    ax[6].set(xlabel="illustrative Hc [A/m]", ylabel="4 mu0 Hc Mr [J/m3]",
              title="G  Illustrative loop scale"); ax[6].legend(fontsize=6)
    ax[7].loglog([x[0] for x in low_area], [x[1] for x in low_area], "o-", label="exact")
    ref = low_area[0][1]*np.array([x[0]/low_area[0][0] for x in low_area])
    ax[7].loglog([x[0] for x in low_area], ref, "k--", label="proportional to Omega")
    ax[7].set(xlabel="Omega [rad/s]", ylabel="area [J/m3/cycle]",
              title="H  Quasistatic asymptote"); ax[7].legend(fontsize=7)
    fig.suptitle("Magnetic Carousel Step 6A: scale represented by tau_lag")
    fig.savefig(PNG_PATH, dpi=180); plt.close(fig)

    slope = np.polyfit(np.log([x[0] for x in low_area[:8]]),
                       np.log([x[1] for x in low_area[:8]]), 1)[0]
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    print(f"wrote {PNG_PATH}")
    print(f"low-frequency loop-area exponent p={slope:.8f}")
    for name, gap, od, q in carousel:
        print(name, f"B={q['mean_B']*1e3:.3f}mT Omega={q['Omega']:.3f} "
              f"tau={q['mean_torque']:.6e}Nm W={q['energy_density']:.3f}J/m3 "
              f"ratio_eddy={q['ratio']:.2f}")


if __name__ == "__main__":
    run()

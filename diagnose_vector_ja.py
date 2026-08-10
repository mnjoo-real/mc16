"""Step 6B-1 standalone scalar/vector Jiles--Atherton diagnostics.

All parameter sets are synthetic sensitivity cases. H means a prescribed
constitutive-input field; it is not asserted to equal the actual ball's
internal field, and no demagnetization or production-dynamics coupling occurs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hysteresis_scale import linear_metrics, linear_steady_state
from magnetic_carousel import MU0, Params
from magnetic_hysteresis import (SYNTHETIC_PARAMETER_SETS, alternating_path,
    circular_path, integrate_scalar_path, integrate_vector_path,
    scalar_loop_metrics, vector_cycle_work)


OUT = Path("out")
CSV_PATH = OUT/"vector_ja_diagnostic.csv"
PNG_PATH = OUT/"vector_ja_diagnostic.png"
B_VALUES = (.005, .010, .020, .030, .050)
OMEGA_REF = 53.41
N = 500
CYCLES = 5


def row(model, parameter_set, field_path, H0, Omega, p=None, **kw):
    d = dict(model=model, parameter_set=parameter_set, field_path=field_path,
             H0_A_m=H0, B_equivalent_T=MU0*H0, Omega_rad_s=Omega,
             Ms_A_m=np.nan, a_A_m=np.nan, k_A_m=np.nan, c=np.nan, alpha=np.nan,
             Hc_A_m=np.nan, Mr_A_m=np.nan, Mr_over_Ms=np.nan,
             loop_area_J_m3_cycle=np.nan, minor_loop_area_J_m3_cycle=np.nan,
             rotational_loss_J_m3_cycle=np.nan,
             mean_torque_density_proxy_J_m3=np.nan,
             max_angle_lag_deg=np.nan, path_increment_A_m=np.nan,
             convergence_level="", normalized_discrepancy=np.nan, note="")
    if p is not None:
        d.update(Ms_A_m=p.Ms, a_A_m=p.a, k_A_m=p.k, c=p.c, alpha=p.alpha)
    d.update(kw); return d


def last_cycle(x, n=N):
    return x[-n-1:]


def angle_lag(H, M):
    hn = np.linalg.norm(H, axis=1); mn = np.linalg.norm(M, axis=1)
    good = (hn > 0) & (mn > 0)
    cs = np.sum(H[good]*M[good], axis=1)/(hn[good]*mn[good])
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def run():
    OUT.mkdir(exist_ok=True)
    rows, results = [], {}
    for name, p in SYNTHETIC_PARAMETER_SETS.items():
        for B0 in B_VALUES:
            H0 = B0/MU0
            H = alternating_path(H0, N, CYCLES)
            s = integrate_scalar_path(H, p)
            met = scalar_loop_metrics(last_cycle(H), last_cycle(s["M"]))

            # A centered half-amplitude minor loop entered from the final +H0
            # major-loop state; several cycles remove entry transients.
            ramp = np.linspace(H0, .5*H0, N//8+1)
            th = np.linspace(0, 2*np.pi*CYCLES, N*CYCLES+1)
            hm = np.r_[ramp, .5*H0*np.cos(th)[1:]]
            sm = integrate_scalar_path(hm, p, s["final_state"])
            minor = scalar_loop_metrics(last_cycle(hm), last_cycle(sm["M"]))

            C = circular_path(H0, N, CYCLES)
            vc = integrate_vector_path(C, p)
            Hc, Mc = last_cycle(C), last_cycle(vc["M"])
            Wrot = vector_cycle_work(Hc, Mc)
            lag = angle_lag(Hc, Mc)
            torque_density = MU0*np.linalg.norm(np.cross(Mc, Hc), axis=1)
            results[(name, B0)] = dict(H=H, scalar=s, major=met, C=C,
                                       vector=vc, rotational_loss=Wrot,
                                       lag=lag, torque_density=torque_density,
                                       minor=minor)
            rows.append(row("scalar_ja", name, "alternating_major", H0,
                            OMEGA_REF, p, Hc_A_m=met["Hc"], Mr_A_m=met["Mr"],
                            Mr_over_Ms=met["Mr"]/p.Ms,
                            loop_area_J_m3_cycle=met["loop_area"],
                            minor_loop_area_J_m3_cycle=minor["loop_area"],
                            path_increment_A_m=2*np.pi*H0/N,
                            convergence_level=f"N={N}",
                            note="synthetic rate-independent scalar reference"))
            rows.append(row("vector_ja", name, "circular", H0, OMEGA_REF, p,
                            Hc_A_m=met["Hc"], Mr_A_m=met["Mr"],
                            Mr_over_Ms=met["Mr"]/p.Ms,
                            loop_area_J_m3_cycle=met["loop_area"],
                            minor_loop_area_J_m3_cycle=minor["loop_area"],
                            rotational_loss_J_m3_cycle=Wrot,
                            mean_torque_density_proxy_J_m3=float(np.mean(torque_density)),
                            max_angle_lag_deg=float(np.max(lag)),
                            path_increment_A_m=2*np.pi*H0/N,
                            convergence_level=f"N={N}",
                            note="prescribed H path; B equivalent is mu0 H only"))

    # Mandatory scalar reduction evidence at the representative medium case.
    q = results[("synthetic_medium", .020)]; p = SYNTHETIC_PARAMETER_SETS["synthetic_medium"]
    H = q["H"]; sv = q["scalar"]
    Hv = np.column_stack((H, np.zeros((len(H), 2))))
    vv = integrate_vector_path(Hv, p)
    reduction = float(np.max(np.abs(vv["M"][:, 0]-sv["M"]))/p.Ms)
    rows.append(row("validation", p.label, "scalar_reduction", .020/MU0,
                    OMEGA_REF, p, normalized_discrepancy=reduction,
                    convergence_level=f"N={N}",
                    note="max |Mx_vector-M_scalar|/Ms; transverse M also zero"))

    # Rotational covariance and traversal-rate invariance of the same circle.
    C = results[(p.label, .020)]["C"]
    vc = results[(p.label, .020)]["vector"]
    axis = np.array((1., 2., 3.)); axis /= np.linalg.norm(axis); ang = .73
    K = np.array(((0., -axis[2], axis[1]), (axis[2], 0., -axis[0]),
                  (-axis[1], axis[0], 0.)))
    R = np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*(K@K)
    vr = integrate_vector_path(C@R.T, p)
    covariance = float(np.max(np.linalg.norm(vr["M"]-vc["M"]@R.T, axis=1))/p.Ms)
    rows.append(row("validation", p.label, "rotational_covariance", .020/MU0,
                    OMEGA_REF, p, normalized_discrepancy=covariance,
                    rotational_loss_J_m3_cycle=results[(p.label,.020)]["rotational_loss"],
                    note="fixed arbitrary 3-D rotation; loss invariant"))
    for Om in (.01, 1., 53.41, 117.81):
        rows.append(row("vector_ja", p.label, "circular_rate_check", .020/MU0,
                        Om, p,
                        rotational_loss_J_m3_cycle=results[(p.label,.020)]["rotational_loss"],
                        max_angle_lag_deg=float(np.max(results[(p.label,.020)]["lag"])),
                        note="identical geometric H path; traversal rate changes timestamps only"))

    # Path-increment convergence.
    convergence = []
    for nn in (800, 1600, 3200):
        h = alternating_path(.020/MU0, nn, CYCLES)
        ss = integrate_scalar_path(h, p)
        mm = scalar_loop_metrics(h[-nn-1:], ss["M"][-nn-1:])
        convergence.append((nn, mm))
        rows.append(row("validation", p.label, "path_convergence", .020/MU0,
                        OMEGA_REF, p, Hc_A_m=mm["Hc"], Mr_A_m=mm["Mr"],
                        Mr_over_Ms=mm["Mr"]/p.Ms,
                        loop_area_J_m3_cycle=mm["loop_area"],
                        path_increment_A_m=2*np.pi*(.020/MU0)/nn,
                        convergence_level=f"N={nn}"))

    # Rate-independent JA versus reversible and old tau_lag at identical H path.
    comparison = []
    for Om in (.01, .1, 1., 20.42, 53.41, 117.81):
        lag = linear_metrics(.020, Om, Params().tau_lag,
                             Params().chi_eff, Params().Ms)
        Wja = results[("synthetic_medium", .020)]["major"]["loop_area"]
        comparison.append((Om, Wja, lag["loop_area_exact"]))
        rows.append(row("reversible", "chi_eff=3", "alternating", .020/MU0,
                        Om, loop_area_J_m3_cycle=0, Mr_A_m=0, Hc_A_m=0,
                        note="instantaneous reversible equilibrium"))
        rows.append(row("tau_lag", "tau=4ms", "alternating", .020/MU0, Om,
                        loop_area_J_m3_cycle=lag["loop_area_exact"],
                        Hc_A_m=lag["apparent_Hc"], Mr_A_m=lag["apparent_Mr"],
                        Mr_over_Ms=lag["remanence_ratio"],
                        note="dynamic intercepts vanish with Omega"))
        rows.append(row("vector_ja", p.label, "alternating", .020/MU0, Om, p,
                        loop_area_J_m3_cycle=Wja,
                        Hc_A_m=results[(p.label, .020)]["major"]["Hc"],
                        Mr_A_m=results[(p.label, .020)]["major"]["Mr"],
                        Mr_over_Ms=results[(p.label, .020)]["major"]["Mr"]/p.Ms,
                        note="same geometric path: rate independent"))

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    fig, axs = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    ax = axs.ravel()
    for name in SYNTHETIC_PARAMETER_SETS:
        q = results[(name, .050)]; ax[0].plot(last_cycle(q["H"])/1e3,
            last_cycle(q["scalar"]["M"])/1e3, label=name.replace("synthetic_", ""))
    ax[0].set(xlabel="H [kA/m]", ylabel="M [kA/m]", title="A  Scalar JA major loops"); ax[0].legend(fontsize=7)
    ax[1].plot(last_cycle(H)/1e3, last_cycle(sv["M"])/1e3, label="scalar")
    ax[1].plot(last_cycle(H)/1e3, last_cycle(vv["M"][:, 0])/1e3, "--", label="vector Mx")
    ax[1].set(xlabel="Hx [kA/m]", ylabel="M [kA/m]", title="B  Vector -> scalar reduction"); ax[1].legend(fontsize=7)
    q = results[("synthetic_medium", .020)]; Hc, Mc = last_cycle(q["C"]), last_cycle(q["vector"]["M"])
    ax[2].plot(Hc[:, 0]/np.max(np.linalg.norm(Hc, axis=1)), Hc[:, 1]/np.max(np.linalg.norm(Hc, axis=1)), label="H/H0")
    ax[2].plot(Mc[:, 0]/p.Ms, Mc[:, 1]/p.Ms, label="M/Ms")
    ax[2].axis("equal"); ax[2].set(xlabel="x component", ylabel="y component", title="C  Circular vector trajectories"); ax[2].legend(fontsize=7)
    signed_lag = np.degrees(np.arctan2(np.cross(Hc, Mc)[:, 2],
                                       np.einsum("ij,ij->i", Hc, Mc)))
    ax[3].plot(np.linspace(0, 360, len(signed_lag)), signed_lag)
    lag_center = float(np.mean(signed_lag))
    ax[3].set_ylim(lag_center-.1, lag_center+.1)
    ax[3].ticklabel_format(axis="y", style="plain", useOffset=False)
    ax[3].set(xlabel="path angle [deg]", ylabel="signed angle(H,M) [deg]", title="D  Rotational angle lag")
    for name in SYNTHETIC_PARAMETER_SETS:
        ax[4].plot(np.array(B_VALUES)*1e3, [results[(name,b)]["rotational_loss"] for b in B_VALUES], "o-", label=name.replace("synthetic_", ""))
    ax[4].set(xlabel="mu0 H0 [mT]", ylabel="rotational loss [J/m3/cycle]", title="E  Rotational-loss sensitivity"); ax[4].legend(fontsize=7)
    ax[5].loglog([x[0] for x in comparison], [x[1] for x in comparison], "o-", label="vector JA")
    ax[5].loglog([x[0] for x in comparison], [x[2] for x in comparison], "s-", label="tau_lag")
    ax[5].set(xlabel="Omega [rad/s]", ylabel="alternating loss [J/m3/cycle]", title="F  Rate-independent vs lag"); ax[5].legend(fontsize=7)
    names = list(SYNTHETIC_PARAMETER_SETS)
    ax[6].bar(np.arange(3)-.18, [results[(n,.05)]["major"]["Hc"] for n in names], .36, label="Hc [A/m]")
    ax[6].bar(np.arange(3)+.18, [results[(n,.05)]["major"]["Mr"]/SYNTHETIC_PARAMETER_SETS[n].Ms*1e4 for n in names], .36, label="Mr/Ms x1e4")
    ax[6].set_xticks(range(3), [n.replace("synthetic_", "") for n in names]); ax[6].set(title="G  Synthetic set span"); ax[6].legend(fontsize=7)
    for name in names:
        ax[7].semilogy(np.array(B_VALUES)*1e3, [results[(name,b)]["major"]["loop_area"] for b in B_VALUES], "o-", label=f"{name[10:]} alternating")
        ax[7].semilogy(np.array(B_VALUES)*1e3, [results[(name,b)]["rotational_loss"] for b in B_VALUES], "--", alpha=.7)
    ax[7].axhspan(29, 3216, color="k", alpha=.12, label="Step-6A required")
    ax[7].set(xlabel="mu0 H0 [mT]", ylabel="loss [J/m3/cycle]", title="H  Required-loss coverage"); ax[7].legend(fontsize=6)
    fig.suptitle("Magnetic Carousel Step 6B-1: synthetic isotropic vector JA")
    fig.savefig(PNG_PATH, dpi=180); plt.close(fig)

    all_losses = [q["major"]["loop_area"] for q in results.values()]+[q["rotational_loss"] for q in results.values()]
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    print(f"wrote {PNG_PATH}")
    print(f"scalar reduction discrepancy={reduction:.3e}")
    print(f"rotational covariance discrepancy={covariance:.3e}")
    print(f"synthetic loss coverage={min(all_losses):.3f}..{max(all_losses):.3f} J/m3/cycle")
    for name in names:
        q = results[(name, .050)]
        print(name, f"Hc={q['major']['Hc']:.2f} Mr/Ms={q['major']['Mr']/SYNTHETIC_PARAMETER_SETS[name].Ms:.4f} "
              f"Wmajor={q['major']['loop_area']:.2f} Wminor={q['minor']['loop_area']:.2f} "
              f"Wrot={q['rotational_loss']:.2f} lagmax={np.max(q['lag']):.2f}deg")


if __name__ == "__main__":
    run()

"""Step 5B-2 offline Maxwell-stress diagnostics (no dynamics coupling).

The snapshot calculation treats a real, fixed Carousel field pattern as the
complex amplitude of one linearly oscillating component.  The optional
multi-frequency estimate weights independent copies of that spatial pattern
by the measured Step-5A spectral powers.  This is a separable-spectrum scale
estimate, not a transient solution: unequal-frequency stress cross terms are
discarded by long-time averaging, while coefficients at one frequency are
combined before evaluating the quadratic Maxwell stress.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_skin_effect import DiagnosticField3D
from magnetic_carousel import (MU0, DemagSphereSolver, Params,
                               PointSetFieldGrid, VolumeFieldGrid,
                               static_magnetic_response)
from magnetic_diffusion_sphere import response_l
from magnetic_diffusion_wrench import (
    AppliedMode, axisymmetric_linear_gradient_modes, maxwell_wrench,
    modes_from_coefficients, project_radial_field,
    rotating_uniform_analytic, tau_lag_rotating_torque,
    uniform_field_modes)


OUT = Path("out")
CSV_PATH = OUT/"diffusion_wrench_diagnostic.csv"
PNG_PATH = OUT/"diffusion_wrench_diagnostic.png"
OMEGAS = np.array((20.42, 26.70, 34.56, 53.41, 72.26, 103.67, 117.81))
GAP_OMEGA = {0.003: 20.42, 0.005: 26.70, 0.008: 34.56, 0.016: 53.41}
PHASES = (0.0, np.pi/24, np.pi/12)
FIELD_AMPLITUDES = (0.010, 0.020, 0.030)


def row(kind, **kw):
    base = dict(benchmark=kind, case="", gap_m=np.nan, phase_rad=np.nan,
                l=np.nan, m=np.nan, omega_rad_s=np.nan, B_amplitude_T=np.nan,
                Fx_N=np.nan, Fy_N=np.nan, Fz_N=np.nan,
                taux_Nm=np.nan, tauy_Nm=np.nan, tauz_Nm=np.nan,
                joule_power_W=np.nan, stress_radius_over_a=np.nan,
                n_mu=np.nan, n_phi=np.nan, static_diffusion_ratio=np.nan,
                tau_lag_Nm=np.nan, diffusion_tau_lag_ratio=np.nan,
                volume_Fx_N=np.nan, volume_Fy_N=np.nan, volume_Fz_N=np.nan,
                relative_error=np.nan, spectral_weight=np.nan,
                note="")
    base.update(kw)
    return base


def norm(v):
    return float(np.linalg.norm(v))


def measured_peaks():
    """Top-three center-field Step-5A peaks for the four representative gaps."""
    wanted = {"A_locked_3mm_2": .003, "B_transition_5mm_7p8": .005,
              "C_retrograde_8mm_16": .008, "D_weak_16mm_10": .016}
    found = {gap: [] for gap in wanted.values()}
    path = OUT/"skin_effect_diagnostic.csv"
    if path.exists():
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                if (r["row_type"] == "spectral_peak" and
                        r["case"] in wanted and r["material_point"] == "center" and
                        float(r["sigma_S_m"]) == 6e6 and
                        float(r["mu_r_ac"]) == 100.0):
                    gap = wanted[r["case"]]
                    key = (int(r["peak_rank"]), float(r["Omega_rad_s"]),
                           float(r["peak_power_fraction"]))
                    if key not in found[gap]:
                        found[gap].append(key)
    for gap in found:
        found[gap] = sorted(found[gap])[:3]
        if not found[gap]:
            found[gap] = [(1, GAP_OMEGA[gap], 1.0)]
    return found


def controlled_coefficients(l, m, radius, H0):
    # This makes the characteristic applied radial H at r=a equal to H0.
    return H0*radius**(1-l)/l if m == 0 else H0*radius**(1-l)/l


def run():
    OUT.mkdir(exist_ok=True)
    p0 = Params()
    a, sigma, mur = p0.ball_R, p0.sigma, p0.mu_r_ac
    rows = []

    # Surface-radius and angular convergence: static l=1+l=2 exact-force case.
    H0, grad = 1e4, 2e6
    alpha_H = 4*np.pi*a**3*(mur-1)/(mur+2)
    F_exact = MU0*alpha_H*H0*grad
    conv = []
    for nmu, nphi in ((12, 24), (18, 36), (24, 48), (32, 64)):
        for rr in (1.05, 1.2, 1.5, 2.0):
            modes = axisymmetric_linear_gradient_modes(H0, grad, 0, a, sigma, mur)
            F, T = maxwell_wrench(modes, a, rr*a, nmu, nphi, False)
            err = norm(F-np.array((0, 0, F_exact)))/abs(F_exact)
            conv.append((nmu, rr, err))
            rows.append(row("stress_convergence", omega_rad_s=0, Fx_N=F[0],
                            Fy_N=F[1], Fz_N=F[2], taux_Nm=T[0], tauy_Nm=T[1],
                            tauz_Nm=T[2], stress_radius_over_a=rr, n_mu=nmu,
                            n_phi=nphi, relative_error=err,
                            note="static source-free l=1+l=2; exact Fz=mu0 alpha H0 G"))

    # Circular uniform-field analytic torque, direct stress, power, tau_lag.
    torque_curves = {B: [] for B in FIELD_AMPLITUDES}
    for B0 in FIELD_AMPLITUDES:
        for om in OMEGAS:
            modes = uniform_field_modes((B0/MU0, -1j*B0/MU0, 0), om,
                                        a, sigma, mur)
            F, T = maxwell_wrench(modes, a, 1.2*a, 24, 48, True)
            ta, power = rotating_uniform_analytic(B0, om, a, sigma, mur)
            tl = tau_lag_rotating_torque(B0, om, a, mur, p0.tau_lag)
            torque_curves[B0].append((om, ta, tl, power, T[2]))
            rows.append(row("rotating_uniform", l=1, omega_rad_s=om,
                            B_amplitude_T=B0, Fx_N=F[0], Fy_N=F[1], Fz_N=F[2],
                            taux_Nm=T[0], tauy_Nm=T[1], tauz_Nm=T[2],
                            joule_power_W=power, stress_radius_over_a=1.2,
                            n_mu=24, n_phi=48, tau_lag_Nm=tl,
                            diffusion_tau_lag_ratio=ta/tl,
                            relative_error=abs(T[2]-ta)/abs(ta),
                            note="circular phasor B=(x-i y)B0; P_J=Omega*tau_z"))

    # Controlled modes.  Real m=0 modes alone have no wrench; adjacent l modes
    # produce force.  A single complex Y_l,l is a travelling angular wave and
    # can supply torque.
    Bc, omc, Hc = .020, 53.41, .020/MU0
    modal = []
    for ell in range(1, 7):
        c0 = controlled_coefficients(ell, 0, a, Hc)
        mode0 = [AppliedMode(ell, 0, c0, response_l(ell, omc, a, sigma, mur))]
        F0, T0 = maxwell_wrench(mode0, a, 1.2*a, 28, 56, True)
        ct = controlled_coefficients(ell, ell, a, Hc)
        travelling = [AppliedMode(ell, ell, ct,
                                  response_l(ell, omc, a, sigma, mur))]
        Ft, Tt = maxwell_wrench(travelling, a, 1.2*a, 28, 56, True)
        modal.append((ell, norm(F0), norm(T0), Tt[2]))
        rows.append(row("isolated_real_mode", l=ell, m=0, omega_rad_s=omc,
                        B_amplitude_T=Bc, Fx_N=F0[0], Fy_N=F0[1], Fz_N=F0[2],
                        taux_Nm=T0[0], tauy_Nm=T0[1], tauz_Nm=T0[2],
                        stress_radius_over_a=1.2, n_mu=28, n_phi=56,
                        note="isolated standing spatial harmonic: neither force nor torque"))
        rows.append(row("travelling_mode", l=ell, m=ell, omega_rad_s=omc,
                        B_amplitude_T=Bc, Fx_N=Ft[0], Fy_N=Ft[1], Fz_N=Ft[2],
                        taux_Nm=Tt[0], tauy_Nm=Tt[1], tauz_Nm=Tt[2],
                        stress_radius_over_a=1.2, n_mu=28, n_phi=56,
                        note="single complex Y_l,l angular travelling wave"))
    adjacent = []
    for ell in range(1, 6):
        modes = []
        for ll in (ell, ell+1):
            modes.append(AppliedMode(ll, 0, controlled_coefficients(ll, 0, a, Hc),
                                     response_l(ll, omc, a, sigma, mur)))
        F, T = maxwell_wrench(modes, a, 1.2*a, 28, 56, True)
        adjacent.append((ell, F[2]))
        rows.append(row("adjacent_mode_coupling", l=ell, m=0,
                        omega_rad_s=omc, B_amplitude_T=Bc,
                        Fx_N=F[0], Fy_N=F[1], Fz_N=F[2],
                        taux_Nm=T[0], tauy_Nm=T[1], tauz_Nm=T[2],
                        stress_radius_over_a=1.2, n_mu=28, n_phi=56,
                        note=f"real l={ell}+{ell+1}: force from quadratic cross term"))

    # Real Carousel snapshots: static bridge, single-frequency correction, and
    # separable top-three spectrum estimate.
    peak_map = measured_peaks()
    snapshot = []
    bridge = []
    lmax_snapshot = []
    for gap, omega in GAP_OMEGA.items():
        p = replace(p0, gap=gap, ball_model="volume_demag",
                    volume_quadrature="medium", demag_resolution="fine")
        sampler = DiagnosticField3D(p)
        vg = VolumeFieldGrid(p, verbose=False)
        ds = DemagSphereSolver(a, p.mu_r, p.demag_resolution)
        sg = PointSetFieldGrid(p, ds.surface_points, verbose=False)
        centre = np.array((p.r_mag, 0., 0.))
        for phase in PHASES:
            def sample(offset, _phase=phase):
                return np.array([sampler.sample(centre[0]+q[0], centre[1]+q[1],
                                                q[2], _phase) for q in offset])
            B_snapshot = norm(sample(np.zeros((1, 3)))[0])
            coefficients = project_radial_field(sample, .95*a, 6, 24, 48)
            smodes = modes_from_coefficients(coefficients, 0, a, sigma, mur)
            Fs, Ts = maxwell_wrench(smodes, a, 1.05*a, 28, 56, False)
            vol = static_magnetic_response(p, centre, phase, vg, sg, ds)
            ferr = norm(Fs-vol["force"])/max(norm(vol["force"]), 1e-300)
            bridge.append((gap, phase, Fs.copy(), vol["force"].copy(), ferr))
            rows.append(row("carousel_static_bridge", case=f"gap_{gap*1e3:g}mm",
                            gap_m=gap, phase_rad=phase, omega_rad_s=0,
                            B_amplitude_T=B_snapshot,
                            Fx_N=Fs[0], Fy_N=Fs[1], Fz_N=Fs[2],
                            taux_Nm=Ts[0], tauy_Nm=Ts[1], tauz_Nm=Ts[2],
                            stress_radius_over_a=1.05, n_mu=28, n_phi=56,
                            volume_Fx_N=vol["force"][0], volume_Fy_N=vol["force"][1],
                            volume_Fz_N=vol["force"][2], relative_error=ferr,
                            note="l<=6 modal Maxwell stress vs static volume_demag"))
            # Static phasor uses half the instantaneous stress and is the proper
            # no-diffusion denominator for a same-amplitude harmonic field.
            Fref, Tref = maxwell_wrench(smodes, a, 1.05*a, 28, 56, True)
            amodes = modes_from_coefficients(coefficients, omega, a, sigma, mur)
            Fa, Ta = maxwell_wrench(amodes, a, 1.05*a, 28, 56, True)
            ratio = norm(Fa)/max(norm(Fref), 1e-300)
            snapshot.append((gap, phase, ratio, Ta.copy(), Fa.copy(), Fref.copy()))
            rows.append(row("carousel_single_frequency", case=f"gap_{gap*1e3:g}mm",
                            gap_m=gap, phase_rad=phase, omega_rad_s=omega,
                            B_amplitude_T=B_snapshot,
                            Fx_N=Fa[0], Fy_N=Fa[1], Fz_N=Fa[2],
                            taux_Nm=Ta[0], tauy_Nm=Ta[1], tauz_Nm=Ta[2],
                            stress_radius_over_a=1.05, n_mu=28, n_phi=56,
                            static_diffusion_ratio=ratio,
                            note="one-frequency scale estimate; denominator is zero-frequency phasor"))
            if phase == 0.0:
                for lmax in range(1, 7):
                    ms_l = modes_from_coefficients(coefficients, 0, a, sigma, mur,
                                                   keep_l=range(1, lmax+1))
                    ma_l = modes_from_coefficients(coefficients, omega, a, sigma, mur,
                                                   keep_l=range(1, lmax+1))
                    Fsl, Tsl = maxwell_wrench(ms_l, a, 1.05*a, 28, 56, False)
                    Fal, Tal = maxwell_wrench(ma_l, a, 1.05*a, 28, 56, True)
                    lmax_snapshot.append((gap, lmax, norm(Fsl), norm(Fal)))
                    rows.append(row("carousel_lmax_convergence",
                                    case=f"gap_{gap*1e3:g}mm", gap_m=gap,
                                    phase_rad=phase, l=lmax, omega_rad_s=omega,
                                    B_amplitude_T=B_snapshot,
                                    Fx_N=Fal[0], Fy_N=Fal[1], Fz_N=Fal[2],
                                    taux_Nm=Tal[0], tauy_Nm=Tal[1], tauz_Nm=Tal[2],
                                    stress_radius_over_a=1.05, n_mu=28, n_phi=56,
                                    static_diffusion_ratio=norm(Fal)/max(.5*norm(Fsl), 1e-300),
                                    note="cumulative spatial harmonics l=1..listed l"))
            peaks = peak_map[gap]
            weights = np.array([q[2] for q in peaks]); weights /= weights.sum()
            Fsum = np.zeros(3); Tsum = np.zeros(3)
            for weight, (_, omp, _) in zip(weights, peaks):
                mm = modes_from_coefficients(coefficients, omp, a, sigma, mur)
                Fp, Tp = maxwell_wrench(mm, a, 1.05*a, 28, 56, True)
                Fsum += weight*Fp; Tsum += weight*Tp
                rows.append(row("carousel_spectral_component",
                                case=f"gap_{gap*1e3:g}mm", gap_m=gap,
                                phase_rad=phase, omega_rad_s=omp,
                                Fx_N=weight*Fp[0], Fy_N=weight*Fp[1], Fz_N=weight*Fp[2],
                                taux_Nm=weight*Tp[0], tauy_Nm=weight*Tp[1],
                                tauz_Nm=weight*Tp[2], spectral_weight=weight,
                                stress_radius_over_a=1.05, n_mu=28, n_phi=56,
                                note="weighted independent-frequency contribution"))
            rows.append(row("carousel_spectral_sum", case=f"gap_{gap*1e3:g}mm",
                            gap_m=gap, phase_rad=phase,
                            Fx_N=Fsum[0], Fy_N=Fsum[1], Fz_N=Fsum[2],
                            taux_Nm=Tsum[0], tauy_Nm=Tsum[1], tauz_Nm=Tsum[2],
                            static_diffusion_ratio=norm(Fsum)/max(norm(Fref), 1e-300),
                            stress_radius_over_a=1.05, n_mu=28, n_phi=56,
                            note="offline separable-spectrum sum; not transient simulation"))

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # Eight compact evidence panels.
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    a0 = ax.ravel()
    for nmu in sorted(set(q[0] for q in conv)):
        q = [x for x in conv if x[0] == nmu]
        a0[0].semilogy([x[1] for x in q], [max(x[2], 1e-16) for x in q], "o-", label=f"nmu={nmu}")
    a0[0].set(xlabel="stress radius / a", ylabel="relative force error", title="A  Stress convergence"); a0[0].legend(fontsize=7)
    for B, q in torque_curves.items():
        a0[1].plot([x[0] for x in q], [x[1] for x in q], "o-", label=f"{B*1e3:g} mT")
    a0[1].set(xlabel="Omega [rad/s]", ylabel="tau diffusion [N m]", title="B  Rotating uniform field"); a0[1].legend(fontsize=7)
    q = torque_curves[.020]
    a0[2].plot([x[0]*x[1] for x in q], [x[3] for x in q], "o")
    lim = max(x[3] for x in q); a0[2].plot([0, lim], [0, lim], "k--")
    a0[2].set(xlabel="Omega tau [W]", ylabel="Joule power [W]", title="C  Power consistency")
    a0[3].semilogy([x[0] for x in q], [x[1] for x in q], "o-", label="diffusion")
    a0[3].semilogy([x[0] for x in q], [x[2] for x in q], "s-", label="tau_lag")
    a0[3].set(xlabel="Omega [rad/s]", ylabel="torque [N m]", title="D  Same 20 mT field"); a0[3].legend(fontsize=7)
    labels = [f"{g*1e3:g}/{ph*180/np.pi:.1f}" for g, ph, *_ in bridge]
    a0[4].plot(range(len(bridge)), [norm(x[2]) for x in bridge], "o-", label="modal stress")
    a0[4].plot(range(len(bridge)), [norm(x[3]) for x in bridge], "s--", label="volume_demag")
    a0[4].set_xticks(range(len(labels)), labels, rotation=70, fontsize=6)
    a0[4].set(ylabel="|F| [N]", title="E  Static Carousel bridge"); a0[4].legend(fontsize=7)
    a0[5].plot(range(len(snapshot)), [x[2] for x in snapshot], "o-")
    a0[5].axhline(1, color="k", lw=.8); a0[5].set_xticks(range(len(labels)), labels, rotation=70, fontsize=6)
    a0[5].set(ylabel="|Fdiff| / |Fstatic|", title="F  Snapshot force ratio")
    a0[6].plot(range(len(snapshot)), [norm(x[3]) for x in snapshot], "o-")
    a0[6].set_xticks(range(len(labels)), labels, rotation=70, fontsize=6)
    a0[6].set(ylabel="|tau| [N m]", title="G  Snapshot diffusion torque")
    for gap in GAP_OMEGA:
        ql = [x for x in lmax_snapshot if x[0] == gap]
        final = ql[-1][2]
        a0[7].plot([x[1] for x in ql], [x[2]/final for x in ql], "o-",
                   label=f"{gap*1e3:g} mm")
    a0[7].set(xlabel="cumulative lmax", ylabel="static |F| / lmax=6",
              title="H  Real-field spatial modes"); a0[7].legend(fontsize=7)
    fig.suptitle("Magnetic Carousel Step 5B-2: offline diffusion wrench")
    fig.savefig(PNG_PATH, dpi=180); plt.close(fig)

    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    print(f"wrote {PNG_PATH}")
    print("bridge max relative force error", max(x[4] for x in bridge))
    print("snapshot force-ratio range", min(x[2] for x in snapshot), max(x[2] for x in snapshot))
    print("snapshot torque range", min(norm(x[3]) for x in snapshot), max(norm(x[3]) for x in snapshot))


if __name__ == "__main__":
    run()

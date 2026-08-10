"""Step 5A: material-frame spectrum and skin-depth diagnostic only.

This script does not modify or couple a skin model to the production dynamics.
It reconstructs a body->lab orientation from point-model trajectories, samples
the existing rotating cuboid FieldGrid at body-fixed material points, and
computes frequency-dependent penetration scalings.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from magnetic_carousel import (MU0, FieldGrid, Params, VolumeFieldGrid,
                               _field_lab, simulate)

OUT = Path("out")
TRAJ_CACHE = OUT/"skin_trajectory_cache"
CASES = (
    ("A_locked_3mm_2", .003, 2.0),
    ("B_transition_5mm_7p8", .005, 7.8),
    ("C_retrograde_8mm_16", .008, 16.0),
    ("D_weak_16mm_10", .016, 10.0),
    ("E_5mm_16", .005, 16.0),
)
SIGMAS = (1e6, 3e6, 6e6, 1e7)
MU_R_AC = (10.0, 50.0, 100.0, 500.0)
REFERENCE_SIGMA = 6e6
REFERENCE_MUR = 100.0
T_END = 8.0
T_EXCLUDE = 4.0
DT = 2e-5
STRIDE = 25
MAX_SPECTRUM_HZ = 100.0


def quat_mul(a, b):
    """Scalar-first Hamilton product a tensor b."""
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array((aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw))


def quat_matrix(q):
    """Active rotation matrix mapping body components to lab components."""
    w, x, y, z = q
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def reconstruct_orientation(t, omega_lab):
    """Integrate qdot=.5*[0,omega_lab] tensor q, with q body->lab."""
    q = np.empty((len(t), 4)); q[0] = (1.0, 0.0, 0.0, 0.0)
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        wmid = .5*(omega_lab[i]+omega_lab[i+1])
        wn = np.linalg.norm(wmid)
        if wn*dt < 1e-12:
            dq = np.array((1.0, *(0.5*dt*wmid)))
        else:
            half = .5*wn*dt
            dq = np.array((np.cos(half), *(np.sin(half)*wmid/wn)))
        q[i+1] = quat_mul(dq, q[i])
        q[i+1] /= np.linalg.norm(q[i+1])
    return q


class DiagnosticField3D:
    """Linear-z interpolation between existing FieldGrid planes."""

    def __init__(self, p):
        pv = replace(p, ball_model="volume_independent", volume_quadrature="medium")
        vg = VolumeFieldGrid(pv, verbose=False)
        z = list(vg.z_offsets); fields = [vg.F[i] for i in range(len(z))]
        for zoff in (-p.ball_R, p.ball_R):
            ps = replace(p, gap=p.gap+zoff, ball_model="point")
            g = FieldGrid(ps, verbose=False)
            z.append(zoff); fields.append(g.F)
        order = np.argsort(z)
        self.z = np.asarray(z)[order]
        self.F = np.stack([fields[i] for i in order])
        self.x0, self.y0, self.dx, self.dy = vg.x0, vg.y0, vg.dx, vg.dy

    def sample(self, x, y, zoff, angle):
        k = int(np.searchsorted(self.z, zoff))
        k = min(max(k, 1), len(self.z)-1)
        z0, z1 = self.z[k-1], self.z[k]
        f = (zoff-z0)/(z1-z0)
        b0, _ = _field_lab(self.F[k-1], self.x0, self.y0, self.dx, self.dy,
                           x, y, angle)
        b1, _ = _field_lab(self.F[k], self.x0, self.y0, self.dx, self.dy,
                           x, y, angle)
        return (1-f)*b0+f*b1


def load_trajectory(name, gap, omega):
    TRAJ_CACHE.mkdir(parents=True, exist_ok=True)
    fn = TRAJ_CACHE/f"{name}_T{T_END:g}_dt{DT:g}_s{STRIDE}.npz"
    p = Params(gap=gap, omega=omega, ball_model="point")
    if fn.exists():
        with np.load(fn) as d:
            return p, d["t"], d["state"]
    result = simulate(p, t_end=T_END, dt=DT, stride=STRIDE,
                      grid=FieldGrid(p, verbose=False))
    np.savez_compressed(fn, t=result.t, state=result.state)
    return p, result.t, result.state


def material_points(radius):
    return {
        "center": np.zeros(3),
        "+body_x": np.array((radius, 0., 0.)),
        "-body_x": np.array((-radius, 0., 0.)),
        "+body_y": np.array((0., radius, 0.)),
        "-body_y": np.array((0., -radius, 0.)),
        "+body_z": np.array((0., 0., radius)),
        "-body_z": np.array((0., 0., -radius)),
        "inner_xyz": radius*.45*np.array((1., 1., 1.))/np.sqrt(3),
    }


def sample_material_fields(p, t, state, q):
    sampler = DiagnosticField3D(p)
    pts = material_points(p.ball_R)
    out = {name: np.empty((len(t), 3)) for name in pts}
    for it, ti in enumerate(t):
        R = quat_matrix(q[it])
        for name, rb in pts.items():
            rl = R@rb
            Bl = sampler.sample(state[it, 0]+rl[0], state[it, 1]+rl[1],
                                rl[2], p.omega*ti)
            out[name][it] = R.T@Bl
    return out


def spectrum(t, B):
    dt = np.median(np.diff(t)); n = len(t)
    window = np.hanning(n)
    X = np.fft.rfft((B-B.mean(axis=0))*window[:, None], axis=0)
    power = np.sum(np.abs(X)**2, axis=1)
    freq = np.fft.rfftfreq(n, dt)
    keep = (freq > 0) & (freq <= MAX_SPECTRUM_HZ)
    freq, power = freq[keep], power[keep]
    total = power.sum()
    power = power/max(total, 1e-300)
    pk, _ = find_peaks(power, distance=2, prominence=max(power.max()*1e-5, 1e-15))
    if len(pk) == 0:
        pk = np.array([int(np.argmax(power))])
    pk = pk[np.argsort(power[pk])[::-1]]
    selected, used = [], np.zeros(len(power), dtype=bool)
    for k in pk:
        lo, hi = max(0, k-1), min(len(power), k+2)
        if used[lo:hi].any():
            continue
        band_power = power[lo:hi].sum()
        selected.append(dict(index=int(k), frequency_hz=freq[k],
                             Omega=2*np.pi*freq[k], power=band_power))
        used[lo:hi] = True
        if len(selected) == 5:
            break
    omega = 2*np.pi*freq
    omega_rms = np.sqrt(np.sum(power*omega**2)/max(np.sum(power), 1e-300))
    return freq, power, selected, omega_rms


def skin_values(Omega, sigma, mur, radius, tau_lag):
    if Omega <= 1e-14:
        return dict(delta=np.inf, delta_over_a=np.inf, a_over_delta=0.0,
                    tau_diff=MU0*mur*sigma*radius**2, Pi1=0.0,
                    Omega_tau_lag=0.0, center_amplitude=1.0,
                    center_phase_rad=0.0, shell_fraction=1.0)
    tau_diff = MU0*mur*sigma*radius**2
    delta = np.sqrt(2.0/(Omega*MU0*mur*sigma))
    da = delta/radius; ad = radius/delta
    shell = min(delta, radius)
    shell_fraction = 1.0-(1.0-shell/radius)**3
    return dict(delta=delta, delta_over_a=da, a_over_delta=ad,
                tau_diff=tau_diff, Pi1=Omega*tau_diff,
                Omega_tau_lag=Omega*tau_lag,
                center_amplitude=np.exp(-ad), center_phase_rad=-ad,
                shell_fraction=shell_fraction)


def analyze_case(case):
    name, gap, omega = case
    p, t, state = load_trajectory(name, gap, omega)
    q = reconstruct_orientation(t, state[:, 4:7])
    fields = sample_material_fields(p, t, state, q)
    mask = t >= T_EXCLUDE
    ta = t[mask]
    spectra = {}
    for point, B in fields.items():
        spectra[point] = spectrum(ta, B[mask])
    aggregate_power = np.mean([v[1] for v in spectra.values()], axis=0)
    aggregate_power /= aggregate_power.sum()
    freq = next(iter(spectra.values()))[0]
    # Re-run peak selection through a synthetic one-component signal is not
    # appropriate; select local maxima directly from aggregate vector power.
    pk, _ = find_peaks(aggregate_power, distance=2,
                       prominence=max(aggregate_power.max()*1e-5, 1e-15))
    pk = pk[np.argsort(aggregate_power[pk])[::-1]]
    peaks, used = [], np.zeros(len(freq), bool)
    for k in pk:
        lo, hi = max(0, k-1), min(len(freq), k+2)
        if used[lo:hi].any(): continue
        peaks.append(dict(index=int(k), frequency_hz=freq[k],
                          Omega=2*np.pi*freq[k], power=aggregate_power[lo:hi].sum()))
        used[lo:hi] = True
        if len(peaks) == 5: break
    omega_orbit = np.polyfit(ta, np.unwrap(np.arctan2(state[mask, 1], state[mask, 0])), 1)[0]
    spin_mag = np.mean(np.linalg.norm(state[mask, 4:7], axis=1))
    tm = .5*(ta[0]+ta[-1])
    early = mask & (t <= tm); late = mask & (t >= tm)
    phi_all = np.unwrap(np.arctan2(state[:, 1], state[:, 0]))
    orbit_early = np.polyfit(t[early], phi_all[early], 1)[0]
    orbit_late = np.polyfit(t[late], phi_all[late], 1)[0]
    spin_early = np.mean(np.linalg.norm(state[early, 4:7], axis=1))
    spin_late = np.mean(np.linalg.norm(state[late, 4:7], axis=1))
    omega_rms = np.sqrt(np.sum(aggregate_power*(2*np.pi*freq)**2))
    return dict(name=name, p=p, t=t, state=state, q=q, fields=fields, mask=mask,
                freq=freq, aggregate_power=aggregate_power, peaks=peaks,
                point_spectra=spectra, omega_orbit=omega_orbit,
                spin_mag=spin_mag, omega_rms=omega_rms,
                orbit_early=orbit_early, orbit_late=orbit_late,
                spin_early=spin_early, spin_late=spin_late,
                omega6=6*abs(omega-omega_orbit),
                omega18=18*abs(omega-omega_orbit))


def rows_for_cases(results):
    rows = []
    for result in results:
        p = result["p"]
        for point, (_, _, peaks, omega_rms) in result["point_spectra"].items():
            cumulative = np.cumsum([x["power"] for x in peaks[:3]])
            for rank, peak in enumerate(peaks[:3], 1):
                for sigma in SIGMAS:
                    for mur in MU_R_AC:
                        sv = skin_values(peak["Omega"], sigma, mur,
                                         p.ball_R, p.tau_lag)
                        rows.append(dict(row_type="spectral_peak", case=result["name"],
                                         gap_m=p.gap, omega_disc= p.omega,
                                         material_point=point, peak_rank=rank,
                                         frequency_hz=peak["frequency_hz"],
                                         Omega_rad_s=peak["Omega"],
                                         peak_power_fraction=peak["power"],
                                         cumulative_top_power=cumulative[rank-1],
                                         Omega_rms=omega_rms,
                                         omega_orbit=result["omega_orbit"],
                                         omega_orbit_early=result["orbit_early"],
                                         omega_orbit_late=result["orbit_late"],
                                         mean_spin_magnitude=result["spin_mag"],
                                         spin_early=result["spin_early"],
                                         spin_late=result["spin_late"],
                                         Omega6_orbit=result["omega6"],
                                         Omega18_orbit=result["omega18"],
                                         sigma_S_m=sigma, mu_r_ac=mur, **sv))
        for rank, peak in enumerate(result["peaks"][:3], 1):
            sv = skin_values(peak["Omega"], REFERENCE_SIGMA, REFERENCE_MUR,
                             p.ball_R, p.tau_lag)
            rows.append(dict(row_type="case_aggregate", case=result["name"],
                             gap_m=p.gap, omega_disc=p.omega,
                             material_point="aggregate", peak_rank=rank,
                             frequency_hz=peak["frequency_hz"],
                             Omega_rad_s=peak["Omega"],
                             peak_power_fraction=peak["power"],
                             cumulative_top_power=np.sum([x["power"] for x in result["peaks"][:rank]]),
                             Omega_rms=result["omega_rms"],
                             omega_orbit=result["omega_orbit"],
                             omega_orbit_early=result["orbit_early"],
                             omega_orbit_late=result["orbit_late"],
                             mean_spin_magnitude=result["spin_mag"],
                             spin_early=result["spin_early"], spin_late=result["spin_late"],
                             Omega6_orbit=result["omega6"],
                             Omega18_orbit=result["omega18"],
                             sigma_S_m=REFERENCE_SIGMA, mu_r_ac=REFERENCE_MUR, **sv))
    return rows


def write_csv(rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    OUT.mkdir(exist_ok=True)
    with (OUT/"skin_effect_diagnostic.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def make_plot(results):
    fig, ax = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    r0 = results[0]; m = r0["mask"]
    tt = r0["t"][m]; B = r0["fields"]["center"][m]*1e3
    show = tt <= tt[0]+1.0
    for j, label in enumerate(("Bx", "By", "Bz")):
        ax[0, 0].plot(tt[show]-tt[0], B[show, j], label=label)
    ax[0, 0].set(title="A. Locked-case center field in body frame",
                 xlabel="analysis time [s]", ylabel="B [mT]")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

    for c, r in zip(colors, results):
        ax[0, 1].semilogy(r["freq"], np.maximum(r["aggregate_power"], 1e-14),
                          color=c, label=r["name"].split("_")[0])
    ax[0, 1].set_xlim(0, 40); ax[0, 1].set(title="B. Aggregate material-frame spectra",
                 xlabel="frequency [Hz]", ylabel="normalized vector power/bin")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    x = np.arange(len(results)); measured = [r["peaks"][0]["Omega"] for r in results]
    ax[0, 2].plot(x, measured, "o-", label="measured dominant")
    ax[0, 2].plot(x, [r["omega6"] for r in results], "s--", label=r"$6|\omega_d-\omega_o|$")
    ax[0, 2].plot(x, [r["omega18"] for r in results], "^--", label=r"$18|\omega_d-\omega_o|$")
    ax[0, 2].plot(x, [r["spin_mag"] for r in results], "d:", label=r"mean $|\omega_b|$")
    ax[0, 2].set_xticks(x, [r["name"].split("_")[0] for r in results])
    ax[0, 2].set(title="C. Measured and kinematic frequencies", ylabel=r"$\Omega$ [rad/s]")
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=.3)

    ref = [skin_values(o, REFERENCE_SIGMA, REFERENCE_MUR, results[i]["p"].ball_R,
                       results[i]["p"].tau_lag) for i, o in enumerate(measured)]
    ax[0, 3].plot(x, [s["delta_over_a"] for s in ref], "o-")
    ax[0, 3].axhline(1, color="k", ls="--")
    ax[0, 3].set_xticks(x, [r["name"].split("_")[0] for r in results])
    ax[0, 3].set(title="D. Reference skin depth", xlabel="operating case",
                 ylabel=r"$\delta/a$"); ax[0, 3].grid(alpha=.3)

    depth = np.linspace(0, 1, 200)
    for c, r, s in zip(colors, results, ref):
        profile = np.exp(-(1+1j)*depth/s["delta_over_a"])
        ax[1, 0].plot(depth, np.abs(profile), color=c, label=r["name"].split("_")[0])
        ax[1, 1].plot(depth, np.angle(profile), color=c, label=r["name"].split("_")[0])
    ax[1, 0].set(title="E. Harmonic penetration amplitude", xlabel="depth s/a",
                 ylabel=r"$|B/B_s|$"); ax[1, 0].grid(alpha=.3); ax[1, 0].legend()
    ax[1, 1].set(title="F. Harmonic penetration phase", xlabel="depth s/a",
                 ylabel="phase [rad]"); ax[1, 1].grid(alpha=.3)

    lows, highs = [], []
    for r, om in zip(results, measured):
        vals = [skin_values(om, sig, mur, r["p"].ball_R, r["p"].tau_lag)["delta_over_a"]
                for sig in SIGMAS for mur in MU_R_AC]
        lows.append(min(vals)); highs.append(max(vals))
    mid = [s["delta_over_a"] for s in ref]
    ax[1, 2].errorbar(x, mid, yerr=[np.array(mid)-lows, np.array(highs)-mid], fmt="o")
    ax[1, 2].axhline(1, color="k", ls="--")
    ax[1, 2].set_xticks(x, [r["name"].split("_")[0] for r in results])
    ax[1, 2].set_yscale("log"); ax[1, 2].set(title="G. Material uncertainty range",
                 ylabel=r"$\delta/a$"); ax[1, 2].grid(True, which="both", alpha=.3)

    ax[1, 3].plot(x, [s["Omega_tau_lag"] for s in ref], "o-", label=r"$\Omega\tau_{lag}$")
    ax[1, 3].plot(x, [s["Pi1"] for s in ref], "s-", label=r"$\Pi_1=\Omega\tau_{diff}$")
    ax[1, 3].axhline(1, color="k", ls="--")
    ax[1, 3].set_xticks(x, [r["name"].split("_")[0] for r in results])
    ax[1, 3].set_yscale("log"); ax[1, 3].set(title="H. Lag and diffusion scales")
    ax[1, 3].legend(); ax[1, 3].grid(True, which="both", alpha=.3)
    fig.suptitle("Material-frame magnetic spectrum and skin-depth diagnostic", fontsize=15)
    fig.savefig(OUT/"skin_effect_diagnostic.png", dpi=180)


def main():
    results = []
    for case in CASES:
        print(f"analyzing {case[0]} ...", flush=True)
        results.append(analyze_case(case))
    rows = rows_for_cases(results)
    write_csv(rows); make_plot(results)
    sample_dt = DT*STRIDE
    print(f"sampling={1/sample_dt:.1f} Hz Nyquist={.5/sample_dt:.1f} Hz "
          f"resolution={1/(T_END-T_EXCLUDE):.3f} Hz window=Hann")
    for r in results:
        peak = r["peaks"][0]
        sv = skin_values(peak["Omega"], REFERENCE_SIGMA, REFERENCE_MUR,
                         r["p"].ball_R, r["p"].tau_lag)
        cum = np.cumsum([p["power"] for p in r["peaks"][:3]])
        print(f"{r['name']}: orbit={r['omega_orbit']:+.4f} spin={r['spin_mag']:.3f} "
              f"peak={peak['Omega']:.3f} rad/s ({peak['frequency_hz']:.3f} Hz) "
              f"power1/2/3={cum} delta/a={sv['delta_over_a']:.3f} "
              f"Pi1={sv['Pi1']:.3f} Omega*tau={sv['Omega_tau_lag']:.3f}")
        print(f"  stationarity: orbit halves {r['orbit_early']:+.3f}/{r['orbit_late']:+.3f}, "
              f"spin halves {r['spin_early']:.2f}/{r['spin_late']:.2f}")
    print(f"wrote {OUT/'skin_effect_diagnostic.csv'}")
    print(f"wrote {OUT/'skin_effect_diagnostic.png'}")


if __name__ == "__main__":
    main()

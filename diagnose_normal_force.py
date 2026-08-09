"""Audit vertical magnetic force, normal load, and friction capacity.

This diagnostic uses the existing FieldGrid, force calculation, contact law, and
integrator without changing simulation behaviour.  Reported steady statistics
use the final half of each run.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_carousel import G_ACC, FieldGrid, Params, simulate


GAPS = (0.003, 0.005, 0.008, 0.010, 0.016)
OMEGAS = (2.0, 5.0, 10.0, 16.0)
T_END = 1.6
# Match the parameter-sweep integration resolution.  The optional numba
# accelerator makes the full matrix substantially faster but is not required.
DT = 2.5e-5
STRIDE = 40
REPRESENTATIVE = {
    (0.003, 2.0): "small-gap prograde",
    (0.005, 5.0): "transition-near",
    (0.008, 16.0): "retrograde",
}


def stats(a: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.mean(a)), float(np.sqrt(np.mean(a**2))), float(np.min(a)), float(np.max(a))


def analyze_run(p: Params, grid: FieldGrid):
    result = simulate(p, t_end=T_END, dt=DT, stride=STRIDE, grid=grid)
    cut = len(result.t) // 2
    sl = slice(cut, None)
    weight = p.mass * G_ACC
    fz = result.Fmag[sl, 2]
    n_phys_raw = weight - fz
    n_current = result.Nload[sl]
    slip = result.slip[sl]

    # Exact magnitude used by the current regularized Coulomb law.
    friction_current = p.mu_k * n_current * np.tanh(slip / p.u_reg)
    # Offline comparison requested in the audit.  It should be identical because
    # the existing code already uses max(mg-Fz, 0).
    n_corrected = np.maximum(n_phys_raw, 0.0)
    friction_corrected = p.mu_k * n_corrected * np.tanh(slip / p.u_reg)

    fmean, frms, fmin, fmax = stats(fz)
    nmean, _, nmin, nmax = stats(n_phys_raw)
    ratio = np.divide(n_phys_raw, n_current, out=np.full_like(n_phys_raw, np.nan),
                      where=n_current > 0)
    current_over_mg = n_current / weight
    return {
        "result": result,
        "slice": sl,
        "Fz_mean": fmean,
        "Fz_rms": frms,
        "Fz_min": fmin,
        "Fz_max": fmax,
        "Fz_mean_mg": fmean / weight,
        "Fz_rms_mg": frms / weight,
        "N_mean_mg": nmean / weight,
        "N_min_mg": nmin / weight,
        "N_max_mg": nmax / weight,
        "contact_loss": bool(np.any(n_phys_raw <= 0.0)),
        # Enhancement relative to a hypothetical gravity-only normal load.
        "enh_mean": float(np.mean(n_phys_raw / weight)),
        "enh_min": float(np.min(n_phys_raw / weight)),
        "enh_max": float(np.max(n_phys_raw / weight)),
        # Corrected/current ratio: expected to be exactly one in contact.
        "corrected_current_mean": float(np.nanmean(ratio)),
        "corrected_current_min": float(np.nanmin(ratio)),
        "corrected_current_max": float(np.nanmax(ratio)),
        "slip_mean": float(np.mean(slip)),
        "slip_rms": float(np.sqrt(np.mean(slip**2))),
        "friction_mean": float(np.mean(friction_current)),
        "friction_max": float(np.max(friction_current)),
        "offline_friction_mean": float(np.mean(friction_corrected)),
        "offline_friction_max": float(np.max(friction_corrected)),
        "friction_difference_max": float(np.max(np.abs(friction_corrected-friction_current))),
        "current_N_mean_mg": float(np.mean(current_over_mg)),
    }


def quasi_static_force_harmonics(grid: FieldGrid, p: Params, n_phi: int = 3072):
    """Fz(phi) for a fixed ball with its moment instantaneously equilibrated.

    This isolates the spatial normal-force harmonics from orbital motion and lag.
    It uses the same sampled B, reconstructed gradient tensor, saturation rule,
    and F_i=sum_j m_j*dB_j/dx_i as the dynamical implementation.
    """
    phi, q = grid.sample_circle(p.r_mag, n_phi)
    b = q[:, :3]
    bnorm = np.linalg.norm(b, axis=1)
    meq_mag = np.minimum(p.alpha * bnorm, p.m_sat)
    m = meq_mag[:, None] * b / np.maximum(bnorm[:, None], 1e-18)
    # J[z,:] = (dBx/dz,dBy/dz,dBz/dz)
    jz = np.column_stack((q[:, 6], q[:, 7], -(q[:, 3] + q[:, 4])))
    fz = np.sum(jz * m, axis=1)
    coeff = np.fft.rfft(fz) / len(fz)
    amp = 2.0 * np.abs(coeff)
    amp[0] = np.abs(coeff[0])
    order = np.argsort(amp[1:])[::-1][:10] + 1
    return phi, fz, amp, order


def print_table(rows: list[tuple[Params, dict]]) -> None:
    print("\nSteady-state force/load table (final 50% of each run)")
    print("gap  omega   Fz_mean    Fz_rms     Fz_min     Fz_max   "
          "Fz_mean/mg Fz_rms/mg  Nmean/mg Nmin/mg Nmax/mg loss")
    print(" mm  rad/s      [N]        [N]        [N]        [N]")
    for p, d in rows:
        print(f"{p.gap*1e3:3.0f} {p.omega:6.1f} "
              f"{d['Fz_mean']:+10.4e} {d['Fz_rms']:10.4e} "
              f"{d['Fz_min']:+10.4e} {d['Fz_max']:+10.4e} "
              f"{d['Fz_mean_mg']:+10.4f} {d['Fz_rms_mg']:10.4f} "
              f"{d['N_mean_mg']:8.4f} {d['N_min_mg']:8.4f} {d['N_max_mg']:8.4f} "
              f"{'YES' if d['contact_loss'] else 'no'}")

    print("\nFriction-capacity ratios")
    print("gap omega   Nphys/(mg), mean/min/max       corrected/current, mean/min/max")
    for p, d in rows:
        print(f"{p.gap*1e3:3.0f} {p.omega:5.1f}   "
              f"{d['enh_mean']:8.4f}/{d['enh_min']:8.4f}/{d['enh_max']:8.4f}       "
              f"{d['corrected_current_mean']:8.4f}/"
              f"{d['corrected_current_min']:8.4f}/"
              f"{d['corrected_current_max']:8.4f}")


def make_figure(rows: list[tuple[Params, dict]], phi, static_fz, output: str) -> None:
    weight = Params().mass * G_ACC
    rep = next(d for p, d in rows if np.isclose(p.gap, 0.008) and np.isclose(p.omega, 10.0))
    result, sl = rep["result"], rep["slice"]
    time = result.t[sl]
    fz = result.Fmag[sl, 2]
    nphys = weight - fz
    slip = result.slip[sl]
    current_capacity = result.p.mu_k * result.Nload[sl]
    corrected_capacity = result.p.mu_k * np.maximum(nphys, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    ax = axes[0, 0]
    ax.plot(np.degrees(phi), static_fz / weight, lw=1.0)
    ax.set(title="A. Spatial vertical-force cycle (default gap)",
           xlabel="disc-frame angle [deg]", ylabel=r"$F_{mag,z}/(mg)$", xlim=(0, 360))

    ax = axes[0, 1]
    ax.plot(time, nphys / weight, lw=1.0)
    ax.set(title=r"B. Dynamic normal load: gap 8 mm, $\omega=10$ rad/s",
           xlabel="time [s]", ylabel=r"$N_{phys}/(mg)$")

    ax = axes[1, 0]
    ax.plot(time, current_capacity, lw=1.2, label="current code")
    ax.plot(time, corrected_capacity, "--", lw=1.0, label=r"offline $\mu\max(mg-F_z,0)$")
    ax2 = ax.twinx()
    ax2.plot(time, slip * 1e3, color="0.55", alpha=0.55, lw=0.8, label="slip")
    ax.set(title="C. Current and physically implied capacity coincide",
           xlabel="time [s]", ylabel="Coulomb capacity [N]")
    ax2.set_ylabel("contact slip [mm/s]", color="0.4")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for omega in OMEGAS:
        x = [p.gap * 1e3 for p, _ in rows if np.isclose(p.omega, omega)]
        y = [d["N_mean_mg"] for p, d in rows if np.isclose(p.omega, omega)]
        ax.plot(x, y, "o-", ms=4, label=fr"$\omega={omega:g}$ rad/s")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set(title="D. Magnetic enhancement of mean normal load",
           xlabel="gap [mm]", ylabel=r"mean $N_{phys}/(mg)$")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = Params()
    weight = base.mass * G_ACC
    print("Normal-force diagnostic (existing model unchanged)")
    print(f"mass={base.mass:.9e} kg  weight={weight:.9e} N")
    print("coordinate convention: magnet top z=0, magnet centres z<0, ball centre z>0")
    print("current contact law: N_current=max(mg-F_mag,z,0)")

    rows: list[tuple[Params, dict]] = []
    grids: dict[float, FieldGrid] = {}
    for gap in GAPS:
        grid_p = Params(gap=gap)
        grids[gap] = FieldGrid(grid_p, verbose=True)
        for omega in OMEGAS:
            p = Params(gap=gap, omega=omega)
            d = analyze_run(p, grids[gap])
            rows.append((p, d))
            print(f"  completed gap={gap*1e3:g} mm omega={omega:g} rad/s", flush=True)

    print_table(rows)

    print("\nRepresentative contact/slip cases")
    print("case                 gap omega slip_mean slip_rms friction_mean friction_max "
          "offline_mean offline_max max_difference")
    print("                              [mm/s]   [mm/s]       [N]          [N]          [N]        [N]         [N]")
    for p, d in rows:
        label = REPRESENTATIVE.get((p.gap, p.omega))
        if label:
            print(f"{label:<20} {p.gap*1e3:3.0f} {p.omega:5.1f} "
                  f"{d['slip_mean']*1e3:9.4f} {d['slip_rms']*1e3:9.4f} "
                  f"{d['friction_mean']:12.4e} {d['friction_max']:12.4e} "
                  f"{d['offline_friction_mean']:12.4e} {d['offline_friction_max']:12.4e} "
                  f"{d['friction_difference_max']:12.4e}")

    phi, static_fz, amp, order = quasi_static_force_harmonics(grids[0.008], Params(gap=0.008))
    dc = amp[0]
    peak = amp[order[0]]
    print("\nDefault-gap F_mag,z spatial Fourier analysis")
    print("fixed ball at r_mag; instantaneous existing equilibrium-moment/saturation law")
    print(f"DC signed mean={np.mean(static_fz):+.6e} N  |DC|={dc:.6e} N")
    print("mode n : amplitude [N] : amplitude/largest_nonDC : amplitude/|DC|")
    for n in order:
        print(f"{n:6d} : {amp[n]:.6e} : {amp[n]/peak:.6e} : {amp[n]/dc:.6e}")

    output = "out/normal_force_diagnostic.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    make_figure(rows, phi, static_fz, output)
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()

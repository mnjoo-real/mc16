"""Independent validation and diagnostics for the AC conducting-sphere kernel."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_diffusion_sphere import (MU0, fit_reduced_response, response_l,
                                       static_scattered_l)

OUT = Path("out")
A = .006
SIGMA_REF = 6e6
MU_REF = 100.0
TAU_LAG = .004
STEP_OMEGA = np.array((18.85, 20.42, 26.70, 34.56, 53.41,
                       69.12, 72.26, 80.11, 103.67, 117.81))
SIGMAS = (1e6, 3e6, 6e6, 1e7)
MURS = (10.0, 50.0, 100.0, 500.0)


def response_row(row_type, r, reduced=np.nan):
    normalized = r.normalized_scattered
    center = r.normalized_center_field
    return dict(row_type=row_type, l=r.l, omega=r.omega, sigma=r.sigma,
                mu_r=r.mu_r, kappa_a=abs(r.kappa*r.radius),
                kappa_a_real=(r.kappa*r.radius).real,
                kappa_a_imag=(r.kappa*r.radius).imag,
                delta=r.delta, delta_over_a=r.delta/r.radius,
                Pi1=r.Pi1, response_real=normalized.real,
                response_imag=normalized.imag,
                response_magnitude=abs(normalized),
                response_phase_deg=np.angle(normalized, deg=True),
                scattered_real=r.scattered.real, scattered_imag=r.scattered.imag,
                internal_field_magnitude=(abs(center) if r.l == 1 else
                                          abs(r.normalized_internal_boundary)),
                internal_field_phase_deg=(np.angle(center, deg=True) if r.l == 1 else
                                          np.angle(r.normalized_internal_boundary, deg=True)),
                center_field_real=(center.real if r.l == 1 else np.nan),
                center_field_imag=(center.imag if r.l == 1 else np.nan),
                joule_loss=r.joule_loss_per_H0_sq,
                joule_loss_metric=r.joule_loss_metric,
                reduced_response_real=(np.real(reduced) if np.ndim(reduced) == 0 else np.nan),
                reduced_response_imag=(np.imag(reduced) if np.ndim(reduced) == 0 else np.nan))


def collect_data():
    rows = []
    # Mandatory low-frequency analytic validation.
    for mur in (1.0, 10.0, 50.0, 100.0, 500.0):
        for omega in (1e-8, 1e-5, 1e-2):
            r = response_l(1, omega, A, SIGMA_REF, mur)
            row = response_row("static_validation", r)
            if mur == 1:
                row["static_relative_error"] = np.nan
                row["absolute_moment_per_H0"] = abs(r.dipole_moment_per_H0)
            else:
                exact = 4*np.pi*A**3*(mur-1)/(mur+2)
                row["static_relative_error"] = abs(r.dipole_moment_per_H0-exact)/abs(exact)
            rows.append(row)

    # sigma -> 0 validation and higher-l static limits.
    for l in range(1, 7):
        for sigma in (0.0, 1e-12, 1e-6):
            r = response_l(l, 123.0, A, sigma, MU_REF)
            row = response_row("conductivity_limit", r)
            row["static_relative_error"] = abs(r.scattered-static_scattered_l(l, MU_REF))/abs(static_scattered_l(l, MU_REF))
            rows.append(row)

    # Reference frequency sweep through |kappa*a|=30.
    tau_diff = MU0*MU_REF*SIGMA_REF*A*A
    omega_max = 30.0**2/tau_diff
    omega_sweep = np.geomspace(1e-4, omega_max, 320)
    curves = {}
    reduced_models = {}
    for l in range(1, 7):
        reduced_models[l] = fit_reduced_response(l, A, SIGMA_REF, MU_REF,
                                                  omega_max=400.0, order=2)
        vals = []
        for omega in omega_sweep:
            r = response_l(l, omega, A, SIGMA_REF, MU_REF)
            reduced = reduced_models[l].evaluate(omega) if omega <= 400 else np.nan
            rows.append(response_row("frequency_sweep", r, reduced))
            vals.append(r)
        curves[l] = vals

    # Requested Step-5A frequency table and full material uncertainty sweep.
    step_reference = []
    for omega in STEP_OMEGA:
        rr = response_l(1, omega, A, SIGMA_REF, MU_REF)
        rows.append(response_row("step5a_reference", rr,
                                 reduced_models[1].evaluate(omega)))
        step_reference.append(rr)
        for sigma in SIGMAS:
            for mur in MURS:
                rows.append(response_row("step5a_uncertainty",
                                         response_l(1, omega, A, sigma, mur)))

    # Dense ROM-validation rows over the requested 0..400 rad/s interval.
    rom_omega = np.linspace(0.0, 400.0, 401)
    rom_data = {}
    for l, model in reduced_models.items():
        exact = np.array([response_l(l, w, A, SIGMA_REF, MU_REF).normalized_scattered
                          for w in rom_omega])
        approx = model.evaluate(rom_omega)
        rom_data[l] = (model, exact, approx)
        for w, ex, ap in zip(rom_omega, exact, approx):
            r = response_l(l, w, A, SIGMA_REF, MU_REF)
            row = response_row("reduced_model", r, ap)
            row["reduced_magnitude_error"] = abs(abs(ap)/abs(ex)-1)
            row["reduced_phase_error_deg"] = abs(np.angle(ap/ex, deg=True))
            rows.append(row)
    return rows, omega_sweep, curves, step_reference, rom_omega, rom_data


def write_csv(rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    OUT.mkdir(exist_ok=True)
    with (OUT/"sphere_diffusion_response.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def make_plot(omega, curves, step, rom_omega, rom_data):
    fig, ax = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    l1 = curves[1]
    mag = np.array([abs(r.normalized_scattered) for r in l1])
    phase = np.array([np.angle(r.normalized_scattered, deg=True) for r in l1])
    ax[0, 0].semilogx(omega, mag)
    ax[0, 0].scatter(STEP_OMEGA, [abs(r.normalized_scattered) for r in step], s=16)
    ax[0, 0].set(title="A. l=1 total dipole response", xlabel=r"$\Omega$ [rad/s]",
                 ylabel=r"$|m/m_{static}|$"); ax[0, 0].grid(True, which="both", alpha=.3)
    ax[0, 1].semilogx(omega, phase)
    ax[0, 1].scatter(STEP_OMEGA, [np.angle(r.normalized_scattered, deg=True) for r in step], s=16)
    ax[0, 1].set(title="B. l=1 dipole phase", xlabel=r"$\Omega$ [rad/s]",
                 ylabel="phase [deg]"); ax[0, 1].grid(True, which="both", alpha=.3)

    da = np.array([r.delta/A for r in l1])
    center = np.array([abs(r.normalized_center_field) for r in l1])
    order = np.argsort(da)
    ax[0, 2].semilogx(da[order], center[order], label="exact sphere center")
    ax[0, 2].semilogx(da[order], np.exp(-1/da[order]), "--", label=r"naive $e^{-a/\delta}$")
    ax[0, 2].set_xlim(30, .03); ax[0, 2].set(title="C. Center penetration",
                 xlabel=r"$\delta/a$", ylabel="amplitude relative to static")
    ax[0, 2].legend(); ax[0, 2].grid(True, which="both", alpha=.3)

    loss = np.array([r.joule_loss_per_H0_sq for r in l1])
    ax[0, 3].loglog(omega, loss)
    ax[0, 3].set(title="D. Passive Joule loss", xlabel=r"$\Omega$ [rad/s]",
                 ylabel=r"$P_J/|H_0|^2$ [W/(A/m)$^2$]")
    ax[0, 3].grid(True, which="both", alpha=.3)

    for l in range(1, 7):
        ax[1, 0].semilogx(omega, [abs(r.normalized_scattered) for r in curves[l]], label=f"l={l}")
        ax[1, 1].semilogx(omega, [np.angle(r.normalized_scattered, deg=True) for r in curves[l]], label=f"l={l}")
    ax[1, 0].set(title="E. Spatial-mode magnitudes", xlabel=r"$\Omega$ [rad/s]",
                 ylabel="normalized scattered response")
    ax[1, 0].legend(ncol=2); ax[1, 0].grid(True, which="both", alpha=.3)
    ax[1, 1].set(title="F. Spatial-mode phases", xlabel=r"$\Omega$ [rad/s]",
                 ylabel="phase [deg]"); ax[1, 1].grid(True, which="both", alpha=.3)

    exact = rom_data[1][1]
    glag = 1/(1+1j*rom_omega*TAU_LAG)
    ax[1, 2].plot(rom_omega, abs(exact), label="exact diffusion |G|")
    ax[1, 2].plot(rom_omega, abs(glag), "--", label=r"lag $|G|$")
    aphase = ax[1, 2].twinx()
    aphase.plot(rom_omega, np.angle(exact, deg=True), alpha=.7, label="diffusion phase")
    aphase.plot(rom_omega, np.angle(glag, deg=True), "--", alpha=.7, label="lag phase")
    ax[1, 2].set(title="G. Diffusion versus tau_lag", xlabel=r"$\Omega$ [rad/s]",
                 ylabel="magnitude"); aphase.set_ylabel("phase [deg]")
    lines = ax[1, 2].lines+aphase.lines
    ax[1, 2].legend(lines, [x.get_label() for x in lines], fontsize=8)
    ax[1, 2].grid(alpha=.3)

    for l in range(1, 7):
        _, ex, ap = rom_data[l]
        ax[1, 3].semilogy(rom_omega, 100*np.abs(np.abs(ap)/np.abs(ex)-1), label=f"l={l} mag %")
    ax[1, 3].axhline(1, color="k", ls="--", label="1% target")
    ax[1, 3].set(title="H. Two-pole ROM magnitude error", xlabel=r"$\Omega$ [rad/s]",
                 ylabel="magnitude error [%]")
    ax[1, 3].legend(ncol=2, fontsize=7); ax[1, 3].grid(True, which="both", alpha=.3)
    fig.suptitle("Conducting permeable sphere: exact magnetic-diffusion modes", fontsize=15)
    fig.savefig(OUT/"sphere_diffusion_response.png", dpi=180)


def main():
    rows, omega, curves, step, rom_omega, rom_data = collect_data()
    write_csv(rows); make_plot(omega, curves, step, rom_omega, rom_data)
    print("Step-5A l=1 reference responses")
    for r in step:
        n, c = r.normalized_scattered, r.normalized_center_field
        print(f"Omega={r.omega:7.2f} delta/a={r.delta/A:6.3f} "
              f"|m/m0|={abs(n):.7f} phase_m={np.angle(n,deg=True):+7.3f}deg "
              f"|Bc/Bc0|={abs(c):.6f} phase_c={np.angle(c,deg=True):+7.3f}deg "
              f"P/H0^2={r.joule_loss_per_H0_sq:.3e}")
    print("Two-pole reduced models over 0..400 rad/s")
    for l, (model, _, _) in rom_data.items():
        print(f"l={l}: mag={100*model.max_magnitude_error:.5f}% "
              f"phase={model.max_phase_error_deg:.5f}deg poles={model.poles}")
    print(f"wrote {OUT/'sphere_diffusion_response.csv'}")
    print(f"wrote {OUT/'sphere_diffusion_response.png'}")


if __name__ == "__main__":
    main()

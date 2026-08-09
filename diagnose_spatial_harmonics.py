"""Diagnose angular periodicity and harmonics of the existing FieldGrid model.

This script is read-only with respect to the physical model.  It samples the
field already produced by :class:`magnetic_carousel.FieldGrid`, expresses the
vector in the local cylindrical basis, and reports spatial symmetry and FFT
content around the nominal ball orbit.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_carousel import FieldGrid, Params


N_PHI = 3072  # >= 2048 and divisible by the default n_mag=12
GAPS = (0.003, 0.005, 0.010, 0.016)
QUANTITY_NAMES = ("B_r", "B_phi", "B_z", "|B|", "|B|^2")


def rms(a: np.ndarray) -> float:
    """Root mean square over all supplied samples/components."""
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def cylindrical_samples(grid: FieldGrid, radius: float, n_phi: int = N_PHI):
    """Return phi, cylindrical B, |B|, and |B|^2 from FieldGrid samples."""
    phi, raw = grid.sample_circle(radius, n_phi)
    bx, by, bz = raw[:, 0], raw[:, 1], raw[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    br = bx * c + by * s
    bphi = -bx * s + by * c
    bcyl = np.column_stack((br, bphi, bz))
    bmag = np.linalg.norm(bcyl, axis=1)
    return phi, bcyl, bmag, bmag**2


def shifted(a: np.ndarray, samples: int) -> np.ndarray:
    """Return a(phi + shift), assuming an integer periodic sample shift."""
    return np.roll(a, -samples, axis=0)


def symmetry_residuals(
    bcyl: np.ndarray, bmag: np.ndarray, b2: np.ndarray, n_mag: int
) -> dict[str, float]:
    if len(bcyl) % n_mag:
        raise ValueError("n_phi must be divisible by n_mag for exact pitch shifts")
    one = len(bcyl) // n_mag
    two = 2 * one
    bden = rms(bcyl)
    return {
        "B_anti_1": rms(shifted(bcyl, one) + bcyl) / bden,
        "B_repeat_1": rms(shifted(bcyl, one) - bcyl) / bden,
        "B_repeat_2": rms(shifted(bcyl, two) - bcyl) / bden,
        "|B|_repeat_1": rms(shifted(bmag, one) - bmag) / rms(bmag),
        "|B|^2_repeat_1": rms(shifted(b2, one) - b2) / rms(b2),
    }


def fft_amplitudes(values: np.ndarray) -> np.ndarray:
    """One-sided Fourier amplitudes (cosine amplitude convention)."""
    coeff = np.fft.rfft(values) / len(values)
    amp = 2.0 * np.abs(coeff)
    amp[0] = np.abs(coeff[0])
    if len(values) % 2 == 0:
        amp[-1] = np.abs(coeff[-1])
    return amp


def top_modes(values: np.ndarray, count: int = 10):
    """Largest non-DC modes and amplitudes normalized to the largest mode."""
    amp = fft_amplitudes(values)
    order = np.argsort(amp[1:])[::-1][:count] + 1
    peak = amp[order[0]] if len(order) else 1.0
    return [(int(n), float(amp[n] / peak), float(amp[n])) for n in order]


def print_mode_table(quantities: dict[str, np.ndarray]) -> None:
    print("  Fourier modes (non-DC; normalized to each quantity's largest mode)")
    print("    quantity       n:normalized_amplitude [absolute amplitude]")
    for name, values in quantities.items():
        entries = "  ".join(
            f"{n}:{norm:.6g} [{absolute:.6e}]"
            for n, norm, absolute in top_modes(values)
        )
        print(f"    {name:<8} {entries}")


def make_figure(
    phi: np.ndarray,
    quantities: dict[str, np.ndarray],
    p: Params,
    output: str,
) -> None:
    deg = np.degrees(phi)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, name in zip(axes.flat[:5], QUANTITY_NAMES):
        scale = 1e3 if name in ("B_r", "B_phi", "B_z", "|B|") else 1e6
        unit = "mT" if scale == 1e3 else r"mT$^2$"
        ax.plot(deg, quantities[name] * scale, lw=0.9)
        ax.set(xlabel=r"$\phi$ [deg]", ylabel=f"{name} [{unit}]", xlim=(0, 360))
        ax.grid(alpha=0.25)

    ax = axes.flat[5]
    for name in QUANTITY_NAMES:
        amp = fft_amplitudes(quantities[name])
        peak = np.max(amp[1:])
        ax.semilogy(np.arange(1, 61), amp[1:61] / peak, marker="o", ms=2.5,
                    lw=0.8, label=name)
    ax.set(xlabel="angular Fourier mode n",
           ylabel="amplitude / largest non-DC amplitude", xlim=(1, 60),
           ylim=(1e-7, 2))
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"Existing FieldGrid spatial harmonics: gap={p.gap*1e3:g} mm, "
        f"r={p.r_mag*1e3:g} mm, N={p.n_mag}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    default = Params()
    if N_PHI % default.n_mag:
        raise ValueError("N_PHI must be divisible by Params().n_mag")

    print("Spatial-harmonic diagnostic (existing FieldGrid; model unchanged)")
    print(f"n_mag = {default.n_mag}")
    print(f"nominal radius = {default.r_mag:.9g} m")
    print(f"magnet/pole pitch = 2*pi*R/N = {default.wavelength:.9g} m")
    print(f"two-pole repeat length = 4*pi*R/N = {2*default.wavelength:.9g} m")
    print(f"angular samples = {N_PHI}")
    print("mode n maps to temporal angular frequency Omega_n=n*omega_disc")

    default_data = None
    for gap in GAPS:
        p = Params(gap=gap)
        grid = FieldGrid(p, cache_dir=None, verbose=True)
        phi, bcyl, bmag, b2 = cylindrical_samples(grid, p.r_mag)
        residuals = symmetry_residuals(bcyl, bmag, b2, p.n_mag)
        quantities = {
            "B_r": bcyl[:, 0],
            "B_phi": bcyl[:, 1],
            "B_z": bcyl[:, 2],
            "|B|": bmag,
            "|B|^2": b2,
        }

        print(f"\ngap = {gap*1e3:g} mm (ball-centre height={p.z_ball*1e3:g} mm)")
        print(
            "  periodicity residuals: "
            + "  ".join(f"{key}={value:.6e}" for key, value in residuals.items())
        )
        print_mode_table(quantities)
        if np.isclose(gap, default.gap):
            default_data = (phi, quantities, p)

    # The requested gaps do not include the Params default (8 mm), so sample it
    # separately for the requested default-gap figure and compact summary.
    if default_data is None:
        grid = FieldGrid(default, cache_dir=None, verbose=True)
        phi, bcyl, bmag, b2 = cylindrical_samples(grid, default.r_mag)
        default_quantities = {
            "B_r": bcyl[:, 0], "B_phi": bcyl[:, 1], "B_z": bcyl[:, 2],
            "|B|": bmag, "|B|^2": b2,
        }
        default_data = (phi, default_quantities, default)
        residuals = symmetry_residuals(bcyl, bmag, b2, default.n_mag)
        print(f"\ndefault gap = {default.gap*1e3:g} mm (figure/summary sample)")
        print(
            "  periodicity residuals: "
            + "  ".join(f"{key}={value:.6e}" for key, value in residuals.items())
        )
        print_mode_table(default_quantities)

    phi, quantities, p = default_data
    bcyl = np.column_stack((quantities["B_r"], quantities["B_phi"], quantities["B_z"]))
    residuals = symmetry_residuals(bcyl, quantities["|B|"], quantities["|B|^2"], p.n_mag)
    dominant_b = {name: top_modes(quantities[name], 3) for name in QUANTITY_NAMES[:3]}
    dominant_b2 = top_modes(quantities["|B|^2"], 3)
    print("\nCompact default-geometry summary")
    print(f"  n_mag = {p.n_mag}")
    print(f"  pole pitch = {p.wavelength:.9g} m = {p.wavelength*1e3:.6g} mm")
    print("  dominant B harmonics = " + ", ".join(
        f"{name}:" + "/".join(str(row[0]) for row in modes)
        for name, modes in dominant_b.items()))
    print("  dominant |B|^2 harmonics = " + "/".join(str(row[0]) for row in dominant_b2))
    print(f"  B anti-periodicity over one pitch = {residuals['B_anti_1']:.6e}")
    print(f"  B periodicity over one pitch = {residuals['B_repeat_1']:.6e}")
    print(f"  B periodicity over two pitches = {residuals['B_repeat_2']:.6e}")
    print(f"  |B| periodicity over one pitch = {residuals['|B|_repeat_1']:.6e}")
    print(f"  |B|^2 periodicity over one pitch = {residuals['|B|^2_repeat_1']:.6e}")

    output = "out/spatial_harmonics.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    make_figure(phi, quantities, p, output)
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()

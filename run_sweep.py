"""Parameter studies.

  python run_sweep.py phase    -> fig3 (Omega/omega vs omega) + fig4 (phase map)
  python run_sweep.py params   -> fig5 (ball size, magnet number, friction, lag)
  python run_sweep.py hyst     -> fig6 (up/down sweep, step-out hysteresis)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from magnetic_carousel import Params, FieldGrid, simulate

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 130, "axes.axisbelow": True})

T_END, DT, STRIDE = 1.3, 2.5e-5, 60
CACHE: dict[str, FieldGrid] = {}


def grid_for(p):
    k = p.field_key()
    if k not in CACHE:
        CACHE[k] = FieldGrid(p, verbose=False)
    return CACHE[k]


def ratio_of(p, **sim_kw):
    r = simulate(p, t_end=T_END, dt=DT, stride=STRIDE, grid=grid_for(p), **sim_kw)
    return r.summary(), r


# ==========================================================================
def do_phase():
    gaps = np.array([3, 4, 5, 6, 8, 10, 13, 16, 20]) * 1e-3
    omegas = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 90])
    R = np.zeros((len(gaps), len(omegas)))
    for i, gp in enumerate(gaps):
        for j, om in enumerate(omegas):
            p = Params(gap=gp, omega=om)
            s, _ = ratio_of(p)
            R[i, j] = s["ratio"]
        print(f"  gap {gp*1e3:4.1f} mm  " +
              " ".join(f"{v:+5.2f}" for v in R[i]), flush=True)
    np.savez("out/phase.npz", gaps=gaps, omegas=omegas, R=R)

    # --- fig3: transition curves ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, gp in enumerate(gaps):
        ax.semilogx(omegas, R[i], "o-", ms=4, lw=1.3,
                    color=cmap(i / (len(gaps) - 1)), label=f"{gp*1e3:.0f} mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="0.5", lw=0.8, ls=":")
    ax.text(0.55, 1.04, "synchronous lock", fontsize=7.5, color="0.4")
    ax.set_xlabel(r"disc angular velocity $\omega$ [rad/s]")
    ax.set_ylabel(r"$\Omega_{ball}/\omega$")
    ax.set_title("Step-out transition: prograde lock $\\rightarrow$ retrograde rolling")
    ax.legend(title="gap", fontsize=7.5, title_fontsize=8, ncol=2)
    ax.set_ylim(-1.1, 1.35)
    fig.tight_layout()
    fig.savefig("out/fig3_transition.png", bbox_inches="tight")
    print("saved out/fig3_transition.png")

    # --- fig4: phase map ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    Rc = np.clip(R, -1.1, 1.1)
    pc = ax.pcolormesh(omegas, gaps * 1e3, Rc, cmap="RdBu_r", shading="nearest",
                       vmin=-1.1, vmax=1.1)
    ax.contour(omegas, gaps * 1e3, R, levels=[0.0], colors="k", linewidths=1.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("gap [mm]")
    ax.set_title("Phase map  (red = prograde, blue = retrograde)")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(r"$\Omega_{ball}/\omega$")
    fig.tight_layout()
    fig.savefig("out/fig4_phasemap.png", bbox_inches="tight")
    print("saved out/fig4_phasemap.png")


# ==========================================================================
def do_params():
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.0))
    omegas = np.array([1, 2, 4, 8, 16, 32, 64])

    # (a) ball radius -> dimensionless ka
    ax = axs[0, 0]
    for a, c in zip((0.003, 0.004, 0.006, 0.008, 0.010),
                    plt.get_cmap("plasma")(np.linspace(0.1, 0.85, 5))):
        y = []
        for om in omegas:
            p = Params(ball_R=a, omega=om, gap=0.005)
            y.append(ratio_of(p)[0]["ratio"])
        ax.semilogx(omegas, y, "o-", ms=4, color=c,
                    label=f"a={a*1e3:.0f} mm  (ka={Params(ball_R=a).ka:.2f})")
        print(f"  ball_R={a*1e3:.0f}mm done", flush=True)
    ax.set_title("ball radius")

    # (b) number of magnets
    ax = axs[0, 1]
    for n, c in zip((6, 8, 12, 18, 24),
                    plt.get_cmap("plasma")(np.linspace(0.1, 0.85, 5))):
        y = []
        for om in omegas:
            p = Params(n_mag=n, omega=om, gap=0.005)
            y.append(ratio_of(p)[0]["ratio"])
        ax.semilogx(omegas, y, "o-", ms=4, color=c,
                    label=f"N={n}  ($\\lambda$={Params(n_mag=n).wavelength*1e3:.0f} mm)")
        print(f"  N={n} done", flush=True)
    ax.set_title("number of magnets")

    # (c) sliding friction
    ax = axs[1, 0]
    for mu, c in zip((0.02, 0.05, 0.1, 0.2, 0.4),
                     plt.get_cmap("plasma")(np.linspace(0.1, 0.85, 5))):
        y = []
        for om in omegas:
            p = Params(mu_k=mu, omega=om, gap=0.005)
            y.append(ratio_of(p)[0]["ratio"])
        ax.semilogx(omegas, y, "o-", ms=4, color=c, label=f"$\\mu_k$={mu}")
    ax.set_title("plate friction")
    print("  friction done", flush=True)

    # (d) magnetisation lag time
    ax = axs[1, 1]
    for tl, c in zip((0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3),
                     plt.get_cmap("plasma")(np.linspace(0.1, 0.85, 5))):
        y = []
        for om in omegas:
            p = Params(tau_lag=tl, omega=om, gap=0.005)
            y.append(ratio_of(p)[0]["ratio"])
        ax.semilogx(omegas, y, "o-", ms=4, color=c,
                    label=f"$\\tau_{{lag}}$={tl*1e3:.1f} ms")
    ax.set_title("magnetisation lag (eddy / hysteresis)")
    print("  tau_lag done", flush=True)

    for ax in axs.ravel():
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.set_ylabel(r"$\Omega_{ball}/\omega$")
        ax.set_ylim(-1.2, 1.35)
        ax.legend(fontsize=7)
    fig.suptitle("What controls the direction (gap fixed at 5 mm)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("out/fig5_parameters.png", bbox_inches="tight")
    print("saved out/fig5_parameters.png")


# ==========================================================================
def do_hyst():
    """Slow up/down ramp of omega, continuing from the previous state:
    reveals step-out / pull-in hysteresis."""
    gap = 0.005
    omegas = np.concatenate([np.arange(1, 21, 1.0), np.arange(20, 0, -1.0)])
    T_RAMP = 2.2          # long enough for the transient to die at each step
    up, down = [], []
    p0 = Params(gap=gap)
    g = grid_for(p0)
    state = None
    res = []
    for om in omegas:
        p = replace(p0, omega=om)
        kw = dict(y_init=state) if state is not None else {}
        r = simulate(p, t_end=T_RAMP, dt=DT, stride=STRIDE, grid=g, **kw)
        # measure only over the last 40 %, after the transient
        s = dict(ratio=r.orbital_rate(frac=0.4) / om)
        res.append(s["ratio"])
        state = r.state[-1].copy()
        print(f"  w={om:5.1f} ratio={s['ratio']:+6.3f}", flush=True)
    n = len(omegas) // 2
    up, down = res[:n], res[n:]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(omegas[:n], up, "o-", ms=4, color="#D85A30", label=r"$\omega$ increasing")
    ax.plot(omegas[n:], down, "s--", ms=4, color="#378ADD", label=r"$\omega$ decreasing")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$\Omega_{ball}/\omega$")
    ax.set_title("Step-out / pull-in hysteresis (gap = 5 mm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("out/fig6_hysteresis.png", bbox_inches="tight")
    print("saved out/fig6_hysteresis.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "phase"
    {"phase": do_phase, "params": do_params, "hyst": do_hyst}[what]()

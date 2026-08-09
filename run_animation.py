"""Animations of the magnetic carousel.

anim1_mechanism.mp4 : four panels -- top view with the rotating |B| map and the
                      ball, the local B / m vectors that generate the torque,
                      the travelling trap potential, and the running angles.
anim2_compare.mp4   : prograde and retrograde runs side by side.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, FancyArrowPatch

from magnetic_carousel import Params, FieldGrid, simulate, _bilinear

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.2,
                     "figure.dpi": 110, "axes.axisbelow": True})

T_END, DT, NFRAME, FPS = 1.6, 2e-5, 300, 30
STRIDE = max(1, int(T_END / DT) // NFRAME)


def run(p):
    g = FieldGrid(p, verbose=True)
    r = simulate(p, t_end=T_END, dt=DT, stride=STRIDE, grid=g)
    return g, r


def draw_static(ax, p, g, view):
    """|B| map (co-rotating frame) + magnets; returns the artists to rotate."""
    L = g.xs[-1] * 1e3
    B = g.Bmag() * 1e3
    X, Y = np.meshgrid(g.xs, g.ys)
    B = np.where(np.hypot(X, Y) < g.xs[-1], B, np.nan)
    im = ax.imshow(B, extent=[-L, L, -L, L], origin="lower", cmap="magma",
                   vmin=0, vmax=np.nanpercentile(B, 99.5), zorder=0)
    mk = []
    for j in range(p.n_mag):
        a0 = 2 * np.pi * j / p.n_mag
        c = "#F7C1C1" if j % 2 == 0 else "#B5D4F4"
        (ln,) = ax.plot([p.r_mag * np.cos(a0) * 1e3], [p.r_mag * np.sin(a0) * 1e3],
                        marker="s", ms=4.5, mfc="none", mec=c, mew=1.1, zorder=3)
        mk.append(ln)
    ax.add_patch(Circle((0, 0), p.r_rim * 1e3, fill=False, ec="0.5", lw=1.0,
                        ls="--", zorder=2))
    ax.set_xlim(-view * 1e3, view * 1e3)
    ax.set_ylim(-view * 1e3, view * 1e3)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_aspect("equal")
    ax.set_facecolor("#000004")
    ax.grid(False)
    return im, mk


def make_mechanism_anim(p, fname):
    g, r = run(p)
    s = r.summary()
    print(f"  ratio = {s['ratio']:+.3f}")

    # trap potential along the mean orbit, in the disc frame
    r0 = float(np.median(r.r))
    phi_g, U, Bm = g.potential_on_circle(r0, 720)
    U = (U - U.mean()) * 1e3           # mJ

    fig = plt.figure(figsize=(12.4, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 1, 1], hspace=0.40, wspace=0.30)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 1:])

    view = p.r_rim * 1.06
    im, mk = draw_static(axA, p, g, view)
    axA.set_title("top view: $|B|$ at the ball plane (mT)", fontsize=9)
    trail, = axA.plot([], [], lw=1.4, color="#5DCAA5", alpha=0.9, zorder=4)
    ballpt = axA.add_patch(Circle((0, 0), p.ball_R * 1e3, fc="#E1F5EE", ec="k",
                                  lw=0.8, zorder=6))
    varrow = FancyArrowPatch((0, 0), (0, 0), color="#5DCAA5", lw=1.6,
                             arrowstyle="-|>", mutation_scale=11, zorder=7)
    axA.add_patch(varrow)
    txtA = axA.text(0.02, 0.975, "", transform=axA.transAxes, va="top",
                    fontsize=8.5, color="w", family="monospace")
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7)

    # ---- panel B: local B and m in the (tangential, vertical) plane -------
    axB.set_xlim(-1.35, 1.35); axB.set_ylim(-1.35, 1.35)
    axB.set_aspect("equal")
    axB.axhline(0, color="0.7", lw=0.6); axB.axvline(0, color="0.7", lw=0.6)
    axB.set_xlabel("tangential  (prograde $\\rightarrow$)", fontsize=8)
    axB.set_ylabel("vertical", fontsize=8)
    axB.set_title("field vector rotates, $m$ lags behind", fontsize=9, pad=18)
    aB = FancyArrowPatch((0, 0), (0, 0), color="#378ADD", lw=2.2,
                         arrowstyle="-|>", mutation_scale=13)
    aM = FancyArrowPatch((0, 0), (0, 0), color="#D85A30", lw=2.2,
                         arrowstyle="-|>", mutation_scale=13)
    axB.add_patch(aB); axB.add_patch(aM)
    axB.text(0.03, 0.96, "B", color="#185FA5", transform=axB.transAxes,
             va="top", fontsize=10)
    axB.text(0.03, 0.85, "m", color="#993C1D", transform=axB.transAxes,
             va="top", fontsize=10)
    txtB = axB.text(0.5, 1.02, "", transform=axB.transAxes, ha="center",
                    va="bottom", fontsize=8)

    # ---- panel C: travelling trap potential ------------------------------
    axC.set_title("trap potential along the orbit", fontsize=9)
    axC.set_xlabel("lab angle [deg]", fontsize=8)
    axC.set_ylabel("$U$ [mJ]", fontsize=8)
    lU, = axC.plot([], [], color="#534AB7", lw=1.3)
    pU, = axC.plot([], [], "o", color="#D85A30", ms=7, mec="k", mew=0.6)
    axC.set_xlim(0, 360)
    pad = 0.12 * (U.max() - U.min() + 1e-9)
    axC.set_ylim(U.min() - pad, U.max() + pad)

    # ---- panel D: angles --------------------------------------------------
    axD.set_xlabel("t [s]", fontsize=8)
    axD.set_ylabel("rotation [turns]", fontsize=8)
    axD.set_xlim(0, r.t[-1])
    disc_turns = p.omega * r.t / (2 * np.pi)
    ball_turns = (r.phi - r.phi[0]) / (2 * np.pi)
    lo = min(ball_turns.min(), 0) - 0.2
    hi = max(disc_turns.max(), ball_turns.max()) + 0.2
    axD.set_ylim(lo, hi)
    axD.axhline(0, color="k", lw=0.6)
    lD1, = axD.plot([], [], color="0.45", lw=1.4, label="disc")
    lD2, = axD.plot([], [], color="#D85A30", lw=1.7, label="ball")
    axD.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Magnetic carousel   $\\omega$={p.omega:g} rad/s, "
                 f"gap={p.gap*1e3:g} mm, N={p.n_mag}, a={p.ball_R*1e3:g} mm   "
                 f"$\\rightarrow$  $\\Omega_{{ball}}/\\omega$ = {s['ratio']:+.2f}",
                 fontsize=11)

    nf = len(r.t)
    trail_len = 90

    def update(i):
        t = r.t[i]
        ang = p.omega * t
        im.set_transform(Affine2D().rotate(ang) + axA.transData)
        for j, ln in enumerate(mk):
            a0 = 2 * np.pi * j / p.n_mag + ang
            ln.set_data([p.r_mag * np.cos(a0) * 1e3],
                        [p.r_mag * np.sin(a0) * 1e3])
        i0 = max(0, i - trail_len)
        trail.set_data(r.x[i0:i + 1] * 1e3, r.y[i0:i + 1] * 1e3)
        ballpt.center = (r.x[i] * 1e3, r.y[i] * 1e3)
        vsc = 100.0
        varrow.set_positions((r.x[i] * 1e3, r.y[i] * 1e3),
                             (r.x[i] * 1e3 + r.vx[i] * vsc,
                              r.y[i] * 1e3 + r.vy[i] * vsc))
        txtA.set_text(f"t = {t:5.2f} s\nv = {np.hypot(r.vx[i], r.vy[i])*1e3:6.1f} mm/s")

        # local frame
        rr = np.hypot(r.x[i], r.y[i]) + 1e-12
        ex, ey = r.x[i] / rr, r.y[i] / rr           # radial
        tx, ty = -ey, ex                            # tangential (prograde)
        Bv = np.array([0.0, 0.0, 0.0])
        # recompute B and m projections
        from magnetic_carousel import _field_lab
        Bl, Jl = _field_lab(g.F, g.x0, g.y0, g.dx, g.dy, r.x[i], r.y[i], ang)
        Bt, Bz = Bl[0] * tx + Bl[1] * ty, Bl[2]
        mt, mz = r.m[i, 0] * tx + r.m[i, 1] * ty, r.m[i, 2]
        nb = np.hypot(Bt, Bz) + 1e-18
        nm = np.hypot(mt, mz) + 1e-18
        aB.set_positions((0, 0), (Bt / nb, Bz / nb))
        aM.set_positions((0, 0), (mt / nm * 0.82, mz / nm * 0.82))
        cross = (mt * Bz - mz * Bt)
        txtB.set_text(r"$\tau_{radial}$ = " + f"{cross*1e3:+.2f} mN·m"
                      + ("  (rolls it backwards)" if cross > 0
                         else "  (rolls it forwards)"))

        # potential in the lab frame: U_lab(phi) = U_disc(phi - omega t)
        shift = int(round(ang / (2 * np.pi) * len(U))) % len(U)
        Ul = np.roll(U, shift)
        deg = np.degrees(phi_g)
        lU.set_data(deg, Ul)
        pb = np.degrees(np.arctan2(r.y[i], r.x[i])) % 360
        j = int(pb / 360 * len(U)) % len(U)
        pU.set_data([pb], [Ul[j]])

        lD1.set_data(r.t[:i + 1], disc_turns[:i + 1])
        lD2.set_data(r.t[:i + 1], ball_turns[:i + 1])
        return ()

    anim = FuncAnimation(fig, update, frames=range(0, nf, max(1, nf // NFRAME)),
                         blit=False)
    anim.save(fname, writer=FFMpegWriter(fps=FPS, bitrate=2600))
    plt.close(fig)
    print(f"  saved {fname}")


def make_compare_anim(cases, fname):
    data = []
    for p in cases:
        g, r = run(p)
        data.append((p, g, r))
        print(f"  gap={p.gap*1e3:g} w={p.omega:g}: ratio={r.summary()['ratio']:+.3f}")

    fig, axs = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 5.9))
    arts = []
    for ax, (p, g, r) in zip(axs, data):
        view = p.r_rim * 1.06
        im, mk = draw_static(ax, p, g, view)
        trail, = ax.plot([], [], lw=1.5, color="#5DCAA5", zorder=4)
        ball = ax.add_patch(Circle((0, 0), p.ball_R * 1e3, fc="#E1F5EE", ec="k",
                                   lw=0.8, zorder=6))
        s = r.summary()
        lbl = "PROGRADE" if s["ratio"] > 0 else "RETROGRADE"
        ax.set_title(f"{lbl}\n$\\omega$={p.omega:g} rad/s, gap={p.gap*1e3:g} mm"
                     f"   $\\Omega/\\omega$={s['ratio']:+.2f}", fontsize=10)
        txt = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top",
                      color="w", fontsize=8.5, family="monospace")
        arts.append((p, g, r, im, mk, trail, ball, txt, ax))

    nf = min(len(d[2].t) for d in data)

    def update(i):
        for (p, g, r, im, mk, trail, ball, txt, ax) in arts:
            ang = p.omega * r.t[i]
            im.set_transform(Affine2D().rotate(ang) + ax.transData)
            for j, ln in enumerate(mk):
                a0 = 2 * np.pi * j / p.n_mag + ang
                ln.set_data([p.r_mag * np.cos(a0) * 1e3],
                            [p.r_mag * np.sin(a0) * 1e3])
            i0 = max(0, i - 90)
            trail.set_data(r.x[i0:i + 1] * 1e3, r.y[i0:i + 1] * 1e3)
            ball.center = (r.x[i] * 1e3, r.y[i] * 1e3)
            turns = (r.phi[i] - r.phi[0]) / (2 * np.pi)
            txt.set_text(f"t     = {r.t[i]:5.2f} s\n"
                         f"disc  = {p.omega*r.t[i]/(2*np.pi):+6.2f} turns\n"
                         f"ball  = {turns:+6.2f} turns")
        return ()

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=range(0, nf, max(1, nf // NFRAME)),
                         blit=False)
    anim.save(fname, writer=FFMpegWriter(fps=FPS, bitrate=2600))
    plt.close(fig)
    print(f"  saved {fname}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        print("mechanism animation (retrograde case)")
        make_mechanism_anim(Params(gap=0.005, omega=16.0),
                            "out/anim1_mechanism.mp4")
    if which in ("all", "1b"):
        print("mechanism animation (prograde case)")
        make_mechanism_anim(Params(gap=0.005, omega=3.0),
                            "out/anim1b_mechanism_prograde.mp4")
    if which in ("all", "2"):
        print("side-by-side comparison")
        make_compare_anim([Params(gap=0.005, omega=3.0),
                           Params(gap=0.005, omega=16.0)],
                          "out/anim2_compare.mp4")

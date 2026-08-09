"""Detailed diagnostics for two representative runs:
a locked (prograde) case and a rolling (retrograde) case."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from magnetic_carousel import Params, FieldGrid, simulate

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 130, "axes.axisbelow": True})

CASES = [("Prograde / synchronous lock", dict(gap=0.005, omega=3.0), "#1D9E75"),
         ("Retrograde / torque-driven rolling", dict(gap=0.005, omega=16.0), "#D85A30")]

fig, axes = plt.subplots(3, 2, figsize=(11, 9.5))

for col, (title, kw, color) in enumerate(CASES):
    p = Params(**kw)
    g = FieldGrid(p, verbose=True)
    r = simulate(p, t_end=1.6, dt=2e-5, stride=40, grid=g)
    s = r.summary()
    print(f"{title}: ratio={s['ratio']:.3f}  Omega_ball={s['Omega_ball']:.3f} rad/s")

    # ---------------- trajectory -----------------------------------------
    ax = axes[0, col]
    ang = np.linspace(0, 2 * np.pi, 400)
    ax.plot(p.r_mag * np.cos(ang) * 1e3, p.r_mag * np.sin(ang) * 1e3,
            ls="--", lw=0.8, color="0.6")
    for j in range(p.n_mag):
        a0 = 2 * np.pi * j / p.n_mag
        ax.plot(p.r_mag * np.cos(a0) * 1e3, p.r_mag * np.sin(a0) * 1e3,
                marker="s", ms=5,
                color="#D85A30" if j % 2 == 0 else "#378ADD")
    ax.plot(r.x * 1e3, r.y * 1e3, lw=1.2, color=color)
    ax.plot(r.x[0] * 1e3, r.y[0] * 1e3, "ko", ms=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"{title}\n$\\omega$={p.omega:.0f} rad/s, gap={p.gap*1e3:.0f} mm, "
                 f"$\\Omega_{{ball}}/\\omega$={s['ratio']:+.2f}", fontsize=9)

    # ---------------- angles ---------------------------------------------
    ax = axes[1, col]
    ax.plot(r.t, np.degrees(p.omega * r.t), color="0.4", lw=1.2, label="disc")
    ax.plot(r.t, np.degrees(r.phi - r.phi[0]), color=color, lw=1.4, label="ball")
    ax.set_xlabel("t [s]"); ax.set_ylabel("rotation angle [deg]")
    ax.legend(fontsize=8)

    # ---------------- rolling check --------------------------------------
    ax = axes[2, col]
    rr = np.maximum(np.hypot(r.x, r.y), 1e-9)
    w_rad = (r.w[:, 0] * r.x + r.w[:, 1] * r.y) / rr
    ax.plot(r.t, r.v_tan * 1e3, color=color, lw=1.3,
            label=r"$v_{tan}$  (+ = prograde)")
    ax.plot(r.t, -p.ball_R * w_rad * 1e3, "k--", lw=1.0,
            label=r"$-a\,\omega_{radial}$  (pure rolling)")
    ax.plot(r.t, r.slip * 1e3, color="0.6", lw=0.9, label="contact slip")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("velocity [mm/s]")
    ax.legend(fontsize=7.5, loc="best")

fig.suptitle("Magnetic carousel — the two dynamical regimes", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("out/fig1_regimes.png", bbox_inches="tight")
print("saved out/fig1_regimes.png")

# =========================================================================
# force / torque balance figure
# =========================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(13.2, 3.9))
p0 = Params()
gaps = np.linspace(0.002, 0.024, 23)
Ftrap, Btyp = [], []
for gp in gaps:
    pp = Params(gap=gp)
    gg = FieldGrid(pp, verbose=False)
    phi, U, Bm = gg.potential_on_circle(pp.r_mag, 1440)
    ds = pp.r_mag * (phi[1] - phi[0])
    Ftrap.append(np.abs(np.gradient(U, ds)).max())
    Btyp.append(Bm.mean())
Ftrap = np.array(Ftrap)
Btyp = np.array(Btyp)
Tscale = p0.alpha * Btyp ** 2 / p0.ball_R      # tau/a  with the lag factor set to 1
z = gaps + p0.ball_R                            # height of the ball centre
k = p0.k_wave

def slope(y):
    return -np.polyfit(z, np.log(y), 1)[0] / k   # effective decay exponent in kz

ax = axes2[0]
ax.semilogy(gaps * 1e3, Ftrap, "o-", color="#1D9E75",
            label=r"trap force $\max|dU/ds|$   ($e^{-%.1f kz}$)" % slope(Ftrap))
ax.semilogy(gaps * 1e3, Tscale, "s-", color="#D85A30",
            label=r"torque scale $\alpha B^2/a$   ($e^{-%.1f kz}$)" % slope(Tscale))
ax.set_xlabel("gap [mm]")
ax.set_ylabel("force scale [N]")
ax.set_title("Both decay, but the trap decays faster")
ax.legend(fontsize=7.5)

ax = axes2[1]
ax.semilogy(gaps * 1e3, Ftrap / Tscale, "o-", color="#534AB7")
ax.axhline(1.0, color="k", lw=0.9, ls="--")
ax.set_xlabel("gap [mm]")
ax.set_ylabel(r"$F_{trap}\,a\,/\,\tau_{max}$")
ax.set_title("Direction criterion: prograde survives while\n"
             r"the lag factor $g(\Omega\tau_{lag})$ stays below this curve",
             fontsize=9)

ax = axes2[2]
for gp, c in zip((0.003, 0.006, 0.012, 0.020),
                 ("#26215C", "#534AB7", "#7F77DD", "#AFA9EC")):
    pp = Params(gap=gp)
    gg = FieldGrid(pp, verbose=False)
    phi, U, Bm = gg.potential_on_circle(pp.r_mag, 1440)
    m = slice(0, 180)
    ax.plot(np.degrees(phi[m]), (U[m] - U[m].mean()) * 1e3, color=c,
            label=f"gap {gp*1e3:.0f} mm")
ax.set_xlabel(r"$\varphi$ [deg]  (disc frame)")
ax.set_ylabel(r"$U-\langle U\rangle$  [mJ]")
ax.set_title(r"Trap potential $U=-\frac{1}{2}\alpha|B|^2$")
ax.legend(fontsize=7.5)
fig2.tight_layout()
fig2.savefig("out/fig2_scaling.png", bbox_inches="tight")
print("saved out/fig2_scaling.png")

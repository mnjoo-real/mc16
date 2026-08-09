"""Sanity checks for the analytic cuboid field."""
import numpy as np
from magnetic_carousel import (Params, FieldGrid, magnet_potential, MU0,
                               _rect_potential)

# ---- 1. analytic rectangle potential vs brute-force double integral --------
hx, hy, w = 0.005, 0.004, 0.009
x, y = 0.0032, -0.0021
n = 1200
xs = np.linspace(-hx, hx, n)
ys = np.linspace(-hy, hy, n)
XX, YY = np.meshgrid(xs, ys, indexing="ij")
R = np.sqrt((x - XX) ** 2 + (y - YY) ** 2 + w ** 2)
num = np.trapezoid(np.trapezoid(1.0 / R, ys, axis=1), xs)
ana = _rect_potential(x, y, w, hx, hy)
print(f"1. rectangle potential   numeric={num:.9f}  analytic={ana:.9f}  "
      f"rel.err={abs(num-ana)/abs(num):.2e}")

# ---- 2. far field of a cuboid magnet must match a point dipole ------------
p = Params()
hx, hy, hz = 0.005, 0.005, 0.0025
M = p.Br / MU0
mdip = M * (2 * hx) * (2 * hy) * (2 * hz)
z = 0.25
d = 1e-5
psi_p = magnet_potential(0.0, 0.0, z + d, 0, 0, 0, hx, hy, hz, M)
psi_m = magnet_potential(0.0, 0.0, z - d, 0, 0, 0, hx, hy, hz, M)
Bz = -MU0 * (psi_p - psi_m) / (2 * d)
Bz_dip = MU0 / (4 * np.pi) * 2 * mdip / z ** 3
print(f"2. axial far field       cuboid={Bz*1e6:.4f} uT   dipole={Bz_dip*1e6:.4f} uT"
      f"   rel.err={abs(Bz-Bz_dip)/Bz_dip:.2e}")

# ---- 3. grid: Maxwell residuals and field magnitude -----------------------
g = FieldGrid(p, cache_dir=None)
B = g.Bmag()
nx = len(g.xs)
# ring of magnets: look at values near r = r_mag
X, Y = np.meshgrid(g.xs, g.ys)
ring = np.abs(np.hypot(X, Y) - p.r_mag) < 0.004
print(f"3. |B| at ball height ({p.z_ball*1e3:.1f} mm above magnets): "
      f"mean={B[ring].mean()*1e3:.2f} mT  max={B.max()*1e3:.2f} mT")

# divergence residual (should be ~0 by construction; check the raw one)
ds = p.grid_ds
Bx, By, Bz_ = g.F[:, :, 0], g.F[:, :, 1], g.F[:, :, 2]
dxBx = np.gradient(Bx, ds, axis=1)
dyBy = np.gradient(By, ds, axis=0)
dxBy = np.gradient(By, ds, axis=1)
dyBx = np.gradient(Bx, ds, axis=0)
curlz = dxBy - dyBx
scale = np.abs(dxBx[ring]).mean()
print(f"   curl_z residual / |dBx/dx| = {np.abs(curlz[ring]).mean()/scale:.3e}"
      "   (finite-difference consistency of the in-plane field)")

# ---- 4. harmonic content: |B| and its modulation along the ring ----------
for gap in (0.004, 0.008, 0.014, 0.020):
    pp = Params(gap=gap)
    gg = FieldGrid(pp, cache_dir=None)
    phi, U, Bm = gg.potential_on_circle(pp.r_mag, 720)
    mod = (Bm.max() - Bm.min()) / (Bm.max() + Bm.min())
    print(f"4. gap={gap*1e3:5.1f} mm  <|B|>={Bm.mean()*1e3:7.2f} mT   "
          f"|B| modulation depth = {mod*100:6.2f} %")

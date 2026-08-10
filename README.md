# Magnetic carousel — numerical model

A simulation of the IYPT problem *"Magnetic carousel"*: a ring of alternating
neodymium magnets on a spinning disc, a fixed non-magnetic plate above it, and a
steel ball on the plate that may roll **with** or **against** the disc rotation.

---

## 1. The physics being modelled

At the ball's height the alternating magnet ring produces a **travelling
magnetic wave** of wavelength λ = 2πR/N moving at speed v = ωR. Two things
happen at once, and they push the ball in *opposite* directions.

### (A) Trapping force → prograde

A soft-magnetic sphere is attracted to maxima of |B|, i.e. to a point directly
above a magnet:

    U = -½ α |B|²,     F_i = Σ_j m_j ∂B_j/∂x_i,     α = 3V/μ₀

Once the ball is captured in this moving potential well it is simply **carried
along with the disc** — a magnetic coupling, exactly like a synchronous motor.

### (B) Torque → retrograde

Because the wave travels, the **field vector itself rotates** at Ω = kv about a
horizontal (radial) axis. Eddy currents and hysteresis make the ball's
magnetisation **lag** behind B, so τ = m × B ≠ 0 and the ball spins.

The geometry is the crucial part: this spin drives the ball's *contact point*
forwards, so friction rolls the ball **backwards**. It is a rack-and-pinion.

### The competition

Coupling translation and rotation through the rolling constraint v = aω gives

    (7/5) m v̇ = F_x + τ_y / a

so the sign of `F_x + τ_y/a` decides the direction. Since (in the ideal
single-harmonic limit) the in-plane modulation of |B|² comes only from the
*cross term* between the 1st and 3rd harmonics, the trap force decays much
faster with height than the torque:

    F_trap ∝ e^(-4kz)        τ ∝ e^(-2kz)

→ **small gap / low speed = prograde, large gap / high speed = retrograde.**

---

## 2. What the code actually does

### Field (`FieldGrid`)

* Every magnet is a uniformly magnetised cuboid. Its scalar potential is
  computed in **closed form** from the equivalent magnetic surface charge
  (Coulombian model), using the rectangle antiderivative
  `Φ(u,v,w) = u·asinh(v/√(u²+w²)) + v·asinh(u/√(v²+w²)) − w·atan(uv/(wR))`.
* B = −μ₀∇ψ is sampled on a Cartesian grid **in the frame co-rotating with the
  disc**, where the field is static. The expensive evaluation therefore happens
  once and is cached to disk; at run time only a rotation is applied.
* Above the magnets the field is curl- and divergence-free, so the full 3×3
  gradient tensor is reconstructed from in-plane derivatives alone:
  `∂B_x/∂z = ∂B_z/∂x`, `∂B_y/∂z = ∂B_z/∂y`, `∂B_z/∂z = −(∂B_x/∂x + ∂B_y/∂y)`.
  This enforces Maxwell exactly instead of differencing noisy z-slices.

### Ball

A point dipole at the sphere centre with a **relaxing** moment written in the
body frame:

    dm/dt = ω_ball × m + (m_eq − m)/τ_lag
    m_eq  = min(3V|B|/μ₀, V·M_s) · B̂

The `ω_ball × m` term is what makes the torque vanish when the ball spins
synchronously with the field — i.e. genuine induction/hysteresis-motor physics,
not a hand-inserted drag term. `τ_lag` lumps eddy currents and hysteresis into
one measurable parameter.

Three ball response models are available:

* `ball_model="point"` is the original and still-default model. The entire
  sphere is represented by one relaxing dipole sampled at its centre.
* `ball_model="volume_independent"` is the original diagnostic finite-size
  model (`"volume"` remains a legacy alias). Deterministic
  spherical quadrature samples the applied field and gradient throughout the
  ball, evolves one local magnetisation-density state per volume element, and
  sums both dipole torque `dm × B` and distributed-force torque `r × dF`. It
  applies the point model's effective whole-sphere `3H` response locally and is
  retained for diagnosis, not as a self-consistent material law. Its persistent
  `M_i` histories are attached to lab-aligned quadrature cells while also using
  `omega_ball × M_i`; that combination is not a consistent material-point model.
* `ball_model="volume_demag"` is an instantaneous linear-reference model with
  intrinsic `mu_r`. A sphere boundary-element solve determines the global
  surface magnetic charge and hence the self-consistent internal demagnetizing
  field before applying `M=chi H_int`. It has no lag or material history and is
  exposed through `static_magnetic_response`; the dynamic RK4 driver rejects
  it deliberately.

The demagnetizing model uses a linear, constant permeability. Real steel has a
field-dependent permeability, so the default `mu_r=100` is a documented
reference value rather than a material measurement. It still neglects
eddy-current/skin-depth shielding, true hysteresis/remanence, nonlinear
saturation feedback, and interaction with a conductive plate.
The default quadrature is `volume_quadrature="medium"` (192 elements); `coarse`
(48) and `fine` (768) are available for convergence checks. The separate
`demag_resolution` selects the surface discretization.

Electrical conductivity `sigma` and AC permeability `mu_r_ac` are diagnostic
metadata for the Step 5A skin-depth study; they do not alter any force, torque,
or integration path. Their defaults (`6e6 S/m` and `100`) are labelled
reference assumptions, not measured properties of the ball. The standalone
`diagnose_skin_effect.py` reconstructs a body-to-lab quaternion from the
point-model trajectory and analyzes the full external-field vector in the
material frame. No skin correction is currently applied to the simulation.

`magnetic_diffusion_sphere.py` is the isolated Step 5B-1 physics kernel for a
homogeneous conducting permeable sphere. It uses the `exp(+i omega t)` phasor
convention, `kappa=(1+i)/delta`, and decoupled spherical-Bessel modes. The
module is independently tested against the exact magnetostatic multipole
limits, conductivity and frequency limits, passivity, and numerical continuity.
It also provides stable two-pole reduced models for later study. These AC
responses are not yet connected to the Carousel RK4 dynamics.

## Model development status

* **M0 — point model.** The original point-dipole sphere with the
  phenomenological `tau_lag` relaxation law remains the production model.
* **M1 — independent finite-volume diagnostic.** Retained for comparison only:
  applying the effective whole-sphere `3H` law independently at every volume
  point is not a physically self-consistent constitutive model.
* **M2 — self-consistent finite-size magnetostatics.** A linear permeable sphere
  with internal demagnetization, validated against the analytic response of a
  sphere in a uniform field.
* **M3 — magnetic diffusion / skin effect.** The exact spherical AC diffusion
  kernel and an independent external Maxwell-stress force/torque evaluator are
  validated diagnostic/reference modules. Across the Carousel snapshots,
  diffusion changed trapping/radial force by at most about **0.0157%**, vertical
  force by **0.0166%**, and the offline multi-frequency force estimate by
  **0.0199%**. In an ideal rotating uniform field its torque was only
  **2.03--2.43%** of the torque produced by the existing `tau_lag=4 ms` law.

Consequently, a production diffusion-state model is not currently justified.
The production RK4 remains unchanged, and `tau_lag` must not be interpreted as
homogeneous-sphere eddy-current diffusion. The largest unresolved magnetic
uncertainty is now the ball's measured nonlinear B-H and hysteretic response.
`magnetic_diffusion_sphere.py` and `magnetic_diffusion_wrench.py` remain
validated offline physics references for that conclusion.

### Step 6A — hysteresis-scale diagnostic

Step 5 showed that homogeneous-sphere eddy currents are far too weak to be the
primary meaning of `tau_lag`. Step 6 therefore turns to the measured B-H and
hysteretic response of the ball. The standalone `hysteresis_scale.py` and
`diagnose_hysteresis_scale.py` isolate the present relaxation law in prescribed
linear and circular fields, derive its M-H loop work and dissipation, and
compare representative Carousel trajectories with the exact Step-5 diffusion
loss. They do not alter the production force, contact, or RK4 laws.

The current lag closure has no true remanent state, nonzero quasistatic
coercivity, or rate-independent closed loop. Its finite-frequency M-H ellipse
is purely dynamic: loop area, apparent coercive intercept, and apparent
remanent intercept all vanish linearly as frequency tends to zero. For the four
representative Carousel trajectories, the lag law requires approximately
29--3216 J/m3 per dominant material-frame cycle, about 36--47 times the exact
homogeneous-sphere l=1 eddy-current loss summed over the measured spectrum.

No production hysteresis model has been selected or implemented. Quantitative
calibration still requires the ball's saturation magnetization, initial and
reversible permeability, major-loop coercivity and remanence, full major-loop
shape, and preferably minor and rotational loops over the Carousel field
range. A scalar Jiles--Atherton curve can be useful as a one-dimensional
material-characterization kernel, but it is not a sufficient final law for the
Carousel's rotating vector field.

### Step 6B-1 — standalone isotropic vector hysteresis kernel

`magnetic_hysteresis.py` implements the direct isotropic vector
Jiles--Atherton increment formulation of Bergqvist (1996), with the later
five-parameter notation used by Guérin et al. The exact frozen equations,
loading criterion, units, and references are recorded in
`docs/vector_ja_formulation.md`. The input is a prescribed constitutive field
`H [A/m]`; the output and irreversible/reversible states are vector
magnetizations `[A/m]`. The Langevin anhysteretic law and its tensor derivative
use analytic zero-field series.

The independent scalar reference has nonzero remanence, coercivity, and finite
quasistatic loop area. Under collinear excitation the vector implementation
reduces to it to floating-point accuracy. Circular, elliptical, nonplanar, and
rotated 3-D paths test vector behavior, nonnegative closed-cycle work, and
isotropic rotational covariance. Unlike `tau_lag`, its result depends on the
field path rather than traversal frequency.

The `synthetic_soft`, `synthetic_medium`, and `synthetic_hard` cases are
deliberately uncalibrated sensitivity sets. They share isotropic scalar
parameters in every direction and span `Ms=1.2--1.5 MA/m`, `a=4--9 kA/m`,
`k=25--1600 A/m`, `c=0.10--0.35`, and `alpha=3--5e-5`. They are not “steel
parameters” and no claim is made that they reproduce the ball. Passing vector
tests also does not establish quantitative rotational-loss accuracy for real
steel; measured alternating, minor, and rotational loops remain necessary.

This kernel is still offline. It has not replaced `tau_lag`, entered the
production RK4, or been combined with the self-consistent dynamic equation
`H_int=H_ext+H_demag[M]`. That nonlinear history-dependent coupling is reserved
for Step 6B-2 or later.

### Step 6B-2 — offline hysteretic sphere and demagnetization gate

`magnetic_hysteretic_sphere.py` adds a body-fixed Cartesian material mesh,
finite-cuboid arbitrary-`M` demagnetization operator, and commit-once nonlinear
vector-JA coupling. The exact assumptions and the audit of the older linear
surface-charge BEM are in `docs/hysteretic_demag_formulation.md`. The old BEM
is valid for a homogeneous linear sphere but cannot represent the volume charge
`-div(M)` of a general spatially varying hysteretic state.

The cuboid operator uses finite rectangular-face sources and an exact cube self
term. A symmetric macro/micro projection assigns `-I/3` to the uniform sphere
mode, retains resolved zero-mean interactions, and bounds unresolved voxel
modes to the physical demagnetizing spectrum. The 32/179/389-cell meshes have
represented-volume errors of 4.51%, 0.331%, and 1.91%; their conservative
half-voxel-diagonal surface scales are 43.3%, 24.7%, and 19.2% of the radius.
The nonuniform energy probe differs by 1.44% between medium and fine.

Independent linear checks pass. Uniform magnetization gives `Hdemag=-M/3` to
roundoff, and `mu_r=10, 50, 100, 500` reproduce
`3(mu_r-1)/(mu_r+2) Hext`. In the actual 8 mm FieldGrid, the medium-mesh linear
force is `Fz=-0.2195 N`, its Maxwell-stress check is `-0.2146 N`, and the
independent Step-4B surface BEM gives `-0.2026 N`. Across 3/5/8/16 mm, the new
volume-force error relative to Step 4B is 8.85%/10.68%/8.35%/4.01%.

For each physical increment the previous JA state is frozen, every nonlinear
trial starts from that same state, and the state is committed only once after
convergence. An under-relaxed predictor and Broyden iteration are used, with an
independent least-squares fallback and optional physical-path bisection for
controlled offline work. Residuals are normalized by the larger of `a` and
the RMS applied field. Tolerances from `1e-4` through `1e-8` agree in the
controlled increment test.

Self-consistent demagnetization strongly changes the synthetic-medium uniform
loop: at a 15 mT-equivalent applied amplitude, the bare material loop has
`Mr=23.98 kA/m`, `Hc=321.15 A/m`, and `W=1122.44 J/m3/cycle`; the whole-sphere
external loop has `Mr=0.724 kA/m`, `Hc=252.60 A/m`, and
`W=34.02 J/m3/cycle`. Controlled uniform rotating fields converge for all
three synthetic parameter sets with nonnegative work.

The real nonuniform Carousel-history gate does **not** yet pass. Under the
documented diagnostic compute cap, Broyden fails at early increments for the
3--8 mm medium cases and both 8 mm sensitivity cases; the 16 mm case advances
farther but also misses tolerance. A least-squares solve proves that at least
the first failed 8 mm increment has a residual root, but completing a history
by numerical-Jacobian fallback and adaptive bisection takes minutes even on the
32-cell mesh. Force, torque, and synthetic-parameter claims for the hysteretic
Carousel histories are therefore deliberately left unreported rather than
inferred from unconverged states.

This cost and robustness failure rules out direct use inside the `20 us` RK4.
Step 6C should use a reduced macro/micro or spherical-mode basis, pre-factorized
demagnetization, and a multirate constitutive update or calibrated surrogate.
Production `tau_lag` remains unchanged; skin effect and the conducting plate
are not reimplemented here. All JA parameter sets remain synthetic.

### Contact

Regularised Coulomb friction (`tanh(|u|/u_reg)`) at the contact point plus
rolling resistance and drilling friction, so stick and slip are handled by a
single smooth law and no complementarity solver is needed. A soft retaining rim
represents the lip of a real plate.

### Integration

Fixed-step RK4 on a 10-dimensional state `[x, y, vx, vy, ωx, ωy, ωz, mx, my, mz]`,
JIT-compiled with numba (falls back to pure Python if numba is absent).
About 2–4 s per 1.5 s of simulated time.

---

## 3. Validation (`test_field.py`)

| check | result |
|---|---|
| analytic rectangle potential vs. numeric double integral | rel. err **6.5e-8** |
| cuboid far field vs. point dipole | rel. err **6.0e-4** |
| curl residual of the gridded field | **8.7e-16** |

---

## 4. Results

### `fig1_regimes.png`
Trajectories, accumulated angles and velocities for a locked and a rolling case.
The key panel is the bottom one: in the retrograde case `v_tan` lies exactly on
top of `−a·ω_radial`, which is the pure-rolling condition. **This proves the
motion is torque-driven rolling, not dragging.**

### `fig2_scaling.png`
Trap force and torque scale vs. gap, the resulting direction criterion, and the
shape of the trap potential. Note: the measured decay exponents are
`e^(-1.8kz)` and `e^(-1.4kz)`, **not** the idealised 4 and 2 — the magnets are
discrete and of finite width and the gaps are not ≫ 1/k. The *ratio* still falls
monotonically, which is what sets the direction. Worth reporting honestly: the
asymptotic scaling argument gives the right trend but not the right numbers.

### `fig3_transition.png`, `fig4_phasemap.png`
The central result. Below a critical speed `Ω_ball/ω = 1.000` exactly
(synchronous lock); above it the ball flips to retrograde. The critical speed
falls monotonically with gap:

| gap [mm] | 3 | 4 | 5 | 6 | 8 | 10 | 13 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| ω_crit [rad/s] | ~14 | ~10 | ~7 | ~5 | ~3.5 | ~2.5 | ~1.5 | ~0.7 | none |

At 20 mm the ball is retrograde at every speed. The retrograde ratio also
**peaks near ω ≈ 12–16 rad/s and then decays**, and at the largest speeds and
gaps it creeps slightly positive again — the Kloss-type torque–slip curve, since
the torque falls once Ω·τ_lag ≫ 1. A distinctive, testable prediction.

### `fig5_parameters.png`
* **Ball radius / magnet number** — the retrograde effect is strongest near
  `ka = 2πa/λ ≈ 1.2` and disappears when `ka ≳ 2` (the ball averages over
  several poles). Both scans peak at the same `ka`, confirming it is the
  controlling dimensionless group.
* **Plate friction** — with `μ_k ≤ 0.05` the ball never goes retrograde. The
  torque exists but cannot be converted into translation. This is the single
  cleanest experimental test of the mechanism: *make the plate slippery and the
  retrograde motion must vanish while the prograde lock survives.*
* **Magnetisation lag** — larger `τ_lag` moves the transition to lower speed and
  deepens the retrograde branch, as expected for the eddy/hysteresis torque.

### `fig6_hysteresis.png`
Slow up/down ramp of ω, each step continuing from the **full** final state of the
previous one (position, velocity, spin and magnetisation — carrying only the
position leaves a transient that swamps the measurement). The result is a wide
bistable window: on the up-ramp the lock survives to ω ≈ 12 rad/s, on the
down-ramp the ball stays retrograde until ω ≈ 5 rad/s. Between those the
outcome depends on history — exactly the step-out / pull-in bistability of a
synchronous motor, and a sharp experimental prediction. Note the from-rest
critical speed (≈ 7 rad/s in fig3) sits inside this window, as it must.

### Animations
* `anim1_mechanism.mp4` — four panels: the rotating |B| map with the ball and its
  trail; **the B and m vectors showing the lag angle that creates the torque**;
  the travelling trap potential U(φ) with the ball's position on it; and the
  accumulated turns of disc vs. ball.
* `anim1b_mechanism_prograde.mp4` — same layout for the locked case.
* `anim2_compare.mp4` — prograde and retrograde side by side. In the prograde
  panel the ball sits pinned on a bright |B| spot; in the retrograde panel it
  slides backwards past them.

---

## 5. Running it

```bash
pip install numpy scipy matplotlib numba      # numba optional but ~50x faster
python test_field.py                          # validation
python run_single.py                          # fig1, fig2
python run_sweep.py phase                     # fig3, fig4   (~6 min)
python run_sweep.py params                    # fig5         (~7 min)
python run_sweep.py hyst                      # fig6         (~2 min)
python run_animation.py all                   # the three mp4 files
```

Field grids are cached in `fieldcache/`; delete it to force a rebuild.
All outputs land in `out/`.

Everything is driven by the `Params` dataclass:

```python
from magnetic_carousel import Params, simulate
p = Params(n_mag=12, r_mag=0.060, gap=0.005, ball_R=0.006,
           tau_lag=4e-3, mu_k=0.15, omega=16.0)
r = simulate(p, t_end=1.5)
print(r.summary())          # -> {'ratio': -0.62, ...}
```

---

## 6. Known limitations — worth stating in a report

* The ball is a **point dipole**. When `ka ≳ 1` the field varies appreciably
  across the ball and finite-size averaging matters; the code will overestimate
  both the force and the torque there. A proper treatment integrates the
  magnetisation over the sphere volume, or uses FEM.
* `τ_lag` is a **single lumped phenomenological parameter** standing in for
  eddy currents *and* hysteresis, which have different frequency dependence.
  Linear relaxation over-predicts the moment at Ω·τ ≫ 1, where real skin-effect
  shielding is stronger.
* **The plate is inert.** If it is aluminium, eddy currents in the plate itself
  attenuate and phase-shift the field reaching the ball. Not modelled — compare
  acrylic and aluminium plates experimentally to isolate this.
* Ball magnetisation is assumed reversible. Real steel acquires remanence, which
  strengthens synchronous locking; anisotropy is ignored.
* The ball is constrained to the plate (no bouncing) and the plate is perfectly
  flat and level.

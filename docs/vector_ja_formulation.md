# Frozen isotropic vector Jiles–Atherton formulation (Step 6B-1)

## Sources and scope

This diagnostic implements the isotropic specialization of A. J. Bergqvist,
“A simple vector generalisation of the Jiles–Atherton model of hysteresis,”
*IEEE Transactions on Magnetics* 32(5), 4213–4215 (1996),
doi:10.1109/20.539337. The direct differential form and its use with the five
isotropic parameters are also described by C. Guérin et al., “Using a
Jiles–Atherton vector hysteresis model for isotropic magnetic materials with
the finite element method, Newton–Raphson method, and relaxation procedure,”
*International Journal of Numerical Modelling* 30(5), e2189 (2017),
doi:10.1002/jnm.2189.

For equation transcription, Bergqvist's increment law is reproduced explicitly
as Eqs. (3.46)–(3.48) in A. R. P. J. Vijn, *Development of a Closed-loop
Degaussing System: Towards Magnetic Unobservable Vessels*, TU Delft
dissertation (2021), doi:10.4233/uuid:ae431c9d-cbb3-4b5f-88c9-86d4ee81cabe.
This secondary reproduction
was checked against the notation and loading condition reported by Guérin and
the later vector-JA stability literature; no missing equation was supplied from
memory.

This is an uncalibrated constitutive reference. It is not coupled to the
Carousel RK4 or to dynamic demagnetization.

## Direct convention

The model is **direct**:

* input: applied/internal magnetic field history **H** [A/m];
* output: magnetization history **M** [A/m];
* state: irreversible and reversible accumulated increments,
  **M** = **M**_irr + **M**_rev [A/m].

Direct H-to-M evolution matches the eventual applied-field-to-magnetization
direction of the sphere solver. In Step 6B-2, H must mean the self-consistent
internal field, not B/mu0 and not the externally sampled field by itself:

    H_int = H_ext + H_demag[M].

## Equations

For an isotropic material, define

    H_e = H + alpha M,

    M_an(H_e) = Ms L(|H_e|/a) H_e/|H_e|,

    L(x) = coth(x) - 1/x,

    chi_f = (M_an - M)/k,

    xi = partial M_an / partial H_e.

The frozen Bergqvist increment equation is

    dM = u_f (chi_f . dH_e)^+ + c xi dH_e,

where

    u_f = chi_f/|chi_f|,
    x^+ = max(x, 0),
    dH_e = dH + alpha dM.

The first term is accumulated as dM_irr and the second as dM_rev. When
|chi_f|=0 the irreversible increment is exactly zero. For a trial path
direction dH, each loading case is solved algebraically:

    G_active   = u_f outer chi_f + c xi,
    G_inactive = c xi,
    dM/dH = [I - alpha G]^-1 G.

The active solution is accepted only when chi_f . dH_e > 0; otherwise the
inactive solution is used. This is the positive-part loading/reversal rule in
the cited formulation, not an added epsilon switch.

For a collinear one-dimensional path this reduces to

    dM = sign(chi_f) (chi_f dH_e)^+ + c (dM_an/dH_e) dH_e,

which is the scalar reference implemented here.

The Langevin function and its radial/tangential derivatives use analytic series
at |H_e|/a near zero; no norm clipping is used. At H_e=0,

    M_an = 0,
    xi = Ms/(3a) I.

## Parameters and units

* `Ms` [A/m]: saturation magnetization.
* `a` [A/m]: anhysteretic Langevin field scale.
* `k` [A/m]: irreversible pinning field scale; strictly positive.
* `c` [1]: reversible fraction, 0 <= c <= 1.
* `alpha` [1]: dimensionless mean-field coupling in H_e=H+alpha M.

All Step-6B-1 parameter sets are explicitly synthetic. They are not steel data.

## Work and limitations

For a closed path the reported signed loss is

    W = mu0 integral H . dM  [J/m^3/cycle].

The orientation is chosen so passive major loops give W >= 0. The tests check
alternating, circular, elliptical, and three-dimensional paths. The classical
vector JA extension is phenomenological and is not a general thermodynamically
consistent hysteresis operator; nonnegative work on tested paths is numerical
evidence, not a proof for every possible history.

Rotational covariance and a vector loop do not establish quantitative accuracy
for real rotational hysteresis. Calibration requires measured alternating,
minor-loop, and rotational/vector data for the actual ball material.

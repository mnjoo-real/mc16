# Self-consistent hysteretic sphere formulation (Step 6B-2)

## Audit of the previous solver

`DemagSphereSolver` in `magnetic_carousel.py` is a boundary-element reference
for one homogeneous, linear susceptibility.  It solves for surface charge on
the geometric sphere and then applies `M=chi*(H_external+H_demag)`.  Its
spherical-harmonic interior reconstruction and solid-angle self term are valid
for that problem.

It is not an arbitrary-magnetization operator.  A spatially varying nonlinear
magnetization generally creates both surface charge `sigma_m=M.n` and volume
charge `rho_m=-div(M)`.  A surface-only unknown cannot represent the latter.
The old geometry, field samplers, analytic linear benchmarks, and Maxwell
surface machinery can be reused, but its constitutive BEM matrix cannot be
used as the distributed JA interaction operator.

## Material mesh and arbitrary-M operator

The new mesh consists of body-fixed Cartesian cuboid material cells whose
centres lie inside the sphere.  Unlike the old Gauss points, these are actual
persistent material regions with a finite volume and a six-component JA state.
If used in mechanics later, positions and vector states must rotate with the
body.

For two distinct cells, the `H/M` block is obtained from the closed-form
magnetic-charge potential of all six rectangular cuboid faces.  Its gradient
is evaluated once during precomputation by a centred difference of `1e-5`
cell widths.  The coincident cube has the exact finite self tensor `-I/3`; no
point-dipole singularity or distance cutoff is used.

Cell-centre collocation on the voxelized boundary does not exactly retain the
uniform spherical mode.  The symmetric operator is split into uniform and
zero-mean subspaces.  The exact sphere block `-I/3` is assigned to the three
uniform modes and the resolved cuboid operator is retained for zero-mean
modes.  Unresolved voxel-scale eigenvalues are projected onto the physical
demagnetizing spectrum `[-1,0]`.  Thus the final precomputed operator is
symmetric, negative semidefinite, rotationally covariant when mesh and vectors
are rotated together, and accepts an arbitrary piecewise M field:

    H_demag,i = sum_j D_ij M_j.

This macro/micro consistency projection is disclosed because the nonuniform
modes remain a discretization, not an exact analytic sphere solution.  Their
mesh-refinement test is therefore a required gate.

## Nonlinear constitutive coupling

At physical increment `n -> n+1`, the solver preserves the committed
`(H_int,n, M_irr,n, M_rev,n)` state.  For every nonlinear trial field it
re-integrates JA from that same committed state and evaluates

    R(H_trial) = H_trial - H_external,n+1 - D M_JA(H_int,n -> H_trial).

An under-relaxed Picard predictor is followed by Broyden's second quasi-Newton
method with an Armijo line search.  The normalized convergence criterion is

    RMS_i |R_i| / max(a, RMS_i |H_external,i|) <= tolerance.

The best attained residual, iteration count, and failure flag are retained.
Only after convergence are the trial JA state and internal field committed.
Nonlinear iterations therefore cannot manufacture extra hysteresis history.
For exactly uniform applied histories, the code solves the exact invariant
spherical macro mode directly; this avoids excitation of roundoff-scale voxel
modes and is equivalent to the full projected operator.

## Work, force, and scope

Intrinsic material work uses local internal fields.  A whole-sphere loop
plotted against external field is a different observable because demagnetizing
work changes its remanence, coercivity, and shape.

Offline force and torque use vacuum Maxwell stress on a spherical surface
outside every material cell.  The external-only stress is subtracted to reduce
quadrature cancellation error.  The field on that surface is the applied
`FieldGrid` vacuum field plus the field of all finite cuboid cells.

The parameter sets remain synthetic.  The solver is not connected to the
production RK4, does not add skin effect or plate conductivity, and is not a
quantitative steel model without measured alternating and rotational loops.

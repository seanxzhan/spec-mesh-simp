# Eigenanalysis of the Laplace-Beltrami Operator

## The problem: find the "natural frequencies" of a surface

Given a triangle mesh with cotangent Laplacian `L` and mass matrix `M`, we solve the
**generalized eigenvalue problem**:

```
L φ = λ M φ
```

This gives pairs `(λᵢ, φᵢ)` — eigenvalues and eigenvectors — ordered by increasing `λ`.

## What this means physically

Think of the mesh as a drum membrane. Each eigenvector `φᵢ` is a **vibration mode** — a
standing wave pattern on the surface. The eigenvalue `λᵢ` is proportional to the squared
frequency of that mode.

- `λ₁ ≈ 0`: The trivial mode (constant function). Every vertex vibrates in unison — no
  spatial variation. We discard this one.
- `λ₂, λ₃, λ₄` (small): The lowest non-trivial modes. These are **smooth, global** — they
  partition the surface into a few large regions of positive/negative sign. On a sphere,
  these are the three coordinates (x, y, z projections).
- `λ₅₀, λ₁₀₀` (large): High-frequency modes. **Oscillatory and local** — they capture fine
  geometric detail.

## Why the mass matrix M matters

Without `M`, the standard eigenproblem `L φ = λ φ` would give eigenvectors that are
orthonormal in the Euclidean sense (`φᵢᵀ φⱼ = δᵢⱼ`). But vertices aren't uniformly
distributed — a dense region would dominate simply because it has more vertices.

The mass matrix fixes this: `M_ii` = area owned by vertex `i`. The generalized problem
ensures eigenvectors are **M-orthonormal**:

```
Φᵀ M Φ = I
```

This means eigenvectors are orthonormal with respect to the **area-weighted inner product**
on the surface, not the vertex-count inner product. A vertex owning more area contributes
more to the integral.

## Key properties

1. **Eigenvalues are real and non-negative**: `L` is positive semi-definite, `M` is positive
   definite, so all `λᵢ ≥ 0`.

2. **Eigenvectors form a basis**: Any square-integrable function on the surface can be
   written as `f = Σᵢ cᵢ φᵢ` (spectral decomposition, like Fourier series).

3. **Low frequencies encode global shape**: The first ~30 eigenvectors capture the overall
   geometry — bends, protrusions, symmetries. Two shapes with similar low-frequency spectra
   have similar global geometry.

4. **Multiplicity = symmetry**: On a sphere, `λ₂ = λ₃ = λ₄` (3-fold degeneracy) because
   the sphere is rotationally symmetric — there are three independent "first harmonics"
   (like the p-orbitals in quantum mechanics).

## Why this matters for mesh simplification

The spectral mesh simplification paper (Lescoat et al. 2020) makes this key observation:

> Standard simplification (QSlim) preserves **visual appearance** — but destroys the
> eigenvectors. If you compute functional maps, geodesic distances, or shape descriptors
> on the simplified mesh, you get garbage.

The spectral cost function measures: **"if I collapse this edge, how much do the first K
eigenvectors change?"** By minimizing this, the simplified mesh has nearly the same
low-frequency harmonics as the original — which means all downstream spectral computations
(functional maps, HKS, WKS, diffusion distances) remain faithful.

## The functional map connection

Given eigenbases `Φ` (fine mesh, n×K) and `Φ̃` (coarse mesh, m×K), and a restriction
matrix `P` (m×n) that maps fine signals to coarse:

```
C = Φ̃ᵀ M̃ P Φ        (the K×K functional map)
```

If simplification perfectly preserves the spectrum: **C = Identity**. The further C deviates
from identity, the more the simplification has distorted the spectral structure.

## Practical notes

- We use `scipy.sparse.linalg.eigsh` with **shift-invert** mode (`sigma=0`) to find the
  smallest eigenvalues efficiently. This transforms the problem to finding the *largest*
  eigenvalues of `(L - 0·M)⁻¹ M`, which ARPACK handles well.

- Computing K eigenvectors of a mesh with n vertices costs roughly O(n·K²) with a good
  sparse solver. For n=3000, K=30 this is <1 second.

- The paper's rule of thumb: output mesh should have **≥ 3K vertices** for faithful
  preservation of K eigenvectors.

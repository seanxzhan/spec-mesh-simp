# Spectral Mesh Simplification

> Thibault Lescoat, Hsueh-Ti Derek Liu, Jean-Marc Thiery, Alec Jacobson, Tamy Boubekeur, Maks Ovsjanikov.
> *Eurographics 2020 (Computer Graphics Forum, Volume 39, Number 2).*
> PDF: [`Lescoat et al. - 2020 - Spectral Mesh Simplification.pdf`](./Lescoat%20et%20al.%20-%202020%20-%20Spectral%20Mesh%20Simplification.pdf)

## TL;DR

Standard mesh simplification (QSlim / Garland & Heckbert) preserves **visual appearance** — it minimizes
geometric error from planes fitted at each vertex. But downstream geometry processing (geodesics, shape
matching, segmentation) depends on the **spectrum** of the Laplace-Beltrami operator, which appearance-based
methods destroy. This paper keeps the standard greedy edge-collapse pipeline but replaces the Quadric Error
Metric with a **spectral preservation cost**: "how much does collapsing this edge change the first K
eigenvectors of the Laplacian?" The output is still a manifold triangle mesh (not just a point cloud + matrix),
so you can recompute the cotangent Laplacian on it with standard tools and get nearly the same eigenpairs as
the original.

## The problem

- **Appearance-preserving simplification** (QSlim) keeps the mesh looking right but scrambles intrinsic
  properties: eigenvectors drift, eigenvalues shift, spectral distances distort. Anything that runs on
  `M⁻¹L` downstream (functional maps, diffusion distances, HKS/WKS) gets garbage on the coarse mesh.
- **Algebraic spectral coarsening** (Liu et al. 2019, Nasikun et al. 2018) preserves the operator well but
  outputs a **subset of points + a custom sparse matrix** — no mesh, no connectivity. You can't feed the
  result into algorithms that need triangles (BCICP correspondence, remeshing, rendering).
- **Gap**: no method existed that outputs a proper triangle mesh *and* preserves the spectrum.

This paper fills that gap.

## Background you need

### The cotangent Laplacian

On a triangle mesh `M = (V, F)` with `n` vertices, the discrete Laplace-Beltrami operator is a pair of
sparse matrices:

- `L ∈ ℝⁿˣⁿ` — the **stiffness matrix** (cotangent weights)
- `M ∈ ℝⁿˣⁿ` — the **mass matrix** (diagonal, lumped vertex areas)

The generalized eigenproblem `LΦ = MΦΛ` gives you eigenvectors `Φ = [φ₁ … φₙ]` (the "harmonics" of the
surface) and eigenvalues `Λ = diag(λ₁, …, λₙ)`. Low-frequency eigenvectors capture global shape; high ones
capture local detail.

### Functional maps

A **functional map** `C ∈ ℝᴷˣᴷ` encodes correspondence between two shapes in the spectral domain. Given
eigenbases `Φ` (source) and `Φ̃` (target), and a restriction/prolongation operator `P`:

```
C = Φ̃ᵀ M̃ P Φ
```

If the simplification perfectly preserved the spectrum, `C` would be the **identity matrix** (or block-diagonal
when eigenvalues have multiplicity). The further `C` deviates from identity, the worse the spectral
preservation.

### Edge collapse (recap)

Garland & Heckbert's greedy simplification:
1. Assign a cost to each edge.
2. Put all edges in a priority queue (min-cost first).
3. Pop the cheapest edge, collapse it (merge two vertices into one), update neighbors' costs.
4. Repeat until you reach the target vertex count.

The paper keeps this exact pipeline — only the **cost function** changes.

## Method

### 1. Input / output

- **Input**: manifold triangle mesh `M = (V, F)`, target vertex count `N`, number of eigenvectors to
  preserve `K` (default 100).
- **Output**: simplified mesh `M̃ = (Ṽ, F̃)` with `|Ṽ| = N`, plus optionally a restriction matrix `P`.

### 2. The spectral cost (the core idea)

The paper wants the Laplacian to **commute** with the restriction. For any signal `f` on the fine mesh:

```
         M⁻¹L                          M̃⁻¹L̃
fine: f ───────→ Δf        coarse: Pf ───────→ Δ̃(Pf)
      │                                │
      │ restrict P                     │ should be equal
      ▼                                ▼
      P(Δf)               =?          Δ̃(Pf)
```

"Apply the Laplacian then restrict" should equal "restrict then apply the coarse Laplacian." When these
disagree, the coarse mesh is spectrally unfaithful.

Formalize this over the first `K` eigenvectors `F ∈ ℝⁿˣᴷ` (columns = eigenvectors of `L`):

```
E = ‖PM⁻¹LF − M̃⁻¹L̃PF‖²_M̃                           (Eq. 2)
```

where `‖X‖²_M̃ = tr(Xᵀ M̃ X)` is the mass-weighted Frobenius norm. Precompute `Z = M⁻¹LF = FΛ` once
(since `LF = MFΛ`), so the cost becomes:

```
E = ‖PZ − M̃⁻¹L̃PF‖²_M̃
```

This also implicitly preserves **eigenvalues** (proven via connection to Laplacian commutativity norm).

### 3. Locality — why it's efficient

The mass-weighted norm decomposes per vertex:

```
E = Σᵥ Eᵥ,      Eᵥ = M̃ᵥ · ‖rowᵥ(PZ − M̃⁻¹L̃PF)‖²
```

When you collapse edge `e = (u, v)`:
- Only `M̃` and `L̃` entries in the **1-ring** of the collapse change.
- Therefore only vertices `H = {u, v} ∪ N₁(u,v)` have their `Eᵥ` affected.
- You only need the **2-ring** of `{u,v}` to evaluate the cost.

```
cost(e) = E_after − E_before = Σ_{w∈H} E^after_w − Σ_{w∈H} E^before_w
```

This makes per-edge cost evaluation **local** — same asymptotic as QSlim, just with larger constant (because
you touch `K`-wide rows of `F` and `Z`).

### 4. Merged vertex positioning

Collapsing edge `(u, v)` produces a merged vertex at position `w(α) = (1−α)u + αv`, `α ∈ [0,1]`. The
position affects `P`, `M̃`, and `L̃` non-linearly.

Strategy: **1D quadratic approximation on the edge**. Evaluate `cost(e, α)` at `α ∈ {0, 0.5, 1}`, fit a
parabola, minimize it analytically to get `α*`. Clamp to `[0, 1]`.

This beats both "always use midpoint" (less accurate) and "full 3D quadratic" (2.5× slower, negligible
improvement) — see Appendix A of the paper.

### 5. Restriction matrix

Each edge collapse `(u, v) → w` produces a local restriction matrix `Q`:
```
Q_ŵŵ = 1−α  (contribution from u)
Q_ŵv = α    (contribution from v)
Q_ww = 1    (all other vertices unchanged)
```

The global restriction matrix accumulates: `P = Qₙ Qₙ₋₁ … Q₂ Q₁`. All coefficients are positive, rows
sum to 1.

After each collapse, update: `P ← QP`, `F ← QF`, `Z ← QZ`.

### 6. Full algorithm

```
Algorithm 1: Spectrum-Preserving Edge-Collapse

Input:  mesh M = (V, F), target size N, K eigenvectors to preserve
Output: simplified mesh M̃ = (Ṽ, F̃)

SETUP:
  Compute L, M (cotangent Laplacian + mass matrix)
  Solve (L, M) for first K eigenvectors → F, eigenvalues → Λ
  Z ← FΛ  (equivalently, Z = M⁻¹LF)
  P ← Identity

INIT QUEUE:
  for each edge e ∈ M:
    compute cost(e) using Eq. 2 (local 2-ring evaluation)
    find optimal α* via 1D quadratic fit
    push (e, cost, α*) to priority queue

DECIMATE:
  while |Ṽ| > N and queue not empty:
    pop edge e = (u,v) with lowest cost
    collapse e, position merged vertex at (1−α*)u + α*v
    update P ← QP,  F ← QF,  Z ← QZ
    for each neighbor edge n of the collapse:
      recompute cost(n), update queue
```

## Evaluation

### Spectral quality metrics

Two norms on the functional map `C` between input and output:

| Norm | Formula | Measures |
|------|---------|----------|
| Laplacian commutativity `‖·‖_L` | `‖CΛ − Λ̃C‖² / ‖C‖²` | Do eigenvalues commute through C? |
| Orthonormality `‖·‖_D` | `‖CᵀC − Id‖²` | Is C an orthonormal transformation? |

Both should be zero for perfect preservation. The paper proves (Theorem 1): `C` is orthonormal *and*
commutes with eigenvalues **if and only if** it preserves eigenfunctions and eigenvalues exactly.

### Key findings

- **vs. QSlim**: spectral norms 10–20× better across the dataset.
- **vs. Liu et al. 2019** (algebraic): slightly worse spectrally at extreme reduction (2% of vertices), but
  Liu et al. degrades at moderate reduction (their optimization gets harder with more output vertices).
- **vs. Nasikun et al. 2018**: comparable quality at small output sizes, but non-deterministic and much
  higher storage (228 B/vertex vs. 48 B/vertex for a mesh).
- **Deterministic**: only depends on input — no random initialization.
- **Storage**: 48 bytes/vertex (just a mesh: 3 floats for position + triangle indices), vs. 228–262 B/v
  for algebraic methods that store a custom operator.
- **Rule of thumb**: output should have **≥ 3× as many vertices as eigenvectors preserved** (`|Ṽ| ≥ 3K`)
  for faithful spectral preservation.

### Timings

On Intel Xeon 3.0 GHz, 32 GB RAM (C++ implementation with Spectra eigensolver):

- **Setup** (eigensolver): dominates for large meshes. Scales with input size, can be swapped for faster
  solvers.
- **Reduction**: linear in number of collapsed edges. Slower than QSlim (per-edge cost touches K-wide
  matrices) but much faster than Liu et al.'s full optimization.

## Applications

### 1. Spectral distance computation

Distances defined on the spectrum (diffusion, biharmonic, commute-time, WKS) are computed on the coarse
mesh as a proxy for the fine mesh. With 96% vertex reduction:

- **Ours**: distances closely match ground truth, smooth iso-contours.
- **QSlim**: distances exhibit spurious local optima and distorted iso-contours.
- Computing spectral distances on the coarse proxy is **~18× faster** than on the full mesh.

### 2. Faster functional maps (shape matching)

Hierarchical scheme: simplify both shapes → compute functional map on coarse pair → lift to fine.

```
C_{X,Y} = C_{Y,Ỹ} · C_{X̃,Ỹ} · C_{X,X̃}
           ↑           ↑           ↑
      from our      matching     from our
      reduction     algorithm    reduction
```

Using BCICP (which needs connectivity — algebraic methods can't be used here): on TOSCA dataset,
correspondence quality significantly better with spectral simplification than QSlim simplification. BCICP
runs out of memory on 30K-vertex meshes but handles 600-vertex simplified proxies in ~67 seconds.

## Limitations

- **Eigensolver is the bottleneck** for large meshes. The reduction itself is linear, but computing K
  eigenvectors of a large sparse system is expensive. Can be mitigated with fast approximate eigensolvers
  or by pre-reducing with QSlim to ~100–200K vertices first (small spectral impact at that scale).
- **Per-edge cost is more expensive than QSlim**: each evaluation touches K-column-wide matrices over
  the 2-ring. Still O(1) per edge, but a larger constant.
- **Edge flips don't help during reduction**: they overfit and create holes (Fig. 18). Post-process flips
  give marginal improvement not worth the cost.
- **No theoretical guarantees** on spectral preservation (unlike some graph-reduction methods). It's a
  greedy heuristic — works well empirically.

## Relationship to related work

- **Garland & Heckbert 1997 (QSlim)**: same algorithm skeleton (greedy edge collapse + priority queue),
  different cost. QSlim uses the Quadric Error Metric (geometric distance to tangent planes). This paper
  replaces it with the spectral commutativity cost. You could combine both (add a small QEM regularizer)
  to get a mesh that preserves both geometry and spectrum.
- **Liu et al. 2019 (Spectral Coarsening)**: the algebraic counterpart. Directly optimizes a sparse
  operator matrix on a selected point subset. Better spectral quality at extreme reduction but: no mesh
  output, non-deterministic, 5× storage, optimization gets harder as output grows, can't be used with
  algorithms requiring connectivity.
- **Nasikun et al. 2018**: approximates eigenpairs via GPU-accelerated methods on Poisson-disk samples.
  Fast for small outputs, uses geodesic-distance-based Laplacians (denser than cotangent). Non-deterministic,
  high memory usage, no mesh output.
- **Functional maps (Ovsjanikov et al. 2012)**: the evaluation framework. The functional map `C` between
  input and output measures spectral fidelity. This paper's cost function is essentially "minimize how far
  `C` is from identity" expressed through Laplacian commutativity.

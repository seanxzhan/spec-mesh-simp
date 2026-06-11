# Thin-Shell Stiffness Matrix

The stiffness matrix `K` answers: "if I displace vertex `v` by a small amount `δ`, how much force pushes it back?"

```
f = K u
```

where `u` is a displacement vector (3 DOFs per vertex: x, y, z) and `f` is the resulting internal force. For `n` vertices, `K` is `3n × 3n`, sparse, symmetric, and positive semi-definite.

## Two physical modes of resistance

A thin shell (cloth, paper, sheet metal) resists deformation in two distinct ways:

```
K = K_membrane + K_bending
```

### Membrane: in-plane stretch and shear

Imagine pulling a sheet of rubber sideways. The triangle distorts in its own plane — edges stretch, angles change. Membrane stiffness resists this.

**Construction (CST — Constant Strain Triangle):**

For each triangle with vertices `p0, p1, p2`:

1. Set up a local 2D coordinate frame in the triangle's plane (two orthonormal axes `e1, e2`).

2. Project the three vertices into this local frame:
   ```
   q0 = (0, 0)
   q1 = (|p1-p0|, 0)
   q2 = (dot(p2-p0, e1), dot(p2-p0, e2))
   ```

3. Compute the strain-displacement matrix `B` (3×6). This encodes how displacements of the 3 nodes produce strain (stretch in x, stretch in y, shear):
   ```
   B = (1 / det_J) * [ y2,     0,    -y2,  0,   0,  0  ]
                      [ 0,    x2-x1,   0, -x2,   0,  x1 ]
                      [ x2-x1, y2,   -x2,  -y2, x1,  0  ]
   ```
   where `det_J = x1 * y2` (twice the triangle area in local coords).

4. Apply the material law. This uses **linear elasticity** (Hooke's law) under the **plane stress** assumption (no through-thickness stress, valid for thin shells). The constitutive matrix relating stress to strain is:
   ```
   D = E/(1-ν²) * [ 1   ν   0       ]
                   [ ν   1   0       ]
                   [ 0   0   (1-ν)/2 ]
   ```
   `E` = Young's modulus (how stiff), `ν` = Poisson's ratio (how much sideways bulge when you pull lengthwise).

   This is **not** Neo-Hookean. Neo-Hookean is a hyperelastic model for large deformations (nonlinear stress-strain). Linear elasticity assumes small strains — stress is proportional to strain (`σ = Dε`), giving a constant stiffness matrix. The distinction:
   - **Linear (Hooke):** `K` is constant, computed once. Valid for small deformations.
   - **Neo-Hookean / StVK:** `K` depends on the current deformation state. Must be recomputed as the mesh deforms. Needed for large stretches (rubber, inflating balloons).

   For the Schur-complement simplification metric, linear elasticity is the right starting point — it gives a fixed `K` that accumulates cleanly. Nonlinear models would require linearization around a specific pose.

5. Element stiffness in local 2D (6×6):
   ```
   Ke_local = area * thickness * Bᵀ D B
   ```
   This comes from minimizing elastic strain energy `U = (1/2) ∫ εᵀDε dV`. Since `ε = Bu` (constant over the triangle) and volume = area × thickness, the integral collapses to `U = (1/2) uᵀ (Bᵀ D B · area · thickness) u`. The stiffness matrix is `∂²U/∂u²`.

6. Rotate back to 3D global frame. Each node's 2 local DOFs map to 3 global DOFs via the rotation `R = [e1 | e2]` (3×2). The element stiffness becomes 9×9.

7. Scatter into the global matrix at the DOF indices of the three vertices.

**Sparsity:** Only vertices sharing a triangle are coupled → **1-ring** connectivity (same as the cotangent Laplacian).

### Bending: out-of-plane folding

Imagine folding a piece of paper along a crease. The bending stiffness resists changes in dihedral angle between adjacent triangles.

**Construction (discrete hinge):**

For each interior edge shared by two triangles — a "hinge" with 4 vertices `[v0, v1, v2, v3]` where `(v0, v1)` is the shared edge and `v2, v3` are the opposite vertices:

```
    v2
   / \
  v0---v1   (shared edge)
   \ /
    v3
```

1. Compute the dihedral angle `θ` between the two triangle normals.

2. The bending energy (linearized around rest state `θ₀`):
   ```
   E_bend = (1/2) * kb * |e|² / (A0 + A1) * (θ - θ₀)²
   ```
   where:
   - `kb = E h³ / (12(1 - ν²))` — bending modulus (note: scales as `h³`, so thin shells bend easily)
   - `|e|` — length of the shared edge
   - `A0, A1` — areas of the two triangles

3. Compute the gradient of `θ` with respect to each vertex position. Using cotangent weights:
   ```
   ∂θ/∂v2 = n0_hat / h0        (normal of tri 0, divided by height of v2)
   ∂θ/∂v3 = -n1_hat / h1       (normal of tri 1, divided by height of v3)
   ∂θ/∂v0 = -cot(∠v0 in tri0) * ∂θ/∂v2 - cot(∠v0 in tri1) * ∂θ/∂v3
   ∂θ/∂v1 = -cot(∠v1 in tri0) * ∂θ/∂v2 - cot(∠v1 in tri1) * ∂θ/∂v3
   ```

4. The element stiffness (12×12) is a rank-1 outer product:
   ```
   Ke = coeff * grad ⊗ grad
   ```
   where `grad` is the 12-vector concatenation of the 4 vertex gradients, and `coeff = kb * |e|² / (A0 + A1)`.

5. Scatter into the global matrix.

**Sparsity:** Each hinge connects 4 vertices — the two edge vertices plus two opposite vertices. The opposite vertices are in the **2-ring** of each other. So bending stiffness has wider sparsity than membrane.

## Comparison with the cotangent Laplacian

| Property | Laplacian `L` | Stiffness `K` |
|----------|--------------|---------------|
| Size | `n × n` (1 DOF/vert) | `3n × 3n` (3 DOFs/vert) |
| Connectivity | 1-ring | 1-ring (membrane) + 2-ring (bending) |
| What it encodes | Geometric smoothness | Mechanical resistance |
| Physical meaning | "How different is this vertex from its neighbors?" | "How much force to deform this region?" |
| Off-diagonal entry | `-0.5(cot α + cot β)` for edge geometry | Depends on material + geometry |
| Null space | Constant functions (rigid translation) | Rigid body motions (3 translations + 3 rotations = 6D) |

The cotangent Laplacian is actually the membrane stiffness of a 1D scalar field on the surface. When you restrict to a single DOF per vertex (e.g., height above a plane), the CST membrane stiffness reduces to cotangent weights. They share DNA — the Laplacian is the "geometric skeleton" of mechanical stiffness.

## Why this matters for mesh simplification

The stiffness matrix encodes *how the mesh deforms*. Two meshes at the same vertex count can have wildly different `K` — different triangle shapes produce different stiffness distributions. A simulation-aware simplification metric would preserve the important mechanical properties of `K` during decimation, rather than just preserving geometric appearance (QEM) or spectral properties (Laplacian eigenvectors).

The Schur complement connects directly: eliminating a vertex from `K` via static condensation folds its mechanical influence into its neighbors — accumulating "how this region resists deformation" in exactly the same way QEM accumulates "how far this region is from its original planes."

## Material parameters

| Parameter | Symbol | Cloth-like | Steel plate |
|-----------|--------|-----------|-------------|
| Young's modulus | `E` | ~1 MPa | ~200 GPa |
| Poisson's ratio | `ν` | 0.3 | 0.3 |
| Thickness | `h` | 0.5 mm | 5 mm |
| Bending modulus | `kb = Eh³/12(1-ν²)` | ~1.1e-8 | ~2.3e6 |

The ratio `kb / (E*h)` = `h²/12(1-ν²)` tells you how bendy vs. stretchy a shell is. Thin cloth: bending is ~10⁶× weaker than membrane. Thick plate: comparable.

## References

- Grinspun et al., "Discrete Shells" (SCA 2003) — the discrete hinge bending model
- Bridson et al., "Simulation of Clothing" (SCA 2005) — practical cloth FEM
- Zienkiewicz & Taylor, "The Finite Element Method" — CST derivation (Ch. 6)

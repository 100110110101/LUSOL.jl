# LUSOL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://100110110101.github.io/LUSOL.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://100110110101.github.io/LUSOL.jl/dev/)
[![Build Status](https://github.com/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia wrapper for the Fortran 90 sparse LU factorization and rank-1 / column update library [**LUSOL Version by Jacob Williams**](https://github.com/jacobwilliams/lusol/) (originally developed by Michael Saunders, Philip Gill, and Walter Murray at Stanford SOL [**LUSOL early versions**](https://web.stanford.edu/group/SOL/software/lusol/), powered cross-platform by [**LUSOL_jll**](https://github.com/JuliaBinaryWrappers/LUSOL_jll.jl).

`LUSOL.jl` provides a familiar `LinearAlgebra`-compatible interface (`F.L`, `F.U`, `F.p`, `F.q`, `F.Rs`, `F \ b`, `ldiv!`, `mul!`) alongside fast in-place and out-of-place factor updates (`singlecolumnupdate!`, `lowrankupdate!`) and $\mathcal{O}(\text{nnz})$ zero-refactorization converters for existing `UMFPACK`, `CHOLMOD`, `KLU.jl`, and `PureKLU.jl` factorizations.

## Installation

Until `LUSOL.jl` is registered in the Julia General Registry, install it directly from GitHub via the Julia package manager (`]`):

```julia
pkg> add https://github.com
```

## Quick Start

### 1. Sparse LU Factorization & Linear Solves

`LUSOL.lu` computes a sparse LU factorization $A[p, q] = L \cdot U$ and integrates with Julia's standard `LinearAlgebra` API:

```julia
using LUSOL, SparseArrays, LinearAlgebra

A = sprand(100, 100, 0.05) + 5I
b = rand(100)

F = LUSOL.lu(A)

# Solve linear systems (vector or multi-RHS matrix)
x = F \ b
ldiv!(F, copy(b))          # in-place solve
y = F * x                  # matrix-vector product via L * U

# Inspect factors and permutations (lazy extraction)
L, U, p, q, Rs = F
@assert F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
```

### 2. Fast Column & Low-Rank Updates

Update an existing factorization without re-factorizing from scratch:

```julia
# Replace column j of A: A[:, j] = anew
j = 10
anew = sprand(100, 0.1)
F_col = singlecolumnupdate(F, j, anew)   # returns updated copy
singlecolumnupdate!(F, j, anew)          # updates F in-place

# Symmetric rank-1 / low-rank update: A ← A + C * C'
c = sprand(100, 0.1)
F_lr = lowrankupdate(F, c)               # returns updated copy
lowrankupdate!(F, c)                     # updates F in-place
```

### 3. Seamless Interoperability with UMFPACK, CHOLMOD & KLU

Already have a factorization from `LinearAlgebra.lu` (`UmfpackLU`), `cholesky` / `ldlt` (`CHOLMOD.Factor`), `KLU.jl`, or `PureKLU.jl`? Pass it directly to `LUSOL.lu`, `singlecolumnupdate`, or `lowrankupdate` to convert its factors in $\mathcal{O}(\text{nnz})$ time without re-factorizing $A$:

```julia
# From Julia stdlib UMFPACK or CHOLMOD
F_umf  = LinearAlgebra.lu(A)
F_chol = LinearAlgebra.cholesky(A' * A + 5I)

F_updated_1 = singlecolumnupdate(F_umf, j, anew)
F_updated_2 = lowrankupdate(F_chol, c)

# Via automatic Package Extensions when KLU or PureKLU is loaded
using KLU
F_klu = KLU.klu(A)
F_updated_3 = singlecolumnupdate(F_klu, j, anew)
```

## Exported Functions

* `LUSOL.lu(A; check=true, kwargs...)` / `LUSOL.lu(F; kwargs...)`
* `singlecolumnupdate(F, j, anew; kwargs...)` & `singlecolumnupdate!(F, j, anew)`
* `lowrankupdate(F, C; kwargs...)` & `lowrankupdate!(F, C)`

## Acknowledgments

* Underlying Fortran 90 implementation by [Jacob Williams (`jacobwilliams/lusol`)](https://github.com), based on the original Stanford SOL LUSOL package by Michael A. Saunders et al.

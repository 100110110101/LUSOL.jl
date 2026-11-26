```@meta
CurrentModule = LUSOL
```

# LUSOL.jl

A Julia wrapper for the Fortran 90 sparse LU factorization and rank-1 / column update library [**LUSOL**](https://github.com), backed cross-platform by `LUSOL_jll`.

## Overview

`LUSOL.jl` provides:
* Sparse LU factorization via [`LUSOL.lu`](@ref) compatible with Julia's `LinearAlgebra` interface (`F.L`, `F.U`, `F.p`, `F.q`, `F.Rs`, `F \ b`, `ldiv!`, `mul!`).
* Fast in-place and out-of-place factor updates via [`singlecolumnupdate!`](@ref) and [`lowrankupdate!`](@ref).
* Zero-refactorization $\mathcal{O}(\text{nnz})$ converters for existing `UMFPACK`, `CHOLMOD`, `KLU.jl`, and `PureKLU.jl` factorizations.

## Quick Start

```julia
using LUSOL, SparseArrays, LinearAlgebra

A = sprand(50, 50, 0.1) + 5I
b = rand(50)

# Compute sparse LU factorization
F = LUSOL.lu(A)
x = F \ b

# Replace column 5 without re-factorizing A
anew = sprand(50, 0.1)
singlecolumnupdate!(F, 5, anew)

# Symmetric rank-1 update: A ← A + c * c'
c = sprand(50, 0.1)
lowrankupdate!(F, c)
```

## Interoperability with Other Sparse Solvers

Existing factorizations from `LinearAlgebra.lu` (`UmfpackLU`), `cholesky` / `ldlt` (`CHOLMOD.Factor`), `KLU.jl`, or `PureKLU.jl` can be passed directly to `LUSOL.lu`, `singlecolumnupdate`, or `lowrankupdate`:

```julia
F_umf = LinearAlgebra.lu(A)
F_updated = singlecolumnupdate(F_umf, 5, anew)
```

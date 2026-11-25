# ==============================================================================
# 1. Struct Definition & Lazy Extraction Helper
# ==============================================================================

"""
    LUSOLFactorization <: Factorization{Float64}

Factorization object returned by `LUSOL.lu(A)`. Supports property access (`F.L`, `F.U`,
`F.p`, `F.q`, `F.Rs`), iteration/destructuring, `issuccess(F)`, `ldiv!`, and `F \\ b`.
"""
mutable struct LUSOLFactorization <: Factorization{Float64}
    parm::LUSOLparm
    objA::LUSOLobjA
    res::LUSOLres
    L_cache::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    U_cache::Union{Nothing,SparseMatrixCSC{Float64,Int}}
end

function LUSOLFactorization(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres)
    return LUSOLFactorization(parm, objA, res, nothing, nothing)
end

function _ensure_extracted!(F::LUSOLFactorization)
    if isnothing(getfield(F, :L_cache)) || isnothing(getfield(F, :U_cache))
        L, U = extrLU(getfield(F, :parm), getfield(F, :objA), getfield(F, :res))
        setfield!(F, :L_cache, L)
        setfield!(F, :U_cache, U)
    end
    return getfield(F, :L_cache), getfield(F, :U_cache)
end

# ==============================================================================
# 2. Properties & Destructuring (`L, U, p, q, Rs = F`)
# ==============================================================================

function Base.getproperty(F::LUSOLFactorization, s::Symbol)
    if s === :L
        L, _ = _ensure_extracted!(F)
        return L
    elseif s === :U
        _, U = _ensure_extracted!(F)
        return U
    elseif s === :p
        return getfield(F, :res).p
    elseif s === :q
        return getfield(F, :res).q
    elseif s === :Rs
        return ones(Float64, getfield(F, :objA).m)
    elseif s === :(:)
        L, U = _ensure_extracted!(F)
        return (L, U, F.p, F.q, F.Rs)
    else
        return getfield(F, s)
    end
end

Base.propertynames(::LUSOLFactorization, private::Bool=false) =
    private ? (:L, :U, :p, :q, :Rs, :parm, :objA, :res, :L_cache, :U_cache) : (:L, :U, :p, :q, :Rs)

function Base.iterate(F::LUSOLFactorization, state::Int=1)
    state == 1 && return (F.L, 2)
    state == 2 && return (F.U, 3)
    state == 3 && return (F.p, 4)
    state == 4 && return (F.q, 5)
    state == 5 && return (F.Rs, 6)
    return nothing
end

# ==============================================================================
# 3. Base Interface (`size`, `show`, `\`)
# ==============================================================================

Base.size(F::LUSOLFactorization) = (getfield(F, :objA).m, getfield(F, :objA).n)

function Base.size(F::LUSOLFactorization, dim::Integer)
    dim == 1 && return getfield(F, :objA).m
    dim == 2 && return getfield(F, :objA).n
    dim > 2 && return 1
    throw(BoundsError(size(F), dim))
end

function Base.show(io::IO, ::MIME"text/plain", F::LUSOLFactorization)
    m, n = size(F)
    print(io, "LUSOLFactorization ($m×$n, inform = $(getfield(F, :res).inform))")
end

function Base.:\(F::LUSOLFactorization, b::AbstractVector{<:Real})
    return LinearAlgebra.ldiv!(F, Vector{Float64}(b))
end

function Base.:\(F::LUSOLFactorization, B::AbstractMatrix{<:Real})
    m, n = size(F)
    size(B, 1) == m || throw(DimensionMismatch("RHS has $(size(B, 1)) rows, expected $m."))
    X = Matrix{Float64}(undef, n, size(B, 2))
    return LinearAlgebra.ldiv!(X, F, B)
end

# ==============================================================================
# 4. LinearAlgebra Interface (`issuccess`, `ldiv!`, `LinearAlgebra.lu`)
# ==============================================================================

LinearAlgebra.issuccess(F::LUSOLFactorization) = getfield(F, :res).inform == 0

function LinearAlgebra.ldiv!(F::LUSOLFactorization, b::Vector{Float64})
    m, n = size(F)
    m == n || throw(DimensionMismatch("In-place ldiv! requires a square matrix, got ($m×$n)."))
    length(b) == m || throw(DimensionMismatch("RHS has length $(length(b)), expected $m."))
    lu6sol!(getfield(F, :parm), getfield(F, :objA), getfield(F, :res), copy(b), b; mode=5)
    return b
end


function LinearAlgebra.ldiv!(x::AbstractVector{Float64}, F::LUSOLFactorization, b::AbstractVector{<:Real})
    m, n = size(F)
    length(b) == m || throw(DimensionMismatch("RHS has length $(length(b)), expected $m."))
    length(x) == n || throw(DimensionMismatch("Destination has length $(length(x)), expected $n."))
    sol = lu6sol(getfield(F, :parm), getfield(F, :objA), getfield(F, :res), Vector{Float64}(b); mode=5)
    return copyto!(x, sol)
end

function LinearAlgebra.ldiv!(X::AbstractMatrix{Float64}, F::LUSOLFactorization, B::AbstractMatrix{<:Real})
    m, n = size(F)
    size(B, 1) == m || throw(DimensionMismatch("RHS has $(size(B, 1)) rows, expected $m."))
    size(X) == (n, size(B, 2)) || throw(DimensionMismatch("Destination has size $(size(X)), expected ($n, $(size(B, 2)))."))
    for j in axes(B, 2)
        sol = lu6sol(getfield(F, :parm), getfield(F, :objA), getfield(F, :res), Vector{Float64}(@view B[:, j]); mode=5)
        @views copyto!(X[:, j], sol)
    end
    return X
end

LinearAlgebra.lu(F::LUSOLFactorization) = lu(F)

# ==============================================================================
# 5. LUSOL.lu Constructors & Factorization Converters
# ==============================================================================

"""
    lu(A::SparseMatrixCSC{Float64, Int}; check::Bool = true, kwargs...) -> LUSOLFactorization

Compute the sparse LU factorization of `A` using LUSOL, returning a `LUSOLFactorization`
object compatible with Julia's `LinearAlgebra.lu` interface (`F.L`, `F.U`, `F.p`, `F.q`, `F.Rs`, `F \\ b`).
"""
function lu(A::SparseMatrixCSC{Float64,Int}; check::Bool=true, kwargs...)
    parm, objA, res = lu1fac(A; kwargs...)
    F = LUSOLFactorization(parm, objA, res)
    if check && !issuccess(F)
        throw(LinearAlgebra.SingularException(getindex(parm.luparm, 16) + 1))
    end
    return F
end

function lu(A::AbstractMatrix; check::Bool=true, kwargs...)
    return lu(SparseMatrixCSC{Float64,Int}(A); check=check, kwargs...)
end

"""
    lu(F::LUSOLFactorization) -> LUSOLFactorization

Returns a deep copy of an existing `LUSOLFactorization` (consistent with `LinearAlgebra.lu(F)`).
"""
function lu(F::LUSOLFactorization)
    return LUSOLFactorization(
        deepcopy(getfield(F, :parm)),
        deepcopy(getfield(F, :objA)),
        deepcopy(getfield(F, :res))
    )
end

"""
    lu(F::SparseArrays.UMFPACK.UmfpackLU; kwargs...) -> LUSOLFactorization

Converts a standard library `UmfpackLU` factorization into a `LUSOLFactorization`
without re-factorizing `A`.
"""
function lu(F::SparseArrays.UMFPACK.UmfpackLU; kwargs...)
    return cnvrtUMFPACK2LUSOL(F; kwargs...)
end

"""
    lu(F::SparseArrays.CHOLMOD.Factor; kwargs...) -> LUSOLFactorization

Converts a standard library `CHOLMOD.Factor` (`cholesky` or `ldlt`) into a
`LUSOLFactorization` without re-factorizing `A`.
"""
function lu(F::SparseArrays.CHOLMOD.Factor; kwargs...)
    return cnvrtCHOLMOD2LUSOL(F; kwargs...)
end

"""
    LUSOLFactorization <: Factorization{Float64}

Factorization object returned by `LUSOL.lu(A)`. Supports property access (`F.L`, `F.U`,
`F.p`, `F.q`, `F.Rs`), iteration/destructuring, `issuccess(F)`, and linear solves via `F \\ b`.
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

# Support destructuring: L, U, p, q, Rs = F
function Base.iterate(F::LUSOLFactorization, state::Int=1)
    state == 1 && return (F.L, 2)
    state == 2 && return (F.U, 3)
    state == 3 && return (F.p, 4)
    state == 4 && return (F.q, 5)
    state == 5 && return (F.Rs, 6)
    return nothing
end

Base.size(F::LUSOLFactorization) = (getfield(F, :objA).m, getfield(F, :objA).n)
Base.size(F::LUSOLFactorization, dim::Integer) = size(F)[dim]

LinearAlgebra.issuccess(F::LUSOLFactorization) = getfield(F, :res).inform == 0

function Base.:\(F::LUSOLFactorization, b::AbstractVector{<:Real})
    return lu6sol(getfield(F, :parm), getfield(F, :objA), getfield(F, :res), Vector{Float64}(b); mode=5)
end

function Base.:\(F::LUSOLFactorization, B::AbstractMatrix{<:Real})
    m, n = size(F)
    size(B, 1) == m || throw(DimensionMismatch("RHS has $(size(B, 1)) rows, expected $m."))
    X = Matrix{Float64}(undef, n, size(B, 2))
    for j in axes(B, 2)
        X[:, j] .= lu6sol(getfield(F, :parm), getfield(F, :objA), getfield(F, :res), Vector{Float64}(@view B[:, j]); mode=5)
    end
    return X
end

function Base.show(io::IO, mime::MIME"text/plain", F::LUSOLFactorization)
    m, n = size(F)
    print(io, "LUSOLFactorization ($m×$n, inform = $(getfield(F, :res).inform))")
end

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

# Fallback for general matrices (e.g. SparseMatrixCSC{Int, Int} or dense Matrix)
function lu(A::AbstractMatrix; check::Bool=true, kwargs...)
    return lu(SparseMatrixCSC{Float64,Int}(A); check=check, kwargs...)
end


"""
    singlecolumnupdate!(F::LUSOLFactorization, jrep::Integer, anew::AbstractVector{<:Real};
                        mode1::Int = 1, mode2::Int = 1) -> LUSOLFactorization

In-place column replacement of the factorization `F`: replaces column `jrep`
with `anew` via `lu8rpc!` and invalidates cached `L` and `U` matrices.
"""
function singlecolumnupdate!(F::LUSOLFactorization, jrep::Integer, anew::AbstractVector{<:Real};
    mode1::Int=1, mode2::Int=1)
    lu8rpc!(getfield(F, :parm), getfield(F, :objA), getfield(F, :res),
        Int(jrep), Vector{Float64}(anew); mode1=mode1, mode2=mode2)

    # Invalidate lazy cache so F.L and F.U will be re-extracted on next access
    setfield!(F, :L_cache, nothing)
    setfield!(F, :U_cache, nothing)

    return F
end

"""
    singlecolumnupdate(F::LUSOLFactorization, jrep::Integer, anew::AbstractVector{<:Real};
                       mode1::Int = 1, mode2::Int = 1) -> LUSOLFactorization

Non-mutating column replacement of the factorization `F`: returns a new
`LUSOLFactorization` where column `jrep` is replaced by `anew`.
"""
function singlecolumnupdate(F::LUSOLFactorization, jrep::Integer, anew::AbstractVector{<:Real};
    mode1::Int=1, mode2::Int=1)
    F_new = LUSOLFactorization(
        deepcopy(getfield(F, :parm)),
        deepcopy(getfield(F, :objA)),
        deepcopy(getfield(F, :res))
    )
    return singlecolumnupdate!(F_new, jrep, anew; mode1=mode1, mode2=mode2)
end

function singlecolumnupdate(F::SparseArrays.UMFPACK.UmfpackLU, jrep::Integer, anew::AbstractVector{<:Real}; kwargs...)
    return singlecolumnupdate!(LUSOL.cnvrtUMFPACK2LUSOL(F; kwargs...), jrep, anew)
end

function singlecolumnupdate(F::SparseArrays.CHOLMOD.Factor, jrep::Integer,
    anew::AbstractVector{<:Real}; kwargs...)
    return singlecolumnupdate!(cnvrtCHOLMOD2LUSOL(F; kwargs...), jrep, anew)
end

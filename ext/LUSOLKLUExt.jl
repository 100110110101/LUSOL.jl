module LUSOLKLUExt

using LUSOL
using KLU
using SparseArrays

function LUSOL.cnvrtKLU2LUSOL(F::KLU.KLUFactorization; kwargs...)
    return LUSOL._pack_klu_to_lusol(
        SparseMatrixCSC{Float64,Int}(F.L),
        SparseMatrixCSC{Float64,Int}(F.U),
        SparseMatrixCSC{Float64,Int}(F.F),
        Vector{Int}(F.p),
        Vector{Int}(F.q),
        Vector{Float64}(F.Rs);
        invert_rs=true, # In KLU.jl: (A ./ Rs)[p, q] = L*U + F
        kwargs...
    )
end

function LUSOL.lu(F::KLU.KLUFactorization; kwargs...)
    return LUSOL.cnvrtKLU2LUSOL(F; kwargs...)
end

function LUSOL.singlecolumnupdate(F::KLU.KLUFactorization, jrep::Integer, anew::AbstractVector{<:Real}; kwargs...)
    return LUSOL.singlecolumnupdate!(LUSOL.cnvrtKLU2LUSOL(F; kwargs...), jrep, anew)
end

function LUSOL.lowrankupdate(F::KLU.KLUFactorization, C::AbstractVecOrMat{<:Real}; kwargs...)
    return LUSOL.lowrankupdate!(LUSOL.cnvrtKLU2LUSOL(F; kwargs...), C)
end

end # module

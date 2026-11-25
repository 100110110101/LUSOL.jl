module LUSOLPureKLUExt

using LUSOL
using PureKLU
using SparseArrays

function LUSOL.cnvrtKLU2LUSOL(F::PureKLU.KLUFactorization; kwargs...)
    return LUSOL._pack_klu_to_lusol(
        SparseMatrixCSC{Float64,Int}(F.L),
        SparseMatrixCSC{Float64,Int}(F.U),
        SparseMatrixCSC{Float64,Int}(F.F),
        Vector{Int}(F.p),
        Vector{Int}(F.q),
        Vector{Float64}(F.Rs);
        invert_rs=true,
        kwargs...
    )
end

function LUSOL.lu(F::PureKLU.KLUFactorization; kwargs...)
    return LUSOL.cnvrtKLU2LUSOL(F; kwargs...)
end

function LUSOL.singlecolumnupdate(F::PureKLU.KLUFactorization, jrep::Integer, anew::AbstractVector{<:Real}; kwargs...)
    return LUSOL.singlecolumnupdate!(LUSOL.cnvrtKLU2LUSOL(F; kwargs...), jrep, anew)
end

function LUSOL.lowrankupdate(F::PureKLU.KLUFactorization, C::AbstractVecOrMat{<:Real}; kwargs...)
    return LUSOL.lowrankupdate!(LUSOL.cnvrtKLU2LUSOL(F; kwargs...), C)
end

end # module

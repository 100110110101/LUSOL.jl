"""
    lowrankupdate!(F::LUSOLFactorization, c::AbstractVector{<:Real}) -> LUSOLFactorization

Update the LU factorization `F` of `A` in-place to a factorization of `A + c * c'`,
by applying column updates (`singlecolumnupdate!`) for each non-zero entry of `c`.
"""
function lowrankupdate!(F::LUSOLFactorization, c::AbstractVector{<:Real})
    m, n = size(F)
    m == n || throw(DimensionMismatch("lowrankupdate! requires a square matrix, got ($m × $n)."))
    length(c) == m || throw(DimensionMismatch("Vector c must have length $m, got $(length(c))."))

    c_vec = Vector{Float64}(c)

    for j in 1:n
        cj = c_vec[j]
        if cj != 0.0
            # Reconstruct current column j using the up-to-date permutations F.p and F.q
            q_pos = findfirst(==(j), F.q)
            pinv = invperm(F.p)
            col_j = (F.L*Vector(F.U[:, q_pos]))[pinv]
            anew = col_j .+ cj .* c_vec
            singlecolumnupdate!(F, j, anew)
        end
    end

    return F
end


"""
    lowrankupdate!(F::LUSOLFactorization, C::AbstractMatrix{<:Real}) -> LUSOLFactorization

Update the LU factorization `F` of `A` in-place to a factorization of `A + C * C'`.
"""
function lowrankupdate!(F::LUSOLFactorization, C::AbstractMatrix{<:Real})
    for col in eachcol(C)
        lowrankupdate!(F, col)
    end
    return F
end

"""
    lowrankupdate(F::LUSOLFactorization, C::AbstractVecOrMat{<:Real}) -> LUSOLFactorization

Non-mutating rank-1 (or rank-k) update of the factorization `F` of `A` to `A + C * C'`.
"""
function lowrankupdate(F::LUSOLFactorization, C::AbstractVecOrMat{<:Real})
    F_new = LUSOLFactorization(
        deepcopy(getfield(F, :parm)),
        deepcopy(getfield(F, :objA)),
        deepcopy(getfield(F, :res))
    )
    return lowrankupdate!(F_new, C)
end

function lowrankupdate(F::SparseArrays.UMFPACK.UmfpackLU, C::AbstractVecOrMat{<:Real}; kwargs...)
    return lowrankupdate!(LUSOL.cnvrtUMFPACK2LUSOL(F; kwargs...), C)
end

function lowrankupdate(F::SparseArrays.CHOLMOD.Factor,
    C::AbstractVecOrMat{<:Real}; kwargs...)
    return lowrankupdate!(cnvrtCHOLMOD2LUSOL(F; kwargs...), C)
end

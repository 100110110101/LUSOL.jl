function LinearAlgebra.mul!(y::Vector{Float64}, F::LUSOLFactorization, v::Vector{Float64})
    m, n = size(F)
    length(v) == n || throw(DimensionMismatch("Expected vector of length $n, got $(length(v))."))
    length(y) == m || throw(DimensionMismatch("Expected output vector of length $m, got $(length(y))."))

    u_prod = F.L * (F.U * v[F.q])
    @inbounds for k in 1:m
        y[F.p[k]] = u_prod[k]
    end
    return y
end

function Base.:*(F::LUSOLFactorization, v::AbstractVector{<:Real})
    y = zeros(Float64, size(F, 1))
    return LinearAlgebra.mul!(y, F, Vector{Float64}(v))
end

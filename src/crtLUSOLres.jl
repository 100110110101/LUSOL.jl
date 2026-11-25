"""
    LUSOLres

Holds permutation vectors, row/column pointers, workspace arrays, and the
status code (`inform`) populated during LUSOL factorization and updates.
"""
mutable struct LUSOLres
    p::Vector{Int}
    q::Vector{Int}
    lenc::Vector{Int}
    lenr::Vector{Int}
    locc::Vector{Int}
    locr::Vector{Int}
    iploc::Vector{Int}
    iqloc::Vector{Int}
    ipinv::Vector{Int}
    iqinv::Vector{Int}
    w::Vector{Float64}
    inform::Int
end

"""
    crtLUSOLres(objA::LUSOLobjA) -> LUSOLres

Allocates all permutation, pointer, and workspace arrays matching the dimensions of `objA`.
"""
function crtLUSOLres(objA::LUSOLobjA)
    return LUSOLres(
        zeros(Int, objA.m),     # p
        zeros(Int, objA.n),     # q
        zeros(Int, objA.n),     # lenc
        zeros(Int, objA.m),     # lenr
        zeros(Int, objA.n),     # locc
        zeros(Int, objA.m),     # locr
        zeros(Int, objA.n),     # iploc
        zeros(Int, objA.m),     # iqloc
        zeros(Int, objA.m),     # ipinv
        zeros(Int, objA.n),     # iqinv
        zeros(Float64, objA.n), # w
        0                       # inform
    )
end

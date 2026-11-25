"""
    LUSOLobjA

Stores the matrix dimensions, nonzero entries, and coordinate indices (`a`, `indc`, `indr`)
including buffer space for fill-in.
"""
mutable struct LUSOLobjA
    m::Int
    n::Int
    nelem::Int
    lena::Int
    a::Vector{Float64}
    indc::Vector{Int}
    indr::Vector{Int}
end

"""
    crtLUSOLobjA(A::SparseMatrixCSC{Float64, Int}, parm::LUSOLparm = crtLUSOLparm()) -> LUSOLobjA

Creates and initializes a `LUSOLobjA` from a square sparse matrix `A`.
"""
function crtLUSOLobjA(A::SparseMatrixCSC{Float64,Int}, parm::LUSOLparm=crtLUSOLparm())
    m, n = size(A)

    rows, cols, vals = findnz(A)
    nelem = length(vals)

    # Calculate initial buffer size for fill-in
    lena = max(ceil(Int, parm.lena_factor * nelem) + 2 * max(m, n), nelem, 1000)


    a = zeros(Float64, lena)
    indc = zeros(Int, lena)
    indr = zeros(Int, lena)

    a[1:nelem] .= vals
    indc[1:nelem] .= rows
    indr[1:nelem] .= cols

    return LUSOLobjA(m, n, nelem, lena, a, indc, indr)
end
"""
    crtLUSOLobjA(m::Int, n::Int, a_vals::Vector{Float64},
                 indc_vals::Vector{Int}, indr_vals::Vector{Int},
                 parm::LUSOLparm = crtLUSOLparm()) -> LUSOLobjA

Constructs a populated `LUSOLobjA` object directly from pre-assembled value and index
vectors, padding the storage arrays to `lena` for LUSOL's internal workspace.
"""
function crtLUSOLobjA(m::Int, n::Int, a_vals::Vector{Float64},
    indc_vals::Vector{Int}, indr_vals::Vector{Int},
    parm::LUSOLparm=crtLUSOLparm())
    nelem = length(a_vals)
    lena_factor = parm.lena_factor
    min_lena = getindex(parm.luparm, 13)
    lena = max(lena_factor * nelem, 2 * max(m, n), min_lena)

    a = zeros(Float64, lena)
    indc = zeros(Int, lena)
    indr = zeros(Int, lena)

    copyto!(a, 1, a_vals, 1, nelem)
    copyto!(indc, 1, indc_vals, 1, nelem)
    copyto!(indr, 1, indr_vals, 1, nelem)

    return LUSOLobjA(m, n, nelem, lena, a, indc, indr)
end


"""
    lu6sol!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, v::Vector{Float64}, w::Vector{Float64}; mode::Int = 5) -> LUSOLres

Low-level in-place solve using the factorization in `(parm, objA, res)` and pre-allocated
work vectors `v` (length `m`) and `w` (length `n`).
"""
function lu6sol!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, v::Vector{Float64}, w::Vector{Float64}; mode::Int=5)
    m, n = objA.m, objA.n
    if length(v) != m
        throw(DimensionMismatch("Work vector v must have length m = $m, got $(length(v))."))
    elseif length(w) != n
        throw(DimensionMismatch("Work vector w must have length n = $n, got $(length(w))."))
    end

    inform_ref = Ref{Int}(0)

    ccall(
        (:__lusol_MOD_lu6sol, liblusol),
        Cvoid,
        (
            Ref{Int}, Ref{Int}, Ref{Int},
            Ptr{Float64}, Ptr{Float64},
            Ref{Int}, Ptr{Int}, Ptr{Float64},
            Ptr{Float64}, Ptr{Int}, Ptr{Int},
            Ptr{Int}, Ptr{Int},
            Ptr{Int}, Ptr{Int}, Ptr{Int}, Ptr{Int},
            Ref{Int}
        ),
        mode, m, n,
        v, w,
        objA.lena, parm.luparm, parm.parmlu,
        objA.a, objA.indc, objA.indr,
        res.p, res.q,
        res.lenc, res.lenr, res.locc, res.locr,
        inform_ref
    )

    res.inform = inform_ref[]

    if res.inform != 0 && res.inform != 1
        error("lu6sol! failed with LUSOL status code inform = $(res.inform)")
    end

    return res
end

"""
    lu6sol(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, b::Vector{Float64}; mode::Int = 5) -> Vector{Float64}

Solves a linear system using the factorization stored in `(parm, objA, res)`.

# Modes
- `1`: Solve `L * v = b` (`v` has length `m`)
- `2`: Solve `L' * v = b` (`v` has length `m`)
- `3`: Solve `U * x = b` (`b` has length `m`, `x` has length `n`)
- `4`: Solve `U' * x = b` (`b` has length `n`, `x` has length `m`)
- `5`: Solve `A * x = b` (`b` has length `m`, `x` has length `n`) [default]
- `6`: Solve `A' * x = b` (`b` has length `n`, `x` has length `m`)
"""
function lu6sol(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, b::Vector{Float64}; mode::Int=5)
    m, n = objA.m, objA.n

    if mode in (1, 2, 3, 5) && length(b) != m
        throw(DimensionMismatch("Expected right-hand side vector of length m = $m for mode = $mode, got $(length(b))."))
    elseif mode in (4, 6) && length(b) != n
        throw(DimensionMismatch("Expected right-hand side vector of length n = $n for mode = $mode, got $(length(b))."))
    end

    v = zeros(Float64, m)
    w = zeros(Float64, n)

    if mode in (1, 2, 3, 5)
        v .= b
    else
        w .= b
    end

    lu6sol!(parm, objA, res, v, w; mode=mode)

    return mode in (3, 5) ? w : v
end

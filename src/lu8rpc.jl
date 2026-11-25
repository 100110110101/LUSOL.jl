"""
    lu8rpc!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, jrep::Int,
            v::Vector{Float64}, w::Vector{Float64};
            mode1::Int = 1, mode2::Int = 1) -> Tuple{Float64, Float64}

Low-level in-place column replacement (`lu8rpc`) using pre-allocated work vectors
`v` (length `m`, containing `a_new` when `mode2 == 1`) and `w` (length `n`).
Returns `(diag, vnorm)` and updates `res.inform`.
"""
function lu8rpc!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, jrep::Int,
    v::Vector{Float64}, w::Vector{Float64};
    mode1::Int=1, mode2::Int=1)
    m, n = objA.m, objA.n
    if !(1 <= jrep <= n)
        throw(BoundsError("Column index jrep = $jrep must be between 1 and n = $n."))
    elseif length(v) != m
        throw(DimensionMismatch("Vector v must have length m = $m, got $(length(v))."))
    elseif length(w) != n
        throw(DimensionMismatch("Work vector w must have length n = $n, got $(length(w))."))
    end

    inform_ref = Ref{Int}(0)
    diag_ref = Ref{Float64}(0.0)
    vnorm_ref = Ref{Float64}(0.0)

    ccall(
        (:__lusol_MOD_lu8rpc, liblusol),
        Cvoid,
        (
            Ref{Int}, Ref{Int}, Ref{Int}, Ref{Int}, Ref{Int},
            Ptr{Float64}, Ptr{Float64},
            Ref{Int}, Ptr{Int}, Ptr{Float64},
            Ptr{Float64}, Ptr{Int}, Ptr{Int},
            Ptr{Int}, Ptr{Int},
            Ptr{Int}, Ptr{Int}, Ptr{Int}, Ptr{Int},
            Ref{Int}, Ref{Float64}, Ref{Float64}
        ),
        mode1, mode2, m, n, jrep,
        v, w,
        objA.lena, parm.luparm, parm.parmlu,
        objA.a, objA.indc, objA.indr,
        res.p, res.q,
        res.lenc, res.lenr, res.locc, res.locr,
        inform_ref, diag_ref, vnorm_ref
    )

    res.inform = inform_ref[]

    # inform in (-1, 0, 1): Update successful (rank decreased, unchanged, or increased)
    # inform == 2: Update completed, but numerically unstable (refactorization recommended)
    if res.inform == 7
        error("lu8rpc! ran out of storage (inform = 7). Increase `lena_factor` in `crtLUSOLparm`.")
    elseif !(res.inform in (-1, 0, 1, 2))
        error("lu8rpc! failed with LUSOL status code inform = $(res.inform)")
    end

    return diag_ref[], vnorm_ref[]
end

"""
    lu8rpc!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, jrep::Int, anew::Vector{Float64};
            mode1::Int = 1, mode2::Int = 1) -> Tuple{Float64, Float64}

Replaces column `jrep` of the factored matrix with `anew` and updates the LU factors
in-place via the Bartels-Golub-Reid algorithm.
"""
function lu8rpc!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, jrep::Int, anew::Vector{Float64};
    mode1::Int=1, mode2::Int=1)
    v = copy(anew)
    w = zeros(Float64, objA.n)
    return lu8rpc!(parm, objA, res, jrep, v, w; mode1=mode1, mode2=mode2)
end
"""
    lu8rpc(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, jrep::Int, anew::Vector{Float64};
           mode1::Int = 1, mode2::Int = 1) -> Tuple{LUSOLparm, LUSOLobjA, LUSOLres, Float64, Float64}

Non-mutating version of `lu8rpc!`. Copies `(parm, objA, res)`, updates the factorization
by replacing column `jrep` with `anew`, and returns `(parm_new, objA_new, res_new, diag, vnorm)`.
"""
function lu8rpc(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres, jrep::Int, anew::Vector{Float64};
    mode1::Int=1, mode2::Int=1)
    parm_new = deepcopy(parm)
    objA_new = deepcopy(objA)
    res_new = deepcopy(res)
    diag_val, vnorm_val = lu8rpc!(parm_new, objA_new, res_new, jrep, anew; mode1=mode1, mode2=mode2)
    return parm_new, objA_new, res_new, diag_val, vnorm_val
end

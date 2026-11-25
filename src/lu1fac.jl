"""
    lu1fac!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres) -> LUSOLres

Computes the sparse LU factorization `P * A * Q = L * U` for general `m × n` matrices.
Automatically expands `objA` storage arrays and retries if `inform == 7`.
"""
function lu1fac!(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres)
    # Backup original entries in case reallocation (inform == 7) is triggered
    a0 = objA.a[1:objA.nelem]
    indc0 = objA.indc[1:objA.nelem]
    indr0 = objA.indr[1:objA.nelem]

    inform_ref = Ref{Int}(0)
    res.inform = 7  # Start condition to enter the loop at least once

    while res.inform == 7
        ccall(
            (:__lusol_MOD_lu1fac, liblusol),
            Cvoid,
            (
                Ref{Int}, Ref{Int}, Ref{Int}, Ref{Int},
                Ptr{Int}, Ptr{Float64},
                Ptr{Float64}, Ptr{Int}, Ptr{Int},
                Ptr{Int}, Ptr{Int},
                Ptr{Int}, Ptr{Int}, Ptr{Int}, Ptr{Int},
                Ptr{Int}, Ptr{Int}, Ptr{Int}, Ptr{Int},
                Ptr{Float64}, Ref{Int}
            ),
            objA.m, objA.n, objA.nelem, objA.lena,
            parm.luparm, parm.parmlu,
            objA.a, objA.indc, objA.indr,
            res.p, res.q,
            res.lenc, res.lenr, res.locc, res.locr,
            res.iploc, res.iqloc, res.ipinv, res.iqinv,
            res.w, inform_ref
        )

        res.inform = inform_ref[]

        # Reallocate and restore original matrix if LUSOL ran out of space
        if res.inform == 7
            minlen = getindex(parm.luparm, 12)
            new_lena = max(2 * objA.lena, minlen + 1000)

            objA.lena = new_lena
            objA.a = zeros(Float64, new_lena)
            objA.indc = zeros(Int, new_lena)
            objA.indr = zeros(Int, new_lena)

            objA.a[1:objA.nelem] .= a0
            objA.indc[1:objA.nelem] .= indc0
            objA.indr[1:objA.nelem] .= indr0
        end
    end

    if res.inform != 0 && res.inform != 1
        error("lu1fac! failed with LUSOL status code inform = $(res.inform)")
    end

    return res
end
"""
    lu1fac(A::SparseMatrixCSC{Float64, Int}; kwargs...) -> Tuple{LUSOLparm, LUSOLobjA, LUSOLres}

Convenience method that creates `parm`, `objA`, and `res` from `A` and immediately
computes the sparse LU factorization via `lu1fac!`.
"""
function lu1fac(A::SparseMatrixCSC{Float64,Int}; kwargs...)
    parm = crtLUSOLparm(; kwargs...)
    objA = crtLUSOLobjA(A, parm)
    res = crtLUSOLres(objA)
    lu1fac!(parm, objA, res)
    return parm, objA, res
end

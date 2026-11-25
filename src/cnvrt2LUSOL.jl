"""
    cnvrtUMFPACK2LUSOL(F::SparseArrays.UMFPACK.UmfpackLU; kwargs...) -> LUSOLFactorization

Internal helper that converts a standard library `UmfpackLU` object directly into a
`LUSOLFactorization` in O(nnz(L) + nnz(U)) time without re-factorizing `A`.
"""
function cnvrtUMFPACK2LUSOL(F::SparseArrays.UMFPACK.UmfpackLU; kwargs...)
    m, n = size(F)
    m == n || throw(DimensionMismatch("Expected square UmfpackLU factorization, got ($m × $n)."))

    L_umf = F.L
    U_umf = F.U
    p = Vector{Int}(F.p)
    q = Vector{Int}(F.q)
    Rs = Vector{Float64}(F.Rs)

    # Absorb row scaling D = diag(Rs[p]) into L_tilde = D^-1 * L * D and U_tilde = D^-1 * U
    d = Rs[p]

    parm = crtLUSOLparm(; kwargs...)
    nelem_est = nnz(L_umf) + nnz(U_umf)
    lena_factor = parm.lena_factor
    min_lena = getindex(parm.luparm, 13)
    lena = max(ceil(Int, lena_factor * nelem_est), 2 * max(m, n), min_lena)

    # Allocate storage arrays for LUSOLobjA
    a = zeros(Float64, lena)
    indc = zeros(Int, lena)
    indr = zeros(Int, lena)

    # Allocate index/pointer arrays for LUSOLres
    lenc = zeros(Int, n)
    lenr = zeros(Int, m)
    locc = zeros(Int, n)
    locr = zeros(Int, m)

    # 1. Pack U_tilde row-by-row at the beginning of a (diagonal entry first per row)
    Ut = sparse(transpose(U_umf))
    Ut_colptr = SparseArrays.getcolptr(Ut)
    Ut_rowval = rowvals(Ut)
    Ut_nzval = nonzeros(Ut)

    ptr = 1
    for k in 1:m
        orig_row = p[k]
        locr[orig_row] = ptr
        inv_dk = 1.0 / d[k]

        # Diagonal entry (k, k) first (required by LUSOL)
        diag_val = 0.0
        for idx in Ut_colptr[k]:(Ut_colptr[k+1]-1)
            if Ut_rowval[idx] == k
                diag_val = Ut_nzval[idx] * inv_dk
                break
            end
        end
        a[ptr] = diag_val
        indr[ptr] = q[k]
        ptr += 1

        # Strictly upper triangular entries of row k
        for idx in Ut_colptr[k]:(Ut_colptr[k+1]-1)
            c = Ut_rowval[idx]
            if c != k
                val = Ut_nzval[idx] * inv_dk
                if val != 0.0
                    a[ptr] = val
                    indr[ptr] = q[c]
                    ptr += 1
                end
            end
        end
        lenr[orig_row] = ptr - locr[orig_row]
    end
    lenU = ptr - 1

    # 2. Pack strictly lower triangular part of L_tilde backwards at the end of a
    L_colptr = SparseArrays.getcolptr(L_umf)
    L_rowval = rowvals(L_umf)
    L_nzval = nonzeros(L_umf)

    l1 = lena + 1
    for k in 1:m
        dk = d[k]
        orig_piv = p[k]
        len_col = 0
        for idx in L_colptr[k]:(L_colptr[k+1]-1)
            r = L_rowval[idx]
            if r > k
                val = L_nzval[idx] * (dk / d[r])
                if val != 0.0
                    l1 -= 1
                    a[l1] = -val
                    indc[l1] = p[r]
                    indr[l1] = orig_piv
                    len_col += 1
                end
            end
        end
        lenc[k] = len_col
    end
    lenL0 = lena + 1 - l1
    nelem = lenU + lenL0

    # Set LUSOL internal bookkeeping counters in luparm
    setindex!(parm.luparm, 0, 10) # inform
    setindex!(parm.luparm, m, 16) # nrank
    setindex!(parm.luparm, m, 20) # numL0
    setindex!(parm.luparm, lenL0, 21) # lenL0
    setindex!(parm.luparm, lenU, 22) # lenU
    setindex!(parm.luparm, lenL0, 23) # lenL
    setindex!(parm.luparm, lenU, 24) # lrow

    # Assemble final LUSOLobjA, LUSOLres, and LUSOLFactorization objects
    objA = LUSOLobjA(m, n, nelem, lena, a, indc, indr)
    res = crtLUSOLres(objA)
    copyto!(res.p, p)
    copyto!(res.q, q)
    copyto!(res.lenc, lenc)
    copyto!(res.lenr, lenr)
    copyto!(res.locc, locc)
    copyto!(res.locr, locr)


    return LUSOLFactorization(parm, objA, res)
end
"""
    cnvrtCHOLMOD2LUSOL(F::SparseArrays.CHOLMOD.Factor; kwargs...) -> LUSOLFactorization

Internal helper that converts a standard library `CHOLMOD.Factor` (`cholesky` or `ldlt`)
directly into a `LUSOLFactorization` in O(nnz) time without re-factorizing `A`.
"""
function cnvrtCHOLMOD2LUSOL(F::SparseArrays.CHOLMOD.Factor; kwargs...)
    m, n = size(F)
    m == n || throw(DimensionMismatch("Expected square CHOLMOD factorization, got ($m × $n)."))

    p = Vector{Int}(F.p)
    q = copy(p)

    # Check whether F is an LDL' factorization (supports F.LD) or LL' Cholesky (uses F.L)
    is_ldlt = true
    local LD::SparseMatrixCSC{Float64,Int}
    try
        LD = sparse(F.LD)
    catch
        is_ldlt = false
        LD = sparse(F.L)
    end

    LD_colptr = SparseArrays.getcolptr(LD)
    LD_rowval = rowvals(LD)
    LD_nzval = nonzeros(LD)

    nelem_est = 2 * nnz(LD)
    parm = crtLUSOLparm(; kwargs...)
    lena_factor = parm.lena_factor
    min_lena = getindex(parm.luparm, 13)
    lena = max(ceil(Int, lena_factor * nelem_est), 2 * max(m, n), min_lena)

    # Allocate storage arrays for LUSOLobjA
    a = zeros(Float64, lena)
    indc = zeros(Int, lena)
    indr = zeros(Int, lena)

    # Allocate index/pointer arrays for LUSOLres
    lenc = zeros(Int, n)
    lenr = zeros(Int, m)
    locc = zeros(Int, n)
    locr = zeros(Int, m)

    # 1. Pack U_tilde row-by-row at the beginning of a
    #    For LDL': U_tilde[k, k] = d_k, U_tilde[k, r] = d_k * L[r, k]
    #    For LL':  L_tilde[r, k] = L[r, k] / L[k, k], d_k = L[k, k]^2 => U_tilde[k, r] = L[k, k] * L[r, k]
    ptr = 1
    for k in 1:m
        orig_row = p[k]
        locr[orig_row] = ptr

        diag_raw = 0.0
        for idx in LD_colptr[k]:(LD_colptr[k+1]-1)
            if LD_rowval[idx] == k
                diag_raw = LD_nzval[idx]
                break
            end
        end

        dk = is_ldlt ? diag_raw : diag_raw^2
        a[ptr] = dk
        indr[ptr] = q[k]
        ptr += 1

        scale_u = is_ldlt ? diag_raw : diag_raw
        for idx in LD_colptr[k]:(LD_colptr[k+1]-1)
            r = LD_rowval[idx]
            if r > k
                val = scale_u * LD_nzval[idx]
                if val != 0.0
                    a[ptr] = val
                    indr[ptr] = q[r]
                    ptr += 1
                end
            end
        end
        lenr[orig_row] = ptr - locr[orig_row]
    end
    lenU = ptr - 1

    # 2. Pack strictly lower triangular part of unit-diagonal L_tilde backwards at the end of a
    l1 = lena + 1
    for k in 1:m
        orig_piv = p[k]
        diag_raw = 1.0
        if !is_ldlt
            for idx in LD_colptr[k]:(LD_colptr[k+1]-1)
                if LD_rowval[idx] == k
                    diag_raw = LD_nzval[idx]
                    break
                end
            end
        end

        len_col = 0
        for idx in LD_colptr[k]:(LD_colptr[k+1]-1)
            r = LD_rowval[idx]
            if r > k
                val = is_ldlt ? LD_nzval[idx] : (LD_nzval[idx] / diag_raw)
                if val != 0.0
                    l1 -= 1
                    a[l1] = -val
                    indc[l1] = p[r]
                    indr[l1] = orig_piv
                    len_col += 1
                end
            end
        end
        lenc[k] = len_col
    end
    lenL0 = lena + 1 - l1
    nelem = lenU + lenL0

    # Set LUSOL internal bookkeeping counters in luparm
    setindex!(parm.luparm, 0, 10) # inform
    setindex!(parm.luparm, m, 16) # nrank
    setindex!(parm.luparm, m, 20) # numL0
    setindex!(parm.luparm, lenL0, 21) # lenL0
    setindex!(parm.luparm, lenU, 22) # lenU
    setindex!(parm.luparm, lenL0, 23) # lenL
    setindex!(parm.luparm, lenU, 24) # lrow

    # Assemble final LUSOLobjA, LUSOLres, and LUSOLFactorization objects
    objA = LUSOLobjA(m, n, nelem, lena, a, indc, indr)
    res = crtLUSOLres(objA)
    copyto!(res.p, p)
    copyto!(res.q, q)
    copyto!(res.lenc, lenc)
    copyto!(res.lenr, lenr)
    copyto!(res.locc, locc)
    copyto!(res.locr, locr)

    return LUSOLFactorization(parm, objA, res)
end

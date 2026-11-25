"""
    cnvrtKLU2LUSOL(F; kwargs...) -> LUSOLFactorization

Converts a KLU factorization object (from `KLU.jl` or `PureKLU.jl`) into a
`LUSOLFactorization`. Methods are provided by package extensions when `KLU` or `PureKLU` is loaded.
"""
function cnvrtKLU2LUSOL end

"""
    _pack_factors_to_lusol(L_unit, U_mat, p, q; kwargs...) -> LUSOLFactorization

Core internal builder that packs a unit-diagonal lower triangular matrix `L_unit` and an
upper triangular matrix `U_mat` (matching `A[p, q] = L_unit * U_mat`) into `LUSOLFactorization`.
"""
function _pack_factors_to_lusol(L_unit::SparseMatrixCSC{Float64,Int},
    U_mat::SparseMatrixCSC{Float64,Int},
    p::Vector{Int}, q::Vector{Int}; kwargs...)
    m, n = size(U_mat)
    m == n || throw(DimensionMismatch("Expected square factorization, got ($m × $n)."))

    parm = crtLUSOLparm(; kwargs...)
    nelem_est = nnz(L_unit) + nnz(U_mat)
    lena_factor = parm.lena_factor
    min_lena = getindex(parm.luparm, 13)
    lena = max(ceil(Int, lena_factor * nelem_est), 2 * max(m, n), min_lena)

    a = zeros(Float64, lena)
    indc = zeros(Int, lena)
    indr = zeros(Int, lena)

    lenc = zeros(Int, n)
    lenr = zeros(Int, m)
    locc = zeros(Int, n)
    locr = zeros(Int, m)

    # 1. Pack U_mat row-by-row at the beginning of a (diagonal entry first per row)
    Ut = sparse(transpose(U_mat))
    Ut_colptr = SparseArrays.getcolptr(Ut)
    Ut_rowval = rowvals(Ut)
    Ut_nzval = nonzeros(Ut)

    ptr = 1
    for k in 1:m
        orig_row = p[k]
        locr[orig_row] = ptr

        diag_val = 0.0
        for idx in Ut_colptr[k]:(Ut_colptr[k+1]-1)
            if Ut_rowval[idx] == k
                diag_val = Ut_nzval[idx]
                break
            end
        end
        if diag_val != 0.0
            a[ptr] = diag_val
            indr[ptr] = q[k]
            ptr += 1
        end

        for idx in Ut_colptr[k]:(Ut_colptr[k+1]-1)
            c = Ut_rowval[idx]
            if c != k
                val = Ut_nzval[idx]
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

    # 2. Pack strictly lower triangular part of L_unit backwards at the end of a
    L_colptr = SparseArrays.getcolptr(L_unit)
    L_rowval = rowvals(L_unit)
    L_nzval = nonzeros(L_unit)

    l1 = lena + 1
    for k in 1:m
        orig_piv = p[k]
        len_col = 0
        for idx in L_colptr[k]:(L_colptr[k+1]-1)
            r = L_rowval[idx]
            if r > k
                val = L_nzval[idx]
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

    nz_lenc = filter(>(0), lenc)
    numL0 = length(nz_lenc)
    fill!(lenc, 0)
    copyto!(lenc, 1, nz_lenc, 1, numL0)
    nrank = count(>(0), lenr)

    # Set LUSOL internal bookkeeping counters in luparm
    setindex!(parm.luparm, 0, 10) # inform
    setindex!(parm.luparm, nrank, 16) # nrank
    setindex!(parm.luparm, numL0, 20) # numL0
    setindex!(parm.luparm, lenL0, 21) # lenL0
    setindex!(parm.luparm, lenU, 22) # lenU
    setindex!(parm.luparm, lenL0, 23) # lenL
    setindex!(parm.luparm, lenU, 24) # lrow
    setindex!(parm.luparm, lenU, 25) # lcol

    # Assemble final LUSOLobjA, LUSOLres, and LUSOLFactorization objects
    objA = LUSOLobjA(m, n, nelem, lena, a, indc, indr)
    res = crtLUSOLres(objA)
    copyto!(res.p, p)
    copyto!(res.q, q)
    copyto!(res.lenc, lenc)
    copyto!(res.lenr, lenr)
    copyto!(res.locc, locc)
    copyto!(res.locr, locr)

    for k in 1:m
        res.ipinv[p[k]] = k
        res.iploc[k] = k
    end
    for k in 1:n
        res.iqinv[q[k]] = k
        res.iqloc[k] = k
    end

    return LUSOLFactorization(parm, objA, res)
end

"""
    cnvrtUMFPACK2LUSOL(F::SparseArrays.UMFPACK.UmfpackLU; kwargs...) -> LUSOLFactorization

Converts a standard library `UmfpackLU` object into a `LUSOLFactorization` without re-factorizing `A`.
"""
function cnvrtUMFPACK2LUSOL(F::SparseArrays.UMFPACK.UmfpackLU; kwargs...)
    p = Vector{Int}(F.p)
    q = Vector{Int}(F.q)
    d = Vector{Float64}(F.Rs)[p]

    # Absorb row scaling D = diag(Rs[p]): A[p, q] = (D^-1 * L * D) * (D^-1 * U)
    L_unit = dropzeros!(spdiagm(0 => 1.0 ./ d) * F.L * spdiagm(0 => d))
    U_mat = dropzeros!(spdiagm(0 => 1.0 ./ d) * F.U)

    return _pack_factors_to_lusol(L_unit, U_mat, p, q; kwargs...)
end

"""
    cnvrtCHOLMOD2LUSOL(F::SparseArrays.CHOLMOD.Factor; kwargs...) -> LUSOLFactorization

Converts a standard library `CHOLMOD.Factor` (`cholesky` or `ldlt`) into a `LUSOLFactorization`.
"""
function cnvrtCHOLMOD2LUSOL(F::SparseArrays.CHOLMOD.Factor; kwargs...)
    p = Vector{Int}(F.p)
    q = copy(p)

    local L_unit::SparseMatrixCSC{Float64,Int}
    local U_mat::SparseMatrixCSC{Float64,Int}

    try
        LD = sparse(F.LD)
        d = diag(LD)
        L_unit = dropzeros!(LD - spdiagm(0 => d) + I)
        U_mat = dropzeros!(spdiagm(0 => d) * sparse(transpose(L_unit)))
    catch
        L_chol = sparse(F.L)
        d_sqrt = diag(L_chol)
        L_unit = dropzeros!(L_chol * spdiagm(0 => 1.0 ./ d_sqrt))
        U_mat = dropzeros!(spdiagm(0 => d_sqrt .^ 2) * sparse(transpose(L_unit)))
    end

    return _pack_factors_to_lusol(L_unit, U_mat, p, q; kwargs...)
end

"""
    _pack_klu_to_lusol(L_klu, U_klu, F_off, p, q, Rs; invert_rs=true, kwargs...) -> LUSOLFactorization

Shared packing logic used by both `LUSOLKLUExt` and `LUSOLPureKLUExt`.
"""
function _pack_klu_to_lusol(L_klu::SparseMatrixCSC{Float64,Int},
    U_klu::SparseMatrixCSC{Float64,Int},
    F_off::SparseMatrixCSC{Float64,Int},
    p::Vector{Int}, q::Vector{Int},
    Rs::Vector{Float64};
    invert_rs::Bool=true, kwargs...)
    d = invert_rs ? (1.0 ./ Rs) : Rs
    U_total = nnz(F_off) == 0 ? U_klu : dropzeros!(U_klu + sparse(LowerTriangular(L_klu) \ Matrix(F_off)))

    L_unit = dropzeros!(spdiagm(0 => 1.0 ./ d) * L_klu * spdiagm(0 => d))
    U_mat = dropzeros!(spdiagm(0 => 1.0 ./ d) * U_total)

    return _pack_factors_to_lusol(L_unit, U_mat, p, q; kwargs...)
end

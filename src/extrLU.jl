"""
    extrLU(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres) -> Tuple{SparseMatrixCSC{Float64, Int}, SparseMatrixCSC{Float64, Int}}

Unpacks the internal LUSOL storage arrays into explicit sparse matrices `L` (m × m)
and `U` (m × n) such that `(L * U)[invperm(res.p), invperm(res.q)] == A`.
Works both after `lu1fac!` and after rank-1 column updates (`lu8rpc!`).
"""
function extrLU(parm::LUSOLparm, objA::LUSOLobjA, res::LUSOLres)
    m, n = objA.m, objA.n
    lena = objA.lena
    numL0 = getindex(parm.luparm, 20)
    lenL0 = getindex(parm.luparm, 21)
    lenU = getindex(parm.luparm, 22)
    lenL = getindex(parm.luparm, 23)

    qinv = invperm(res.q)

    # 1. Extract U (indexed by original row i = res.p[k])
    u_rows = Vector{Int}()
    u_cols = Vector{Int}()
    u_vals = Vector{Float64}()
    sizehint!(u_rows, lenU)
    sizehint!(u_cols, lenU)
    sizehint!(u_vals, lenU)

    for k in 1:m
        orig_row = getindex(res.p, k)
        len_row = getindex(res.lenr, orig_row)
        if len_row > 0
            start_idx = getindex(res.locr, orig_row)
            for idx in start_idx:(start_idx+len_row-1)
                orig_col = getindex(objA.indr, idx)
                perm_col = getindex(qinv, orig_col)
                push!(u_rows, k)
                push!(u_cols, perm_col)
                push!(u_vals, getindex(objA.a, idx))
            end
        end
    end
    U = dropzeros!(sparse(u_rows, u_cols, u_vals, m, n))

    # 2. Extract L     
    pinv = invperm(res.p)
    numL = lenL - lenL0

    if numL == 0
        # Fast path (O(nnz) sparse, no matrix inversion needed after lu1fac!)
        l_rows = collect(1:m)
        l_cols = collect(1:m)
        l_vals = ones(Float64, m)
        sizehint!(l_rows, m + lenL0)
        sizehint!(l_cols, m + lenL0)
        sizehint!(l_vals, m + lenL0)

        l1 = lena + 1
        for k in 1:numL0
            len = getindex(res.lenc, k)
            l = l1
            l1 -= len
            ipiv = getindex(objA.indr, l1)
            perm_col = getindex(pinv, ipiv)
            for _ in 1:len
                l -= 1
                j = getindex(objA.indc, l)
                perm_row = getindex(pinv, j)
                push!(l_rows, perm_row)
                push!(l_cols, perm_col)
                push!(l_vals, -getindex(objA.a, l))
            end
        end
        L = dropzeros!(sparse(l_rows, l_cols, l_vals, m, m))
        return L, U
    else
        # Update path (after lu8rpc!): apply inverse elementary steps in reverse order
        # Working on L_orig' (CSC) makes row operations fast column operations!
        Lt = sparse(1.0I, m, m)

        # 1. Reverse of Stage B (lu8rpc updates)
        for l in (lena-lenL+1):(lena-lenL0)
            i = getindex(objA.indr, l)
            j = getindex(objA.indc, l)
            mult = getindex(objA.a, l)
            if mult != 0.0
                Lt[:, j] .-= mult .* Lt[:, i]
            end
        end

        # 2. Reverse of Stage A (initial lu1fac columns)
        if numL0 > 0
            block_starts = Vector{Int}(undef, numL0)
            l1 = lena + 1
            for k in 1:numL0
                len = getindex(res.lenc, k)
                l1 -= len
                block_starts[k] = l1
            end

            for k in numL0:-1:1
                len = getindex(res.lenc, k)
                l1 = block_starts[k]
                ipiv = getindex(objA.indr, l1)
                for l in l1:(l1+len-1)
                    j = getindex(objA.indc, l)
                    mult = getindex(objA.a, l)
                    if mult != 0.0
                        Lt[:, j] .-= mult .* Lt[:, ipiv]
                    end
                end
            end
        end

        L_perm = sparse(Lt')[res.p, res.p]
        droptol!(L_perm, 1e-13)
        return L_perm, U
    end

end
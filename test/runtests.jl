using LUSOL
using Test
using SparseArrays
using LinearAlgebra
using Random

@testset "LUSOL.jl" begin

    @testset "Initialization (LUSOL.crtLUSOLparm, LUSOL.crtLUSOLobjA, LUSOL.crtLUSOLres)" begin
        parm = LUSOL.crtLUSOLparm()
        @test getindex(parm.luparm, 2) == 0        # Default print_level = 0
        @test getindex(parm.luparm, 6) == 0        # Default pivoting = :TPP
        @test parm.lena_factor == 2.0

        parm_trp = LUSOL.crtLUSOLparm(pivoting=:TRP, print_level=1)
        @test getindex(parm_trp.luparm, 2) == 1
        @test getindex(parm_trp.luparm, 6) == 1


        @test_throws ArgumentError LUSOL.crtLUSOLparm(pivoting=:INVALID)

        A = sparse([1.0 2.0; 3.0 4.0; 5.0 6.0]) # Rectangular 3x2 matrix
        objA = LUSOL.crtLUSOLobjA(A, parm)
        @test objA.m == 3
        @test objA.n == 2
        @test objA.nelem == 6

        res = LUSOL.crtLUSOLres(objA)
        @test length(res.p) == 3
        @test length(res.q) == 2
        @test res.inform == 0
    end

    @testset "Factorization (LUSOL.lu1fac & LUSOL.lu1fac!) and Extraction (LUSOL.extrLU)" begin
        Random.seed!(42)

        # 1. Square nonsingular sparse matrix
        A_sq = sprand(50, 50, 0.15) + 5I
        parm, objA, res = LUSOL.lu1fac(A_sq)
        @test res.inform == 0

        L, U = LUSOL.extrLU(parm, objA, res)
        @test istril(L)
        @test istriu(U)
        @test size(L) == (50, 50)
        @test size(U) == (50, 50)

        A_rec = (L*U)[invperm(res.p), invperm(res.q)]
        @test norm(A_rec - A_sq) < 1e-12

        # 2. Rectangular overdetermined matrix (m > n)
        A_tall = sprand(30, 15, 0.25)
        parm_t, objA_t, res_t = LUSOL.lu1fac(A_tall)
        @test res_t.inform in (0, 1)

        L_t, U_t = LUSOL.extrLU(parm_t, objA_t, res_t)
        @test istril(L_t)
        @test istriu(U_t)
        @test size(L_t) == (30, 30)
        @test size(U_t) == (30, 15)
        @test norm((L_t*U_t)[invperm(res_t.p), invperm(res_t.q)] - A_tall) < 1e-12

        # 3. Rectangular underdetermined matrix (m < n)
        A_wide = sprand(15, 30, 0.25)
        parm_w, objA_w, res_w = LUSOL.lu1fac(A_wide)
        @test res_w.inform in (0, 1)

        L_w, U_w = LUSOL.extrLU(parm_w, objA_w, res_w)
        @test istril(L_w)
        @test istriu(U_w)
        @test size(L_w) == (15, 15)
        @test size(U_w) == (15, 30)
        @test norm((L_w*U_w)[invperm(res_w.p), invperm(res_w.q)] - A_wide) < 1e-12

        # 4. Automatic memory reallocation test (tiny initial lena_factor)
        A_dense = sparse(rand(40, 40))
        parm_small = LUSOL.crtLUSOLparm(lena_factor=0.01)
        objA_small = LUSOL.crtLUSOLobjA(A_dense, parm_small)
        objA_small.lena = 50 # Force immediate inform == 7 reallocation
        objA_small.a = zeros(Float64, 50)
        objA_small.indc = zeros(Int, 50)
        objA_small.indr = zeros(Int, 50)
        rows, cols, vals = findnz(A_dense)
        # Only works if nelem <= 50, so test on a smaller 6x6 dense matrix with lena = 10
        A_6 = sparse(rand(6, 6))
        objA_6 = LUSOL.crtLUSOLobjA(A_6)
        objA_6.lena = 36
        resize!(objA_6.a, 36)
        resize!(objA_6.indc, 36)
        resize!(objA_6.indr, 36)
        res_6 = LUSOL.crtLUSOLres(objA_6)
        LUSOL.lu1fac!(parm, objA_6, res_6)
        @test objA_6.lena > 36
        @test res_6.inform == 0
    end

    @testset "Linear System Solve (LUSOL.lu6sol & LUSOL.lu6sol!)" begin
        Random.seed!(123)
        A = sprand(40, 40, 0.15) + 4I
        x_true = rand(40)
        b = A * x_true
        b_trans = A' * x_true

        parm, objA, res = LUSOL.lu1fac(A)

        # Solve A * x = b (mode = 5)
        x_sol = LUSOL.lu6sol(parm, objA, res, b; mode=5)
        @test isapprox(x_sol, x_true, atol=1e-12)

        # Solve A' * x = b_trans (mode = 6)
        x_sol_t = LUSOL.lu6sol(parm, objA, res, b_trans; mode=6)
        @test isapprox(x_sol_t, x_true, atol=1e-12)

        # Dimension mismatch checks
        @test_throws DimensionMismatch LUSOL.lu6sol(parm, objA, res, rand(39); mode=5)
        @test_throws DimensionMismatch LUSOL.lu6sol(parm, objA, res, rand(41); mode=6)
    end

    @testset "Column Replacement (LUSOL.lu8rpc!)" begin
        Random.seed!(99)
        A = sprand(25, 25, 0.2) + 4I
        parm, objA, res = LUSOL.lu1fac(A)

        # Replace column 7 with a new random sparse/dense column vector
        jrep = 7
        anew = rand(25)
        diag_val, vnorm_val = LUSOL.lu8rpc!(parm, objA, res, jrep, anew)

        @test res.inform == 0
        @test vnorm_val > 0

        # Build the expected modified matrix in Julia
        A_mod = Matrix(A)
        A_mod[:, jrep] .= anew

        # 1. Verify solution of updated system A_mod * x = b
        x_true = rand(25)
        b = A_mod * x_true
        x_sol = LUSOL.lu6sol(parm, objA, res, b)
        @test isapprox(x_sol, x_true, atol=1e-11)

        # 2. Verify unpacked L and U match A_mod
        L, U = LUSOL.extrLU(parm, objA, res)
        @test isapprox((L*U)[invperm(res.p), invperm(res.q)], A_mod, atol=1e-11)
    end

    @testset "Public API (LUSOL.lu)" begin
        A = sprand(30, 30, 0.2) + 5I
        b = rand(30)

        F = LUSOL.lu(A)
        @test issuccess(F)
        @test F.L * F.U ≈ (F.Rs .* A)[F.p, F.q]
        @test F \ b ≈ A \ b

        L, U, p, q, Rs = F
        @test L == F.L && U == F.U && p == F.p && q == F.q && Rs == F.Rs
    end

    @testset "Public API (LUSOL.singlecolumnupdate & LUSOL.singlecolumnupdate!)" begin
        Random.seed!(77)
        A = sprand(30, 30, 0.2) + 5I
        b = rand(30)
        F = LUSOL.lu(A)

        jrep = 5
        anew = rand(30)
        A_mod = Matrix(A)
        A_mod[:, jrep] .= anew

        F_copy = LUSOL.singlecolumnupdate(F, jrep, anew)
        @test F \ b ≈ A \ b
        @test F_copy \ b ≈ A_mod \ b
        @test F_copy.L * F_copy.U ≈ (F_copy.Rs .* A_mod)[F_copy.p, F_copy.q]

        LUSOL.singlecolumnupdate!(F, jrep, anew)
        @test F \ b ≈ A_mod \ b
        @test F.L * F.U ≈ (F.Rs .* A_mod)[F.p, F.q]
    end

    @testset "Public API (LUSOL.lowrankupdate & LUSOL.lowrankupdate!)" begin
        Random.seed!(88)
        A = sprand(25, 25, 0.2) + 6I
        b = rand(25)
        F = LUSOL.lu(A)

        c = sprand(25, 0.3)
        A_mod = Matrix(A) + Vector(c) * Vector(c)'

        F_copy = LUSOL.lowrankupdate(F, c)
        @test F \ b ≈ A \ b
        @test F_copy \ b ≈ A_mod \ b
        @test F_copy.L * F_copy.U ≈ (F_copy.Rs .* A_mod)[F_copy.p, F_copy.q]

        LUSOL.lowrankupdate!(F, c)
        @test F \ b ≈ A_mod \ b
        @test F.L * F.U ≈ (F.Rs .* A_mod)[F.p, F.q]
    end

    @testset "Converter (convertUMFPACK2LUSOL)" begin
        Random.seed!(102)
        A = sprand(25, 25, 0.2) + 5I
        b = rand(25)

        F_umf = LinearAlgebra.lu(A)
        F_lusol = LUSOL.cnvrtUMFPACK2LUSOL(F_umf)

        @test F_lusol isa LUSOL.LUSOLFactorization
        @test issuccess(F_lusol)
        @test F_lusol.p == F_umf.p
        @test F_lusol.q == F_umf.q
        @test F_lusol \ b ≈ A \ b
        @test F_lusol.L * F_lusol.U ≈ (F_lusol.Rs .* A)[F_lusol.p, F_lusol.q]
    end

    @testset "Converter (cnvrtCHOLMOD2LUSOL)" begin
        Random.seed!(103)
        M = sprand(25, 25, 0.2)
        A_spd = M' * M + 5I   # Symmetric positive definite
        b = rand(25)

        # 1. From sparse cholesky(A_spd)
        F_chol = LinearAlgebra.cholesky(A_spd)
        F_lusol_chol = LUSOL.cnvrtCHOLMOD2LUSOL(F_chol)

        @test F_lusol_chol isa LUSOL.LUSOLFactorization
        @test issuccess(F_lusol_chol)
        @test F_lusol_chol \ b ≈ A_spd \ b
        @test F_lusol_chol.L * F_lusol_chol.U ≈ (F_lusol_chol.Rs .* A_spd)[F_lusol_chol.p, F_lusol_chol.q]

        # 2. From sparse ldlt(A_spd)
        F_ldlt = LinearAlgebra.ldlt(A_spd)
        F_lusol_ldlt = LUSOL.cnvrtCHOLMOD2LUSOL(F_ldlt)

        @test F_lusol_ldlt isa LUSOL.LUSOLFactorization
        @test issuccess(F_lusol_ldlt)
        @test F_lusol_ldlt \ b ≈ A_spd \ b
        @test F_lusol_ldlt.L * F_lusol_ldlt.U ≈ (F_lusol_ldlt.Rs .* A_spd)[F_lusol_ldlt.p, F_lusol_ldlt.q]
    end
    return nothing
end

@testset "Extension (LUSOLKLUExt - KLU.jl)" begin
    using KLU
    Random.seed!(201)

    # Create a sparse matrix with multiple BTF blocks (tridiagonal + off-diagonal coupling)
    A = sprand(25, 25, 0.15) + 5I
    b = rand(25)

    F_klu = KLU.klu(A)
    F_lusol = LUSOL.cnvrtKLU2LUSOL(F_klu)

    @test F_lusol isa LUSOL.LUSOLFactorization
    @test issuccess(F_lusol)
    @test F_lusol \ b ≈ A \ b
    @test F_lusol.L * F_lusol.U ≈ (F_lusol.Rs .* A)[F_lusol.p, F_lusol.q]

    # Test direct singlecolumnupdate & lowrankupdate on KLUFactorization
    jrep = 6
    anew = rand(25)
    A_col = Matrix(A)
    A_col[:, jrep] .= anew
    F_col = LUSOL.singlecolumnupdate(F_klu, jrep, anew)
    @test F_col \ b ≈ A_col \ b

    c = sprand(25, 0.25)
    A_lr = Matrix(A) + Vector(c) * Vector(c)'
    F_lr = LUSOL.lowrankupdate(F_klu, c)
    @test F_lr \ b ≈ A_lr \ b
end

@testset "Extension (LUSOLPureKLUExt - PureKLU.jl)" begin
    using PureKLU
    Random.seed!(202)

    A = sprand(25, 25, 0.15) + 5I
    b = rand(25)

    F_pklu = PureKLU.klu(A)
    F_lusol = LUSOL.cnvrtKLU2LUSOL(F_pklu)

    @test F_lusol isa LUSOL.LUSOLFactorization
    @test issuccess(F_lusol)
    @test F_lusol \ b ≈ A \ b
    @test F_lusol.L * F_lusol.U ≈ (F_lusol.Rs .* A)[F_lusol.p, F_lusol.q]

    # Test direct singlecolumnupdate & lowrankupdate on PureKLU.KLUFactorization
    jrep = 6
    anew = rand(25)
    A_col = Matrix(A)
    A_col[:, jrep] .= anew
    F_col = LUSOL.singlecolumnupdate(F_pklu, jrep, anew)
    @test F_col \ b ≈ A_col \ b

    c = sprand(25, 0.25)
    A_lr = Matrix(A) + Vector(c) * Vector(c)'
    F_lr = LUSOL.lowrankupdate(F_pklu, c)
    @test F_lr \ b ≈ A_lr \ b
end

@testset "LinearAlgebra ldiv! & LUSOL.lu(F) on all factorization types" begin
    Random.seed!(404)
    M = sprand(25, 25, 0.2)
    A = M' * M + 5I # SPD so cholesky/ldlt also work on the same matrix
    b = rand(25)
    B = rand(25, 3)

    F_lusol = LUSOL.lu(A)

    # 1. LinearAlgebra.ldiv! interfaces
    b_copy = copy(b)
    @test LinearAlgebra.ldiv!(F_lusol, b_copy) === b_copy
    @test b_copy ≈ A \ b

    x_dest = zeros(25)
    @test LinearAlgebra.ldiv!(x_dest, F_lusol, b) === x_dest
    @test x_dest ≈ A \ b

    X_dest = zeros(25, 3)
    @test LinearAlgebra.ldiv!(X_dest, F_lusol, B) === X_dest
    @test X_dest ≈ A \ B

    # 2. LUSOL.lu(F) & LinearAlgebra.lu(F) on LUSOLFactorization (deep copy check)
    F_copy = LUSOL.lu(F_lusol)
    @test F_copy isa LUSOL.LUSOLFactorization
    @test F_copy !== F_lusol
    @test F_copy \ b ≈ A \ b

    F_la_copy = LinearAlgebra.lu(F_lusol)
    @test F_la_copy isa LUSOL.LUSOLFactorization
    @test F_la_copy \ b ≈ A \ b

    # 3. LUSOL.lu(F) on stdlib UmfpackLU and CHOLMOD.Factor
    F_umf = LinearAlgebra.lu(A)
    F_from_umf = LUSOL.lu(F_umf)
    @test F_from_umf isa LUSOL.LUSOLFactorization
    @test F_from_umf \ b ≈ A \ b

    F_chol = LinearAlgebra.cholesky(A)
    F_from_chol = LUSOL.lu(F_chol)
    @test F_from_chol isa LUSOL.LUSOLFactorization
    @test F_from_chol \ b ≈ A \ b

    F_ldlt = LinearAlgebra.ldlt(A)
    F_from_ldlt = LUSOL.lu(F_ldlt)
    @test F_from_ldlt isa LUSOL.LUSOLFactorization
    @test F_from_ldlt \ b ≈ A \ b

    # 4. LUSOL.lu(F) on KLU.jl & PureKLU.jl extensions
    using KLU, PureKLU
    F_from_klu = LUSOL.lu(KLU.klu(A))
    @test F_from_klu isa LUSOL.LUSOLFactorization
    @test F_from_klu \ b ≈ A \ b

    F_from_pklu = LUSOL.lu(PureKLU.klu(A))
    @test F_from_pklu isa LUSOL.LUSOLFactorization
    @test F_from_pklu \ b ≈ A \ b
end

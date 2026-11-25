module LUSOL

# using LUSOL_jll # Später wieder aktivieren, sobald Yggdrasil gemerged hat
const liblusol = joinpath(@__DIR__, "..", "bin", "liblusol.dll")
using SparseArrays
using LinearAlgebra

# export LUSOLparm, LUSOLobjA, LUSOLres,
#     crtLUSOLparm, crtLUSOLobjA, crtLUSOLres,
#     lu1fac, lu1fac!,
#     lu6sol, lu6sol!, # lu6mul,
#     lu8rpc, lu8rpc!, # lu8rpr!, lu8adc!, lu8adr!, lu8dlc!, lu8dlr!, lu8mod!,
#     extrLU
export lu, singlecolumnupdate, singlecolumnupdate!, lowrankupdate, lowrankupdate!

# Include initialization helpers
include("crtLUSOLparm.jl")  # Create and initialize matrix-independent LUSOL parameters
include("crtLUSOLobjA.jl")  # Create and initialize matrix-dependent storage from SparseMatrixCSC
include("crtLUSOLres.jl")   # Allocate permutation, pointer, and result arrays
include("cnvrt2LUSOL.jl")   # Convert other structs (e.g. UMFPACK and CHOLMOD) tu LUSOL

# Include individual LUSOL routines
include("lu1fac.jl") # Compute sparse LU factorization (PAQ = LU)
include("lu6sol.jl") # Solve linear systems (Ax = b, A'x = b, Lx = b, Ux = b)
# include("lu6mul.jl") # Compute matrix-vector products (b = Ax, b = A'x)
include("lu8rpc.jl") # Replace a column of A (Bartels-Golub-Reid update)
# include("lu8rpr.jl") # Replace a row of A
# include("lu8adc.jl") # Add a column to A (n -> n + 1)
# include("lu8adr.jl") # Add a row to A (m -> m + 1)
# include("lu8dlc.jl") # Delete a column from A (n -> n - 1)
# include("lu8dlr.jl") # Delete a row from A (m -> m - 1)
# include("lu8mod.jl") # Perform a general rank-1 modification (A -> A + beta * v * w')

# Include postprocessing routines
include("extrLU.jl") # Compute sparse LU matrices

# Public high-level interface
include("lu.jl")
include("singlecolumnupdate.jl")
include("lowrankupdate.jl")

end # module

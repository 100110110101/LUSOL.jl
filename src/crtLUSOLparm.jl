"""
    LUSOLparm

Stores matrix-independent control parameters (`luparm`) and floating-point tolerances
(`parmlu`) as well as the initial storage growth factor for LUSOL.
"""
mutable struct LUSOLparm
    luparm::Vector{Int}
    parmlu::Vector{Float64}
    lena_factor::Float64
end

"""
    crtLUSOLparm(; kwargs...) -> LUSOLparm

Creates and initializes a `LUSOLparm` object with default LUSOL settings.
Any parameter can be overridden via keyword arguments.

# Keyword Arguments
- `nout::Int = 6`: Output unit number (`6` = stdout).
- `print_level::Int = 0`: Print level (`0` = silent, `10` = debug).
- `maxcol::Int = 5`: Max columns searched in Markowitz pivoting.
- `pivoting::Symbol = :TPP`: Pivoting strategy (`:TPP`, `:TRP`, `:TCP`).
- `keepLU::Bool = true`: Retain `L` and `U` factors (`true` = 1, `false` = 0).
- `Ltol1::Float64 = 10.0`: Max multiplier allowed in `L` during factorization.
- `Ltol2::Float64 = 10.0`: Max multiplier allowed in `L` during updates.
- `small::Float64 = 3.0e-13`: Drop tolerance for small elements.
- `Utol1::Float64 = 3.7e-11`: Absolute tolerance for small pivots in `U`.
- `Utol2::Float64 = 3.7e-11`: Relative tolerance for small pivots in `U`.
- `Uspace::Float64 = 3.0`: Factor for memory expansion.
- `dens1::Float64 = 0.3`: Threshold for switching to dense LU.
- `dens2::Float64 = 0.5`: Secondary dense threshold.
- `lena_factor::Float64 = 2.0`: Initial buffer multiplier for `lena`.
"""
function crtLUSOLparm(;
    nout::Int=6,
    print_level::Int=0,
    maxcol::Int=5,
    pivoting::Symbol=:TPP,
    keepLU::Bool=true,
    Ltol1::Float64=10.0,
    Ltol2::Float64=10.0,
    small::Float64=3.0e-13,
    Utol1::Float64=3.7e-11,
    Utol2::Float64=3.7e-11,
    Uspace::Float64=3.0,
    dens1::Float64=0.3,
    dens2::Float64=0.5,
    lena_factor::Float64=2.0
)
    luparm = zeros(Int, 30)
    luparm[1] = nout
    luparm[2] = print_level
    luparm[3] = maxcol
    luparm[6] = if pivoting == :TPP
        0
    elseif pivoting == :TRP
        1
    elseif pivoting == :TCP
        2
    else
        throw(ArgumentError("Unknown pivoting strategy: $pivoting. Use :TPP, :TRP, or :TCP."))
    end
    luparm[8] = keepLU ? 1 : 0

    parmlu = zeros(Float64, 30)
    parmlu[1] = Ltol1
    parmlu[2] = Ltol2
    parmlu[3] = small
    parmlu[4] = Utol1
    parmlu[5] = Utol2
    parmlu[6] = Uspace
    parmlu[7] = dens1
    parmlu[8] = dens2

    return LUSOLparm(luparm, parmlu, lena_factor)
end

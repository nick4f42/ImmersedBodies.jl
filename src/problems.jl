"""
    AbstractScheme

A timestepping scheme. Can be one of:
- [`CNAB`](@ref): Crank-Nicolson/Adams-Bashforth
"""
abstract type AbstractScheme end

"""
    timestep(problem::Problem)
    timestep(scheme::AbstractScheme)

The amount of time between consecutive timesteps.
"""
function timestep end

"""
    CNAB(dt, n=2) :: AbstractScheme

An n-step Crank-Nicolson/Adams-Bashforth scheme.
"""
struct CNAB <: AbstractScheme
    β::Vector{Float64} # adams bashforth coefficients
    dt::Float64
    function CNAB(dt, n=2)
        if n != 2
            throw(DomainError("only 2-step CNAB is currently supported"))
        end
        return new([1.5, -0.5], dt)
    end
end

timestep(scheme::CNAB) = scheme.dt

"""
    AbstractState

The state of fluid and bodies at a point in time.
"""
abstract type AbstractState end

"""
   timevalue(state::AbstractState)

The current time of a `state`.
"""
function timevalue end

"""
    timeindex(state::AbstractState)

The current timestep index of a `state`. At the initial time, the index is `1`.
"""
function timeindex end

"""
    Problem(fluid::AbstractFluid, bodies::BodyGroup)

Specifies the fluid and bodies of a problem.
"""
struct Problem{F<:AbstractFluid,B<:AbstractBody}
    fluid::F
    bodies::BodyGroup{B}
end

function Problem(_, _)
    throw(ArgumentError("there is no default scheme for the given fluid and bodies"))
end

"""
    timestep_scheme(problem::Problem) :: AbstractScheme
    timestep_scheme(fluid::AbstractFluid) :: AbstractScheme

The [`AbstractScheme`](@ref) of a problem or fluid.
"""
timestep_scheme(prob::Problem) = timestep_scheme(prob.fluid)

timestep(prob::Problem) = timestep(timestep_scheme(prob))

"""
    statetype(prob::Problem)

The [`AbstractState`](@ref) subtype used for solving [`Problem`](@ref).
"""
function statetype end

"""
    solvertype(prob::Problem)

The [`AbstractSolver`](@ref) subtype used for solving [`Problem`](@ref).
"""
function solvertype end

"""
    initstate(prob::Problem, t=0) :: AbstractState

Creates an empty state at time `t`.
"""
initstate(prob::Problem, t::Float64=0.0) = statetype(prob)(prob, t)

const STATE_FILE_HEADER = "@ImmersedBodies.jl State v1@"

"""
    save_state(filename, state::State)
    save_state(io, state::State)

Write the contents of a [`State`](@ref) to a file or IO stream. The result can be loaded
with [`load_state`](@ref).
"""
function save_state(filename::AbstractString, state::State)
    open(io -> save_state(io, state), filename, "w")
end

function save_state(io::IO, state::State)
    write(io, STATE_FILE_HEADER)
    write(io, state.index, state.t0)
    write(io, state.q, state.q0, state.Γ, state.ψ)
    for x in state.nonlin
        write(io, x)
    end
    write(io, state.panels.xb, state.panels.ds, state.panels.ub)
    write(io, state.F̃b)
    write(io, state.freestream_vel)
    nothing
end

"""
    load_state(filename, prob::Problem)
    load_state(io, prob::Problem)

Read the contents of a [`State`](@ref) from a file or IO stream. The state can be written
with [`save_state`](@ref).
"""
load_state(io, prob) = load_state!(io, State(prob))

"""
    load_state!(filename, state::State)
    load_state!(io, state::State)

[`load_state`](@ref), but update `state` in place.
"""
function load_state!(filename::AbstractString, state::State)
    open(io -> load_state!(io, state), filename)
end

function load_state!(io::IO, state::State)
    if STATE_FILE_HEADER != String(read(io, lastindex(STATE_FILE_HEADER)))
        error("State file is missing header")
    end

    state.index = read(io, Int)
    state.t0 = read(io, Float64)
    set_time_index!(state, state.index)

    read!(io, state.q)
    read!(io, state.q0)
    read!(io, state.Γ)
    read!(io, state.ψ)
    for x in state.nonlin
        read!(io, x)
    end
    read!(io, state.panels.xb)
    read!(io, state.panels.ds)
    read!(io, state.panels.ub)
    read!(io, state.F̃b)
    state.freestream_vel = read(io, SVector{2,Float64})

    update_vars!(state)

    state
end

"""
    SolutionValue

A function of [`State`](@ref) to be saved.

See also [`ArrayValue`](@ref), [`MultiDomainValue`](@ref).
"""
abstract type SolutionValue end

"""
    ArrayValue(f!, T, dims) :: SolutionValue

A function `f!(y, state)` that sets an array `y` with size `dims`.
"""
struct ArrayValue{F,T,N} <: SolutionValue
    f!::F
    type::Type{T}
    dims::Dims{N}
end

const Coords = NTuple{2,LinRange{Float64,Int}}

"""
    MultiDomainValue(f!, coords::AbstractVector)

A function `f!(y, state)` that sets a 3D array `y` with corresponding coordinates `coords`.
`array[i,j,level]` should be at coordinate `(x[i], y[j])` where `(x, y) = coords[level]`.

See also [`coords`](@ref).
"""
struct MultiDomainValue{F} <: SolutionValue
    f!::F
    coords::Vector{Coords}
    dims::Dims{3}
    function MultiDomainValue(f!::F, coords::AbstractVector) where {F}
        nlevel = length(coords)
        dims = (length.(coords[1])..., nlevel)
        new{F}(f!, convert(Vector{Coords}, coords), dims)
    end
end

"""
    Timesteps

Specifies timesteps to save the solution at.
"""
abstract type Timesteps end

"""
    TimestepRange(; start, step, [length]) :: Timesteps

Specifies every `step`th timestep starting at `start`. If specified, only include `length`
timesteps.
"""
@kwdef struct TimestepRange{L<:Union{Int,Nothing}} <: Timesteps
    start::Int = 1
    step::Int
    length::L = nothing
end

# Whether a timestep index is in the range, and the index in the range
function _timestep_in(r::TimestepRange, index::Int)
    steps, rem = divrem(index - r.start, r.step)
    i = steps + 1
    (; inside=rem == 0 && i ≥ 1 && (isnothing(r.length) || i ≤ r.length), index=i)
end

# Number of timesteps in r that are at or before index
function _timestep_count(r::TimestepRange, index::Int)
    c = max(0, (r.step + index - r.start) ÷ r.step)
    isnothing(r.length) ? c : min(r.length, c)
end

"""
    SolutionValues(times::Timesteps, vals)
    SolutionValues(times::Timesteps, vals...)

Specifies to save certain values at each time in `times`. `vals` is a collection of `name
=> value` mapping a name string to a [`SolutionValue`](@ref).
"""
struct SolutionValues{T<:Timesteps}
    times::T
    vals::Vector{Pair{String,Any}}  # Pair{String,<:SolutionValue}
end
function SolutionValues(times, vals::Pair...)
    SolutionValues(times, collect(Pair{String,Any}, vals))
end

"""
    SolutionSaver

A backend for saving [`SolutionValue`](@ref)s at certain [`Timesteps`](@ref).
"""
abstract type SolutionSaver end

"""
    SaveNothing() :: SolutionSaver

Do not save anything.
"""
struct SaveNothing <: SolutionSaver end
struct NothingSaver end
_init(::SaveNothing, _) = NothingSaver()
_update(::NothingSaver, _) = nothing
_finalize(::NothingSaver) = nothing

include("hdf5.jl")

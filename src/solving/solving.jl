include("mem-savers.jl")
include("disk-savers.jl")

"""
    AbstractSolver

Used for advancing a state in time.
"""
abstract type AbstractSolver end

"""
    advance!(state::AbstractState, solver::AbstractSolver)

Update `state` and `solver` to the next timestep.
"""
function advance! end

"""
    Timesteps

Identifies certain timesteps of a simulation using [`solve`](@ref).
"""
abstract type Timesteps end

"""
    timestep_count(problem::Problem, (t0, tf))

The amount of timesteps needed to solve `problem` between times `t0` and `tf`.
"""
function timestep_count(prob::Problem, (t0, tf)::NTuple{2,Float64})
    dt = timestep(prob)
    return length(t0:dt:tf)
end

timestep_times(t::AbstractVector) = TimestepTimes(t)
timestep_times(; start=0.0, step) = TimestepTimeRange(start, step)
timestep_indices(i::AbstractVector) = TimestepIndices(i)
timestep_indices(; start=1, step) = TimestepIndexRange(start, step)

"""
    AllTimesteps() :: Timesteps

Specifies every timestep in the simulation.
"""
struct AllTimesteps <: Timesteps end

function callcounter(::AllTimesteps, ::Problem, tspan)
    return _ -> 1
end

function max_timestep_count(::AllTimesteps, prob::Problem, tspan)
    return timestep_count(prob, tspan)
end

"""
    TimestepTimes(t::AbstractVector{Float64}) :: Timesteps

Specifies the timesteps nearest to each time in `t`. `t` must be sorted.
"""
struct TimestepTimes{V<:AbstractVector} <: Timesteps
    t::V
end

function callcounter(times::TimestepTimes, prob::Problem, tspan)
    t0, tf = tspan
    times = Iterators.Stateful(times.t)
    dt = timestep(prob)

    while peek(times) < t0
        popfirst!(times)
    end

    return function (state::AbstractState)
        # Include times up to halfway until the next timestep
        tmax = timevalue(state) + dt / 2

        count = 0
        while !isempty(times) && peek(times) < tmax
            popfirst!(times)
            count += 1
        end

        return count
    end
end

function max_timestep_count(times::TimestepTimes, ::Problem, tspan)
    return length(times.t)
end

"""
    TimestepTimeRange(start, step) :: Timesteps
    TimestepTimeRange(; start=0.0, step) :: Timesteps

An infinite range of timesteps starting at `start` counting by a positive `step`.
"""
struct TimestepTimeRange <: Timesteps
    start::Float64
    step::Float64
    function TimestepTimeRange(start::Float64, step::Float64)
        @assert step > 0
        return new(start, step)
    end
end

TimestepTimeRange(; start=0.0, step) = TimestepTimeRange(start, step)

function callcounter(times::TimestepTimeRange, prob::Problem, tspan)
    t0, tf = tspan
    dt = timestep(prob)

    tmin = times.start - dt / 2
    return function (state::AbstractState)
        t = timevalue(state)
        return if t > tmin
            t1 = max(times.start, t - dt / 2)
            t2 = min(tf, t + dt / 2)
            i1, i2 = @. ((t1, t2) - times.start) / times.step
            1 + floor(Int, i2) - ceil(Int, i1)
        else
            0
        end
    end
end

function max_timestep_count(times::TimestepTimeRange, prob::Problem, tspan)
    t0, tf = tspan
    dt = timestep(prob)

    tmin = t0 - dt
    tmax = tf + dt

    t1 = times.start < tmin ? tmin : times.start
    t2 = tmax

    i1, i2 = @. ((t1, t2) - times.start) / times.step

    return 1 + floor(Int, i2) - ceil(Int, i1)
end

"""
    TimestepIndices(i::AbstractVector{Int}) :: Timesteps

Specifies the n'th timestep for each n in `i`. `i` must be sorted.
The first timestep is `1`.
"""
struct TimestepIndices{V<:AbstractVector} <: Timesteps
    i::V
end

function callcounter(times::TimestepIndices, ::Problem, tspan)
    indices = Iterators.Stateful(times.i)

    while !isempty(indices) && peek(indices) < 1
        popfirst!(indices)
    end

    return function (state::AbstractState)
        count = 0
        while !isempty(indices) && peek(indices) == timeindex(state)
            popfirst!(indices)
            count += 1
        end

        return count
    end
end

function max_timestep_count(times::TimestepIndices, ::Problem, tspan)
    return length(times.i)
end

"""
    TimestepIndexRange(start, step) :: Timesteps
    TimestepIndexRange(; start=1, step) :: Timesteps

An infinite range of timestep indices starting at `start` counting by `step`.
"""
struct TimestepIndexRange <: Timesteps
    start::Int
    step::Int
    function TimestepIndexRange(start::Int, step::Int)
        @assert step > 0
        return new(start, step)
    end
end

TimestepIndexRange(; start=1, step) = TimestepIndexRange(start, step)

function callcounter(times::TimestepIndexRange, ::Problem, tspan)
    return function (state::AbstractState)
        i = timeindex(state)
        call = i >= times.start && (i - times.start) % times.step == 0
        return call ? 1 : 0
    end
end

function max_timestep_count(
    times::TimestepIndexRange, prob::Problem, (t0, tf)::NTuple{2,Float64}
)
    n = timestep_count(prob, (t0, tf))
    imin = max(1, times.start)
    return length(imin:(times.step):n)
end

"""
    TimestepCondition(cond) :: Timesteps

Specifies a timestep if `cond(state)` with the current [`AbstractState`](@ref) is true.
"""
struct TimestepCondition{F} <: Timesteps
    cond::F # state -> bool
end

function callcounter(times::TimestepCondition, ::Problem)
    return state -> times.cond(state) ? 1 : 0
end

function max_timestep_count(::TimestepCondition, prob::Problem, tspan)
    return timestep_count(prob, tspan)
end

"""
    Callback(f, timesteps::Timesteps)

Specifies that `f(state)` is to be called with the current [`AbstractState`](@ref) at each
timestep in `timesteps`.
"""
struct Callback{T<:Timesteps}
    f::FunctionWrapper{Nothing,Tuple{AbstractState}}
    times::T
    Callback(f, times::T) where {T<:Timesteps} = new{T}(f, times)
end

"""
    ValueGroup(timesteps::Timesteps, quantities::NamedTuple)
    ValueGroup(timesteps::Timesteps; quantities...)

Specifies timesteps and quantities to retrieve at those timesteps.

At each timestep in `timesteps`, each function `f(state)` in `quantities` is called with the
[`AbstractState`](@ref).
"""
struct ValueGroup{T<:Timesteps,V<:NamedTuple{<:Any,<:Tuple{Vararg{Quantity}}}}
    times::T
    quantities::V
end

function ValueGroup(times::Timesteps, quantities::NamedTuple)
    return ValueGroup(times, map(quantity, quantities))
end
ValueGroup(times::Timesteps; quantities...) = ValueGroup(times, NamedTuple(quantities))

struct SaveToMemCallback
    times::Vector{Float64}
    savers::Vector{QuantityMemSaver}
end

function (caller::SaveToMemCallback)(state::AbstractState)
    push!(caller.times, timevalue(state))
    for saver in caller.savers
        update_saver(saver, state)
    end
    return nothing
end

function quantity_mem_savers!(callbacks::AbstractVector{<:Callback}, value_groups)
    return map(value_groups) do vals::ValueGroup
        times = Float64[]
        savers = map(q -> QuantityMemSaver(q, times), vals.quantities)
        callback = SaveToMemCallback(times, collect(savers))
        push!(callbacks, Callback(callback, vals.times))
        savers
    end
end

Base.@kwdef mutable struct SaveToDiskCallback{F<:Union{HDF5.File,HDF5.Group},V<:ValueGroup}
    group::F
    vals::V
    time::HDF5.Dataset
    timeref::HDF5.Reference
    savers::Vector{QuantityDiskSaver}
    initialized::Bool
    timeindex::Int
    maxcount::Int
end

function SaveToDiskCallback(
    parent::Union{HDF5.File,HDF5.Group}, vals::ValueGroup, prob::Problem, tspan
)
    maxcount = max_timestep_count(vals.times, prob, tspan)

    # Generate a unique key for time
    names = [map(string, keys(vals.quantities)); keys(parent)]
    timekeys = (i == 0 ? "time" : "time$i" for i in Iterators.countfrom(0))
    timekey = first(Iterators.filter(!in(names), timekeys))

    return SaveToDiskCallback(;
        group=parent,
        vals=vals,
        time=create_dataset(parent, timekey, Float64, (maxcount,)),
        timeref=HDF5.Reference(parent, timekey),
        savers=Vector{QuantityDiskSaver}(undef, length(vals.quantities)),
        initialized=false,
        timeindex=1,
        maxcount=maxcount,
    )
end

function (caller::SaveToDiskCallback)(state::AbstractState)
    if caller.timeindex > caller.maxcount
        return nothing
    end

    caller.time[caller.timeindex] = timevalue(state)

    foreach(enumerate(pairs(caller.vals.quantities))) do (i, (name, f))
        value = f(state)

        saver = if caller.initialized
            caller.savers[i]
        else
            caller.savers[i] = create_disk_saver(
                caller.group, string(name), f, caller.timeref, value
            )
        end
        update_saver(saver, caller.timeindex, value)
    end

    caller.initialized = true
    caller.timeindex += 1

    return nothing
end

function push_callbacks!(
    callbacks::AbstractVector{<:Callback}, parent, vals::ValueGroup, prob::Problem, tspan
)
    cb = Callback(SaveToDiskCallback(parent, vals, prob, tspan), vals.times)
    push!(callbacks, cb)
    return nothing
end

function push_callbacks!(
    callbacks::AbstractVector{<:Callback}, parent, vals::AbstractDict, prob::Problem, tspan
)
    for (name, items) in vals
        group = create_group(parent, name)
        push_callbacks!(callbacks, group, items, prob, tspan)
    end
    return nothing
end

"""
    solve(problem::Problem, (t0, tf); out=(), call=(), showprogress=false)

Solve the problem between times `t0` and `tf`.  If given, call each [`Callback`](@ref) in
`call` during the simulation. Return the final state.

# Arguments
- `problem::Problem`: Description of the fluid and bodies to simulate.
- `(t0, tf)`: Start and end of the time span to solve for.
- `out=()`: A collection of [`ValueGroup`](@ref) to save from the simulation.
- `call=()`: A collection of [`Callback`](@ref) to call during the simulation.
- `showprogress=false`: Whether to show a progress bar during the simulation.

# Returns
The result of each [`ValueGroup`](@ref) in `out`, mapped with `map`.
"""
function solve(problem::Problem, (t0, tf)::NTuple{2,Float64}; kw...)
    state = initstate(problem, t0)
    return solve!(state, problem, tf; kw...)
end

"""
    solve(file::Union{HDF5.File,HDF5.Group}, problem, (t0, tf); save, kw...)
    solve(filename::String, problem, (t0, tf); save, kw...)

Solve the problem and save values into an HDF5 file. `save` can be either be a
[`ValueGroup`](@ref) or an arbitrarily nested dictionary mapping strings to
[`ValueGroup`](@ref)s.
"""
function solve(
    file::Union{HDF5.File,HDF5.Group}, problem::Problem, (t0, tf)::NTuple{2,Float64}; kw...
)
    state = initstate(problem, t0)
    return solve!(state, file, problem, tf; kw...)
end

function solve(filename::String, args...; kw...)
    # create file if non-existing, preserve contents otherwise
    return h5open(filename, "cw") do file
        solve(file, args...; kw...)
    end
end

"""
    solve!(state::AbstractState, [file], problem::Problem, tf; kw...)

Solve a problem up to time `tf` like [`solve`](@ref) but using `state` as an initial state.
"""
function solve!(
    state::AbstractState, problem::Problem, tf::Float64; out=(), call=(), showprogress=false
)
    tspan = (timevalue(state), tf)

    callbacks = Callback[]
    if showprogress
        prog = Progress(timestep_count(problem, tspan); desc="solving")
        printer = Callback(_ -> next!(prog), AllTimesteps())
        push!(callbacks, printer)
    end

    output_savers = quantity_mem_savers!(callbacks, out) # callbacks for saving output data
    append!(callbacks, call)

    # Track how many times we need to call each callback each timestep
    counters = map(callbacks) do callback::Callback
        counter = callcounter(callback.times, problem, tspan)
        (callback, counter)
    end

    solve!(state, problem, tf) do state
        for (callback, counter) in counters
            # Call each callback n times on the current state
            # Usually either 0 or 1, but can be more if there are duplicate timesteps
            for _ in 1:counter(state)
                callback.f(state)
            end
        end
    end

    return map(savers -> map(mem_saver_values, savers), output_savers)
end

function solve!(
    state::AbstractState,
    file::Union{HDF5.File,HDF5.Group},
    problem::Problem,
    tf::Float64;
    call=(),
    save,
    kw...,
)
    tspan = (timevalue(state), tf)

    callbacks = Callback[]
    push_callbacks!(callbacks, file, save, problem, tspan)
    append!(callbacks, call)

    return solve!(state, problem, tf; call=callbacks, kw...)
end

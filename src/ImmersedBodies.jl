module ImmersedBodies

using HDF5
using FunctionWrappers: FunctionWrapper
using ProgressMeter

export AbstractScheme, CNAB, default_scheme

export FluidConditions, FluidDiscretization, AbstractFluid
export conditions, discretized, gridstep, default_gridstep
export AbstractState, AbstractSolver, Problem
export timestep, timevalue, timeindex, discretized, initstate, advance!

export AbstractFrame, BaseFrame, GlobalFrame, DiscretizationFrame
export Direction, XAxis, YAxis
export OffsetFrame, OffsetFrameInstant

export Curves, Segments, partition
export AbstractBody, BodyGroup, RigidBody, Panels, PanelView, npanels, bodypanels

export AbstractBody, BodyGroup, RigidBody, npanels, Panels, PanelView
export EulerBernoulliBeamBody, ClampIndexBC, ClampParameterBC
export Quantities, quantity, coordinates

export Timesteps, TimestepCondition, AllTimesteps, timestep_times, timestep_indices
export TimestepTimes, TimestepTimeRange, TimestepIndices, TimestepIndexRange
export Callback, ValueGroup, solve, solve!, timestep_count, quantity_values

export FreestreamFlow, PsiOmegaFluidGrid, UniformGrid, MultiLevelGrid

_show(io::IO, x) = _show(io, x, "")

include("dynamics.jl")
using .Dynamics

include("curves.jl")
using .Curves

include("fluids.jl")

include("bodies.jl")
using .Bodies

include("problems.jl")

include("quantities/quantities.jl")
using .Quantities

include("solving/solving.jl")

include("solvers/solvers.jl")
using .Solvers

include("plot-recipes.jl")

end # module ImmersedBodies

module ImmersedBodies

using Base: @kwdef
using LinearAlgebra

using StaticArrays
using FunctionWrappers: FunctionWrapper
using EllipsisNotation
using LinearMaps
using FFTW
using IterativeSolvers
using HDF5

export Fluid, CartesianGrid, MultiDomainGrid, MultiDomainExtents, fluid_grid
export GridPoints, GridVertices, GridU, GridV, GridΓ
export GridMotion, StaticGrid, MovingGrid, GridVelocity
export AbstractBody, PresetBody, FsiBody, Bodies, Panels, PanelState
export any_fsi, n_panels, PanelSection, panel_section, panel_range
export StaticBody, MovingRigidBody
export AbstractScheme, default_scheme, CNAB
export Problem, State
export solve, solve!, advance!, AbstractSolver, create_solver, CnabSolver
export save_state, load_state, load_state!
export Timesteps, TimestepRange, SolutionValues
export SolutionValue, ArrayValue, MultiDomainValue
export SolutionSaver, SaveHDF5, fluid_group, body_group
export Curves

export x_velocity, x_velocity!, y_velocity, y_velocity!
export vorticity, vorticity!
export boundary_force, boundary_force!, boundary_pos, boundary_pos!
export boundary_ds, boundary_ds!, boundary_vel, boundary_vel!
export boundary_total_force, boundary_total_force!

include("problem.jl")
include("fluid.jl")
include("coupling.jl")
include("bodies.jl")
include("io/io.jl")
include("solver.jl")
include("curves.jl")
import .Curves

end # module

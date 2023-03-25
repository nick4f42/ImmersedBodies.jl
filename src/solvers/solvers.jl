module Solvers

using ...ImmersedBodies
using ...ImmersedBodies.Bodies
using ...ImmersedBodies.Quantities
import ...ImmersedBodies: advance!, solve!, statetype, solvertype, _show
import ...ImmersedBodies: gridstep, default_gridstep, body_segment_length
import ...ImmersedBodies: timevalue, timeindex, timestep_scheme, conditions, discretized
import ...ImmersedBodies.Bodies: bodypanels, prescribe_motion!

using EllipsisNotation
using FFTW
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using StaticArrays
using FunctionWrappers: FunctionWrapper

export FreestreamFlow, PsiOmegaFluidGrid, UniformGrid, MultiLevelGrid
export sublevel, baselevel, nlevels

include("fluids.jl")

# Type of a problem where the bodies are all static relative to the discretization
const StaticBodyProblem{S} = Union{
    Problem{
        PsiOmegaFluidGrid{S,GlobalFrame},
        <:RigidBody{<:Union{GlobalFrame,DiscretizationFrame}},
    },
    Problem{<:PsiOmegaFluidGrid{S},RigidBody{DiscretizationFrame}},
}

include("states.jl")
include("fluid-ops.jl")
include("structure-ops.jl")
include("coupling.jl")
include("timestepping.jl")

end # module Solvers

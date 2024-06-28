module ImmersedBodies

using StaticArrays
using KernelAbstractions: get_backend, @index, @kernel

export GridKind, Primal, Dual
export AllAxes
export GridLocation, Node, Edge, Loc_u, Loc_ω
export Grid, gridcorner, gridstep, coord
export IrrotationalFlow, UniformFlow
export AbstractBody, PrescribedBody, StaticBody
export IBProblem

include("utils.jl")
include("problems.jl")
include("operators.jl")

end

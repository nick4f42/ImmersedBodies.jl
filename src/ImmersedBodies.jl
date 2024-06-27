module ImmersedBodies

using StaticArrays
using KernelAbstractions: get_backend, @index, @kernel
using OrderedCollections: OrderedDict

export GridKind, Primal, Dual
export AllAxes
export GridLocation, Node, Edge, Loc_u, Loc_ω
export Grid, gridcorner, gridstep, coord
export IrrotationalFlow, UniformFlow
export AbstractBody, PrescribedBody, StaticBody
export IBProblem

include("util.jl")
include("problems.jl")
include("operators.jl")

end

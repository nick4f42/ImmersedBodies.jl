module ImmersedBodies

using StaticArrays
using OffsetArrays
using KernelAbstractions
using EllipsisNotation
import Adapt

export GridKind, Primal, Dual
export AllAxes
export GridLocation, Node, Edge, Loc_u, Loc_ω
export Grid, gridcorner, gridstep, coord
export IrrotationalFlow, UniformFlow
export AbstractBody, PrescribedBody, StaticBody
export IBProblem

include("fft-r2r.jl")
include("utils.jl")
include("problems.jl")
include("operators.jl")

end

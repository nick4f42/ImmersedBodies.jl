module ImmersedBodies

using LinearAlgebra
using StaticArrays
using StaticArrays: SOneTo
using OffsetArrays
using OffsetArrays: no_offset_view
using KernelAbstractions
using EllipsisNotation
using LinearMaps
import Adapt
import FFTW

export GridKind, Primal, Dual
export AllAxes
export GridLocation, Node, Edge, Loc_u, Loc_ω
export Grid, gridcorner, gridstep, coord, cell_axes
export IrrotationalFlow, UniformFlow
export AbstractBody, PrescribedBody, StaticBody
export IBProblem
export set_time!, step!, zero_vorticity!, surface_force!, surface_force_sum
export CNAB

include("fft-r2r.jl")
include("utils.jl")
include("problems.jl")
include("operators.jl")
include("cnab.jl")

end

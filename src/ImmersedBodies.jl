module ImmersedBodies

using StaticArrays
using KernelAbstractions: get_backend, @index, @kernel
using OrderedCollections: OrderedDict

export Grid
export IrrotationalFlow, UniformFlow
export AbstractBody, PrescribedBody, StaticBody
export IBProblem

include("util.jl")
include("problems.jl")
include("operators.jl")

end

"""
    AbstractFluid

Describes a region of fluid and its discretization.
"""
abstract type AbstractFluid end

"""
    FluidDiscretization

Describes how a fluid is discretized.
"""
abstract type FluidDiscretization end

"""
    discretized(fluid::AbstractFluid)

The [`FluidDiscretization`](@ref) of a fluid.
"""
function discretized end

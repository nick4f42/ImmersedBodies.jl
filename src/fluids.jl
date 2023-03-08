"""
    FluidConditions

The boundary conditions and properties of the flow.
"""
abstract type FluidConditions end

"""
    FluidDiscretization

Describes how a fluid is discretized.
"""
abstract type FluidDiscretization end

"""
    AbstractFluid

Describes a region of fluid to simulate.
"""
abstract type AbstractFluid end

"""
    conditions(fluid::AbstractFluid)

The [`FluidConditions`](@ref) of a fluid.
"""
function conditions end

"""
    discretized(fluid::AbstractFluid)

The [`FluidDiscretization`](@ref) of a fluid.
"""
function discretized end

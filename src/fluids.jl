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
    gridstep(grid::FluidDiscretization)

The minimum spacing of a discretization.
"""
function gridstep end

"""
    default_gridstep(flow::FluidConditions)

A good gridstep to use for the given `flow`.
"""
function default_gridstep end

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

"""
    body_segment_length(fluid::AbstractFluid)

The optimal body segment length of a body simulated with `fluid`.
"""
function body_segment_length end

"""
    partition(curve::Curve, fluid::AbstractFluid) :: Segments

Partition a curve to best be simulated as a body in `fluid`.
"""
function Curves.partition(curve::Curve, fluid::AbstractFluid)
    return partition(curve, body_segment_length(fluid))
end

Base.show(io::IO, ::MIME"text/plain", fluid::AbstractFluid) = _show(io, fluid)

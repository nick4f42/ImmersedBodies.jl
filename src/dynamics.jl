"""
    Dynamics

2D physical reference frames and dynamics calculations.
"""
module Dynamics

using FunctionWrappers: FunctionWrapper
using StaticArrays

export AbstractFrame, BaseFrame, GlobalFrame, DiscretizationFrame
export Direction, XAxis, YAxis, OffsetFrame, OffsetFrameInstant

"""
    AbstractFrame

Describes a physical reference frame over time.
"""
abstract type AbstractFrame end

"""
    BaseFrame <: AbstractFrame

A reference frame not defined by other reference frames.
"""
abstract type BaseFrame <: AbstractFrame end

"""
    GlobalFrame <: BaseFrame

The global base reference frame. Also known as the "lab" frame.
"""
struct GlobalFrame <: BaseFrame end

"""
    DiscretizationFrame <: BaseFrame

The reference frame that the fluid is discretized in.
"""
struct DiscretizationFrame <: BaseFrame end

"""
    Direction

A direction in space.
"""
abstract type Direction end

"""
    XAxis(frame::AbstractFrame) :: Direction

A direction along the x basis vector of `frame`.
"""
struct XAxis{F<:AbstractFrame} <: Direction
    frame::F
end

"""
    YAxis(frame::AbstractFrame) :: Direction

A direction along the y basis vector of `frame`.
"""
struct YAxis{F<:AbstractFrame} <: Direction
    frame::F
end

# Types for scalars and coordinate vectors
const Scalar = Float64
const Coords = SVector{2,Scalar}

# Types for functions of time y = f(t)
const CoordsFunc = FunctionWrapper{Coords,Tuple{Scalar}}
const ScalarFunc = FunctionWrapper{Scalar,Tuple{Scalar}}

"""
    OffsetFrame(f, base::BaseFrame) :: AbstractFrame

A reference frame defined by an offset `f(t)` from another. `f` should return a
[`OffsetFrameInstant`](@ref).
"""
struct OffsetFrame{F<:BaseFrame} <: AbstractFrame
    f
    base::F
end

"""
    OffsetFrameInstant(r, v, θ, Ω)

The instantaneous offset of a reference frame in another.

# Arguments
- `r`: Origin's position in base frame `[x, y]`.
- `v`: Origin's velocity in base frame `[vx, vy]`.
- `θ`: Angular position in base frame.
- `Ω`: Angular velocity in base frame.
"""
struct OffsetFrameInstant <: AbstractFrame
    r::Coords
    v::Coords
    θ::Scalar
    Ω::Scalar
    cθ::Scalar # cos(θ)
    sθ::Scalar # sin(θ)
    function OffsetFrameInstant(r, v, θ, Ω)
        return new(r, v, θ, Ω, cos(θ), sin(θ))
    end
end

"""
    (frame::OffsetFrame)(t) :: OffsetFrameInstant

Evaluate a reference frame at a point in time.
"""
(frame::OffsetFrame)(t::Real) = frame.f(t)

end # module Dynamics

module Curves

export Curve, ClosedCurve, OpenCurve, Segments
export isclosed, arclength, partition
export LineSegment, Circle, ParameterizedCurve

using StaticArrays
using Interpolations
using FunctionWrappers: FunctionWrapper
using LinearAlgebra: norm

"""
    Curve

A mathematical curve in 2D space. Calling `curve(t)` with `0 ≤ t ≤ 1` gives an `[x y]` point
along the curve.
"""
abstract type Curve end
abstract type ClosedCurve <: Curve end
abstract type OpenCurve <: Curve end

"""
    isclosed(curve::Curve)

Return true if the `curve`'s start and end points are equal, false otherwise.
"""
isclosed(::ClosedCurve) = true
isclosed(::OpenCurve) = false

struct Segments
    points::Matrix{Float64}
    lengths::Vector{Float64}
end

"""
    arclength(curve::Curve)

The total arclength of a curve.
"""
function arclength end

"""
    partition(curve::Curve, ds::AbstractFloat) :: Segments
    partition(curve::Curve, n::Integer) :: Segments

Partition a curve into segments of approximately equal length. Either specify the target
segment length `ds` or the target segment count `n`. The curve's endpoints are preserved.
"""
function partition end

# Curves either need to define a method for ds or for n
# The other is satisfied by these defaults
function partition(curve::Curve, ds::AbstractFloat)
    n = round(Int, arclength(curve) / ds) + !isclosed(curve)
    return partition(curve, n)
end
function partition(curve::Curve, n::Integer)
    ds = arclength(curve) / (n - !isclosed(curve))
    return partition(curve, ds)
end

"""
    LineSegment((x1, y1), (x2, y2)) :: Curve

A line segment between two points.
"""
struct LineSegment <: OpenCurve
    p1::SVector{2,Float64}
    p2::SVector{2,Float64}
end

(line::LineSegment)(t) = line.p1 + t * (line.p2 - line.p1)

arclength(line::LineSegment) = hypot((line.p2 .- line.p1)...)

function Base.show(io::IO, ::MIME"text/plain", line::LineSegment)
    print(io, "LineSegment: from ", line.p1, " to ", line.p2)
    return nothing
end

function partition(line::LineSegment, n::Integer)
    @assert n > 1

    xs, ys = (range(x1, x2, n) for (x1, x2) in zip(line.p1, line.p2))
    ds = hypot(xs[2] - xs[1], ys[2] - ys[1])

    points = [xs ys]
    lengths = fill(ds, n)

    return Segments(points, lengths)
end

"""
    Circle(r=1, center=(0, 0)) :: Curve

A circle with radius `r` centered at `center`.
"""
struct Circle <: ClosedCurve
    r::Float64
    center::SVector{2,Float64}
    Circle(r=1, center=(0, 0)) = new(r, center)
end

function (circle::Circle)(t)
    s = 2 * pi * t
    return circle.center + circle.r * SVector(cos(s), sin(s))
end

arclength(circle::Circle) = 2 * pi * circle.r

function Base.show(io::IO, ::MIME"text/plain", circle::Circle)
    print(io, "Circle: radius=", circle.r, ", center=", circle.center)
    return nothing
end

function partition(circle::Circle, n::Integer)
    x0, y0 = circle.center
    r = circle.r

    t = 2pi / n * (0:(n - 1))

    xs = @. x0 + r * cos(t)
    ys = @. y0 + r * sin(t)
    ds = hypot(xs[2] - xs[1], ys[2] - ys[1])

    points = [xs ys]
    lengths = fill(ds, n)

    return Segments(points, lengths)
end

"""
    ParameterizedCurve(f; n_sample=100) :: Curve

A curve defined by the points `(x, y) = f(t)` for `0 ≤ t ≤ 1`.

`n_sample` points are used to determine an equal spacing when partitioning the curve.
"""
struct ParameterizedCurve{F,P} <: Curve
    f::F # parameter -> point
    param::P # equally spaced arclength -> parameter
    arclen::Float64
    closed::Bool # whether f(0) ≈ f(1)
    function ParameterizedCurve(f; n_sample=100)
        fv = SVector{2,Float64} ∘ f

        t_sample = range(0, 1, n_sample)

        s_sample = sample_arclengths(fv, t_sample)
        arclen = s_sample[end]

        s_sample ./= arclen # Normalize to [0, 1]

        param = interpolate((s_sample,), t_sample, Gridded(Linear()))

        closed = norm(fv(1) - fv(0)) / arclen < 1e-8

        return new{typeof(fv),typeof(param)}(fv, param, arclen, closed)
    end
end

arclength(curve::ParameterizedCurve) = curve.arclen
isclosed(curve::ParameterizedCurve) = curve.closed
(curve::ParameterizedCurve)(t) = curve.f(curve.param(t))

function partition(curve::ParameterizedCurve, n::Integer)
    points = Matrix{Float64}(undef, n, 2)

    ts = if isclosed(curve)
        range(0, 1, n + 1)[1:n]
    else
        range(0, 1, n)
    end

    for (i, t) in enumerate(ts)
        points[i, :] .= curve(t)
    end

    lengths = point_array_lengths(points, isclosed(curve))
    return Segments(points, lengths)
end

function sample_arclengths(f, t)
    s = similar(t)

    p1 = f(t[1])
    s[1] = 0
    for i in eachindex(s)[2:end]
        p2 = f(t[i])
        s[i] = s[i - 1] + norm(p2 - p1)
        p1 = p2
    end

    return s
end

function point_array_lengths(points, closed::Bool)
    lengths = Vector{Float64}(undef, size(points, 1))
    return point_array_lengths!(lengths, points, closed)
end

function point_array_lengths!(lengths, points, closed::Bool)
    n = size(points, 1)
    @assert n > 1 "cannot determine length with single point"

    len(p, i, j) = norm(SVector{2}(p[i, :]) - SVector{2}(p[j, :]))

    lens = Vector{Float64}(undef, n - 1)
    for i in eachindex(lens)
        lens[i] = len(points, i, i + 1)
    end

    if closed
        endlen = len(points, n, 1)
        lengths[1] = (endlen + lens[1]) / 2
        lengths[n] = (lens[n - 1] + endlen) / 2
    else
        lengths[1] = lens[1]
        lengths[n] = lens[n - 1]
    end
    for i in 2:(n - 1)
        lengths[i] = (lens[i - 1] + lens[i]) / 2
    end

    return lengths
end

end # module Curves

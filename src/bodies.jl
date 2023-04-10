module Bodies

using ..ImmersedBodies
using ..ImmersedBodies.Dynamics
using ..ImmersedBodies.Curves
import ..ImmersedBodies: _show

using StaticArrays

export AbstractBody, BodyGroup, Panels, PanelView, npanels, bodypanels
export RigidBody, DeformingBody, EulerBernoulliBeam, is_static
export reference_pos, n_variables, deforming
export StructureModel, LinearModel, ClampIndexBC, ClampParameterBC
export DeformationState, DeformationStateView

# typeof(@view matrix[i:j, :])
const MatrixRowView{T} = SubArray{
    T,2,Matrix{T},Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false
}

# typeof(@view vector[i:j])
const VectorView{T} = SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}

"""
    AbstractBody

A structural body.
"""
abstract type AbstractBody end

Base.show(io::IO, ::MIME"text/plain", body::AbstractBody) = _show(io, body)

"""
    initial_pos!(xb::AbstractMatrix{Float64}, body::AbstractBody)

The initial `[x y]` panels on a body.
"""
function initial_pos! end

"""
    initial_lengths!(ds::AbstractVector{Float64}, body::AbstractBody)

The initial body segment lengths.
"""
function initial_lengths! end

"""
    npanels(body::AbstractBody)
    npanels(bodies::BodyGroup)
    npanels(panels::Panels)

The number of structural panels in a body or bodies.
"""
function npanels end

"""
    RigidBody(
        pos::AbstractMatrix{Float64},
        len::AbstractVector{Float64},
        frame::AbstractFrame=DiscretizationFrame()
    ) :: AbstractBody

A rigid body with optional prescribed motion.

# Arguments
- `pos`: Rows of `[x y]` points that define the body.
- `len`: Segment length at each point in `pos`.
- `motion`: Prescribed motion of the body.
"""
struct RigidBody{F<:AbstractFrame} <: AbstractBody
    pos::Matrix{Float64} # panel positions
    len::Vector{Float64} # panel lengths
    frame::F # prescribed motion
end

RigidBody(pos, len) = RigidBody(pos, len, DiscretizationFrame())
initial_pos!(xb, body::RigidBody) = xb .= body.pos
initial_lengths!(ds, body::RigidBody) = ds .= body.len

"""
    RigidBody(segments::Segments, frame=DiscretizationFrame()) :: AbstractBody

# Arguments
- `segments::Segments`: Points and lengths on the body.
- `frame::AbstractFrame`: Prescribed motion of the body.
"""
function RigidBody(segments::Segments, frame=DiscretizationFrame())
    return RigidBody(Curves.points(segments), Curves.lengths(segments), frame)
end

npanels(body::RigidBody) = length(body.len)

is_static(body::RigidBody{DiscretizationFrame}, ::AbstractFrame) = true
is_static(body::RigidBody{F}, ::F) where {F<:BaseFrame} = true
is_static(body::RigidBody, ::AbstractFrame) = false

function _show(io::IO, body::RigidBody, prefix)
    print(io, prefix, "RigidBody:")
    if get(io, :compact, false)
        print(io, " in frame ", body.frame)
    else
        ioc = IOContext(io, :limit => true, :compact => true)

        print(ioc, '\n', prefix, "  points = ")
        summary(ioc, body.pos)

        print(ioc, '\n', prefix, "   frame = ")
        summary(ioc, body.frame)
    end

    return nothing
end

abstract type DeformingBody <: AbstractBody end

abstract type StructureModel end

Base.@kwdef struct LinearModel <: StructureModel
    m::Vector{Float64}
    kb::Vector{Float64}
end

"""
    ClampIndexBC(i::Int)

A boundary condition that index `i` on a body's points is clamped.
"""
struct ClampIndexBC
    i::Int
end

"""
    ClampParameterBC(t::Float64)

A boundary condition that `curve(t)` on a [`Curve`](@ref) is clamped.
"""
struct ClampParameterBC
    t::Float64
end

struct EulerBernoulliBeam{M<:StructureModel} <: DeformingBody
    model::M
    xref::Matrix{Float64} # Reference locations about which displacements are determined
    ds0::Vector{Float64} # Line segment lengths on body in undeformed configuration
    bcs::Vector{ClampIndexBC} # Boundary conditions
end

reference_pos(body::EulerBernoulliBeam) = body.xref

# 2 displacements per poin
n_variables(body::EulerBernoulliBeam{LinearModel}) = 2 * npanels(body)

initial_pos!(xb, body::EulerBernoulliBeam) = xb .= body.xref
initial_lengths!(ds, body::EulerBernoulliBeam) = ds .= body.ds0

"""
    search_closest(xs, x)

Return the index of the closest element in `xs` to `x` assuming `xs` is sorted.
"""
function search_closest(xs, x)
    i2 = searchsortedfirst(xs, x)

    i2 > lastindex(xs) && return lastindex(xs)
    i2 == firstindex(xs) && return firstindex(xs)

    i1 = i2 - 1
    return xs[i2] - x > x - xs[i1] ? i1 : i2
end

function EulerBernoulliBeam(
    ::Type{LinearModel},
    segments::Segments,
    bcs::Vector{<:ClampParameterBC};
    m::Float64,
    kb::Float64,
)
    xref = segments.points
    ds0 = segments.lengths

    nel = size(xref, 1) - 1
    model = LinearModel(; m=fill(m, nel), kb=fill(kb, nel))

    ts = cumsum(ds0)
    @. ts = (ts - ts[1]) / (ts[end] - ts[1])

    bc_indices = map(bcs) do bc
        if !(0 <= bc.t <= 1)
            (throw ∘ DomainError)(
                "boundary condition parameter clamp must be between 0 and 1", bc.t
            )
        end
        ClampIndexBC(search_closest(ts, bc.t))
    end

    return EulerBernoulliBeam(model, xref, ds0, bc_indices)
end

npanels(body::EulerBernoulliBeam) = length(body.ds0)

function _show(io::IO, body::EulerBernoulliBeam, prefix)
    print(io, prefix)
    summary(io, body)
    print(io, ':')
    if get(io, :compact, false)
        print(io, " with bcs ", body.bcs)
    else
        ioc = IOContext(io, :limit => true, :compact => true)

        print(ioc, '\n', prefix, "  reference points = ")
        summary(ioc, body.xref)

        print(ioc, '\n', prefix, "    boundary conds = ", body.bcs)
    end

    return nothing
end

struct DeformingBodyIndex
    i_body::Int
    i_panel::Int
    i_deform_panel::Int
end

"""
    BodyGroup(bodies::Vector{<:AbstractBody})

A collection of bodies.
"""
struct BodyGroup{B<:AbstractBody} <: AbstractVector{B}
    bodies::Vector{B} # all bodies
    npanel::Int
    npanel_deform::Int
    deforming::Vector{DeformingBodyIndex}
    function BodyGroup(bodies::Vector{B}) where {B<:AbstractBody}
        deforming = DeformingBodyIndex[]

        n_panel = 0
        n_panel_deform = 0
        for (i_body, body) in enumerate(bodies)
            n = npanels(body)

            if body isa DeformingBody
                push!(deforming, DeformingBodyIndex(i_body, n_panel, n_panel_deform))
                n_panel_deform += n
            end

            n_panel += n
        end

        return new{B}(bodies, n_panel, n_panel_deform, deforming)
    end
end

Base.size(bodies::BodyGroup) = size(bodies.bodies)
Base.getindex(bodies::BodyGroup, i) = bodies.bodies[i]
Base.IndexStyle(::BodyGroup) = IndexLinear()

npanels(bodies::BodyGroup) = bodies.npanel

deforming(bodies::BodyGroup) = (bodies[i.i_body] for i in bodies.deforming)

Base.show(io::IO, ::MIME"text/plain", bodies::BodyGroup) = _show(io, bodies)

function _show(io::IO, bodies::BodyGroup, prefix)
    print(io, prefix)
    summary(io, bodies)
    print(io, ":\n")

    indent = prefix * "  "
    for body in bodies
        _show(io, body, indent)
        println(io)
    end
    return nothing
end

"""
    PanelView

View into sequences of panels in [`Panels`](@ref).
"""
struct PanelView
    i_panel::Int
    pos::MatrixRowView{Float64}
    vel::MatrixRowView{Float64}
    len::VectorView{Float64}
    traction::MatrixRowView{Float64}
end

npanels(panels::PanelView) = length(panels.len)

"""
    Panels

A group of structural panels for multiple bodies.

Coordinates are given in the frame that the fluid is discretized in.
"""
struct Panels
    pos::Matrix{Float64} # panel positions
    vel::Matrix{Float64} # panel velocities
    len::Vector{Float64} # panel lengths
    traction::Matrix{Float64} # traction on each panel
    perbody::Vector{PanelView} # panels grouped by body
end

function Panels(bodies::BodyGroup)
    n = npanels(bodies)

    pos = zeros(n, 2)
    vel = zeros(n, 2)
    len = zeros(n)
    traction = zeros(n, 2)

    perbody = Vector{PanelView}(undef, length(bodies))
    i_panel = 0
    for (i, body) in enumerate(bodies)
        n_panel = npanels(body)
        r = i_panel .+ (1:n_panel)
        perbody[i] = @views PanelView(i_panel, pos[r, :], vel[r, :], len[r], traction[r, :])
        i_panel += n_panel
    end

    for (body, panels) in zip(bodies, perbody)
        initial_pos!(panels.pos, body)
        initial_lengths!(panels.len, body)
    end

    return Panels(pos, vel, len, traction, perbody)
end

npanels(p::Panels) = length(p.len)

"""
    bodypanels(state::AbstractState) :: Panels

Return the [`Panels`](@ref) of a state.
"""
function bodypanels end

function prescribe_motion!(::PanelView, ::AbstractFrame, ::RigidBody, t::Float64)
    # TODO: Implement prescribed for other reference frames
    throw(ArgumentError("unsupported reference frame"))
end

function prescribe_motion!(
    ::PanelView, ::F1, ::RigidBody{F2}, ::Float64
) where {F1<:BaseFrame,F2<:Union{F1,DiscretizationFrame}}
    return nothing
end

function prescribe_motion!(
    panels::PanelView, ::F1, body::RigidBody{OffsetFrame{F2}}, t::Float64
) where {F1<:BaseFrame,F2<:Union{F1,DiscretizationFrame}}
    f = body.frame(t) # evaluate frame at time t
    r0 = f.r
    v0 = f.v
    Ω = f.Ω
    c = f.cθ # cos(θ)
    s = f.sθ # sin(θ)

    Rx = @SMatrix [c -s; s c]
    Rv = Ω * @SMatrix [-s -c; c -s]

    for (r, v, rb) in zip(eachrow.((panels.pos, panels.vel, body.pos))...)
        r .= r0 + Rx * rb
        v .= v0 + Rv * rb
    end

    return nothing
end

"""
    DeformationStateView

A view into [`DeformationState`](@ref).
"""
struct DeformationStateView
    χ::VectorView{Float64} # Structural displacements
    ζ::VectorView{Float64} # Structural velocities
    ζdot::VectorView{Float64} # Structural accels
end

"""
    DeformationState

The state of deformation of all deforming bodies.
"""
struct DeformationState
    χ::Vector{Float64} # Structural displacements
    ζ::Vector{Float64} # Structural velocities
    ζdot::Vector{Float64} # Structural accels
    perbody::Vector{DeformationStateView}
end

function DeformationState(vars_per_body)
    n_bodies = length(vars_per_body)
    n_vars = sum(vars_per_body; init=0)

    χ = zeros(n_vars)
    ζ = zeros(n_vars)
    ζdot = zeros(n_vars)

    perbody = Vector{DeformationStateView}(undef, n_bodies)
    i = 0
    for (i_body, n) in enumerate(vars_per_body)
        r = i .+ (1:n)
        perbody[i_body] = @views DeformationStateView(χ[r], ζ[r], ζdot[r])
        i += n
    end

    return DeformationState(χ, ζ, ζdot, perbody)
end

function DeformationState(bodies::BodyGroup)
    return DeformationState(n_variables(body) for body in deforming(bodies))
end

function Base.similar(deform::DeformationState)
    vars_per_body = (length(def.χ) for def in deform.perbody)
    return DeformationState(vars_per_body)
end

function Base.copy!(dst::DeformationState, src::DeformationState)
    # Crude check that both states cover the same bodies
    @assert length(dst.perbody) == length(src.perbody)

    dst.χ .= src.χ
    dst.ζ .= src.ζ
    dst.ζdot .= src.ζdot

    return dst
end

end # module Bodies

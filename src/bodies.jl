module Bodies

using ..ImmersedBodies
using ..ImmersedBodies.Dynamics
using ..ImmersedBodies.Curves

using StaticArrays

export AbstractBody, BodyGroup, Panels, PanelView, npanels, bodypanels, body_segment_length
export RigidBody

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
    body_segment_length(fluid::AbstractFluid)

The optimal body segment length of a body simulated with `fluid`.
"""
function body_segment_length end

"""
    RigidBody(
        pos::AbstractMatrix{Float64},
        len::AbstractVector{Float64},
        motion::AbstractFrame=DiscretizationFrame()
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
    RigidBody(
        fluid::AbstractFluid,
        curve::Curve,
        motion::PrescribedMotion=FollowFrame(DiscretizationFrame())
    ) :: AbstractBody

# Arguments
- `fluid`: The fluid that the body will be simulated in.
- `curve`: The curve that defines the shape of the rigid body.
- `motion`: Prescribed motion of the body.
"""
function RigidBody(fluid::AbstractFluid, curve::Curve, motion=DiscretizationFrame())
    segments = partition(curve, body_segment_length(fluid))
    return RigidBody(segments.points, segments.lengths, motion)
end

npanels(body::RigidBody) = length(body.len)

"""
    BodyGroup(bodies::Vector{<:AbstractBody})

A collection of bodies.
"""
struct BodyGroup{B<:AbstractBody} <: AbstractVector{B}
    bodies::Vector{B}
    npanel::Int
    function BodyGroup(bodies::Vector{B}) where {B<:AbstractBody}
        npanel = sum(npanels, bodies)
        return new{B}(bodies, npanel)
    end
end

Base.size(bodies::BodyGroup) = size(bodies.bodies)
Base.getindex(bodies::BodyGroup, i) = bodies.bodies[i]
Base.IndexStyle(::BodyGroup) = IndexLinear()

npanels(bodies::BodyGroup) = bodies.npanel

"""
    PanelView

View into sequences of panels in [`Panels`](@ref).
"""
struct PanelView
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
        perbody[i] = @views PanelView(pos[r, :], vel[r, :], len[r], traction[r, :])
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

function prescribe_motion!(::PanelView, ::RigidBody, t::Float64)
    # TODO: Implement prescribed for other reference frames
    throw(ArgumentError("unsupported reference frame"))
end

prescribe_motion!(::PanelView, ::RigidBody{DiscretizationFrame}, ::Float64) = nothing

function prescribe_motion!(
    panels::PanelView, body::RigidBody{OffsetFrame{DiscretizationFrame}}, t::Float64
)
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

end # module Bodies

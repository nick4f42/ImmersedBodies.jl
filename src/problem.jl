const TimeFunc{T} = FunctionWrapper{T,Tuple{Float64}}

"""
    Panels(x::AbstractMatrix, ds::AbstractVector)

The positions `xb` and lengths `ds` of panels that influence the fluid. `xb` has shape
`(n, 2)` and `ds` has length `n`.
"""
@kwdef struct Panels{T_xb<:AbstractMatrix,T_ds<:AbstractVector}
    xb::T_xb
    ds::T_ds
end

const PanelsF64 = Panels{Matrix{Float64},Vector{Float64}}

function Base.convert(::Type{PanelsF64}, panels::Panels)
    Panels(; xb=convert(Matrix{Float64}, panels.xb), ds=convert(Vector{Float64}, panels.ds))
end

n_panels(p::Panels) = size(p.xb, 1)

"""
    PanelState(; xb::AbstractMatrix, ds::AbstractVector, ub::AbstractMatrix)

The positions `xb`, lengths `ds`, and velocities `ub` of panels that influence the fluid.
`xb` and `ub` have shape `(n, 2)` and `ds` has length `n`.
"""
@kwdef struct PanelState{T_xb<:AbstractMatrix,T_ds<:AbstractVector,T_ub<:AbstractMatrix}
    xb::T_xb
    ds::T_ds
    ub::T_ub
end

PanelState(n::Int) = PanelState(; xb=zeros(n, 2), ds=zeros(n), ub=zeros(n, 2))

const PanelStateF64 = PanelState{Matrix{Float64},Vector{Float64},Matrix{Float64}}

n_panels(p::PanelState) = size(p.xb, 1)

function Base.:(==)(a::PanelState, b::PanelState)
    a.xb == b.xb && a.ds == b.ds && a.ub == b.ub
end

function Base.view(panels::PanelState, i)
    @views PanelState(panels.xb[i, :], panels.ds[i], panels.ub[i, :])
end

"""
    AbstractBody

A body that can be immersed in the fluid.
"""
abstract type AbstractBody end

"""
    n_panels(x)


The number of panels in a thing.
"""
function n_panels end

"""
    bodypanels(body::AbstractBody)

The [`Panels`](@ref) of a body's reference state.
"""
function bodypanels end

"""
    PresetBody <: AbstractBody

A body that is not influenced by the fluid.
"""
abstract type PresetBody <: AbstractBody end

"""
    FsiBody

A body that may be influenced by the fluid.
"""
abstract type FsiBody <: AbstractBody end

"""
    n_variables(::FsiBody)

The number of structural variables in a body.
"""
function n_variables end

"""
    PanelSection

Specifies a subsection of the panels in [`Bodies`](@ref).
"""
abstract type PanelSection end

"""
    section_body(section::PanelSection)

The [`AbstractBody`](@ref) of a `section`.
"""
function section_body end

"""
    panel_range(bodies::Bodies, body::Symbol)
    panel_range(section::PanelSection)

The index range in the panels of the given body.
"""
function panel_range end

struct PresetBodySection{B<:PresetBody} <: PanelSection
    index::Int
    body::B
    panel_range::UnitRange{Int}
end
section_body(part::PresetBodySection) = part.body
panel_range(part::PresetBodySection) = part.panel_range

struct FsiBodySection{B<:FsiBody} <: PanelSection
    index::Int
    body::B
    panel_range::UnitRange{Int}
    fsi_range::UnitRange{Int}
end
section_body(part::FsiBodySection) = part.body
panel_range(part::FsiBodySection) = part.panel_range
fsi_range(part::FsiBodySection) = part.fsi_range

"""
    Bodies(; bodies::AbstractBody...)

A group of bodies. Keyword arguments specify bodies by their name.
"""
struct Bodies
    preset::Vector{PresetBodySection}
    fsi::Vector{FsiBodySection}
    byname::Dict{Symbol,Any}  # Dict{Symbol,<:PanelSection}
    npanel::Int
    static::Bool
    function Bodies(bodies::AbstractDict{Symbol,<:AbstractBody})
        preset = PresetBodySection[]
        fsi = FsiBodySection[]

        panel_offset = 0
        fsi_offset = 0

        byname = Dict{Symbol,Any}()

        for (name, body) in bodies
            npanel = n_panels(body)
            panels = panel_offset .+ (1:npanel)
            panel_offset += npanel

            if body isa PresetBody
                let index = length(preset) + 1,
                    part = PresetBodySection(index, body, panels)

                    push!(preset, part)
                    byname[name] = part
                end
            elseif body isa FsiBody
                let index = length(fsi) + 1,
                    nvar = n_variables(body),
                    part = FsiBodySection(index, body, panels, fsi_offset .+ (1:nvar))

                    push!(fsi, part)
                    fsi_offset += nvar
                    byname[name] = part
                end
            end
        end

        static = isempty(fsi) && all(part -> part.body isa StaticBody, preset)
        new(preset, fsi, byname, panel_offset, static)
    end
end

Bodies(; bodies...) = Bodies(bodies)

"""
    eachbody(bodies::Bodies)

Dictionary mapping body names to [`PanelSection`](@ref)s.
"""
eachbody(bodies::Bodies) = bodies.byname

n_panels(bodies::Bodies) = bodies.npanel

"""
    any_fsi(bodies::Bodies)

Whether there are any FSI bodies in `bodies`.
"""
any_fsi(bodies::Bodies) = !isempty(bodies.fsi)

"""
    panel_section(bodies::Bodies, body::Symbol)

The [`PanelSection`](@ref) of the body named `body` in `bodies`.
"""
panel_section(bodies::Bodies, body::Symbol) = bodies.byname[body]

"""
    panel_range(::Bodies, name::Symbol)

The indices of the given body's panels in all of the panels.
"""
panel_range(bodies::Bodies, body::Symbol) = panel_range(panel_section(bodies, body))

"""
    fsi_range(::Bodies, name::Symbol)

The indices of the given body's FSI variables in all of the FSI variables.
"""
fsi_range(bodies::Bodies, body::Symbol) = fsi_range(panel_section(bodies, body))

Base.getindex(bodies::Bodies, body::Symbol) = section_body(panel_section(bodies, body))

"""
    StaticBody(panels::Panels) :: PresetBody

A body that is fixed in place at `panels`.
"""
struct StaticBody <: PresetBody
    panels::PanelsF64
end

n_panels(body::StaticBody) = n_panels(body.panels)
bodypanels(body::StaticBody) = body.panels

"""
    RigidBodyTransform(pos, vel, angle, angular_vel)
    RigidBodyTransform(; pos, vel, angle, angular_vel)

The transformation of a rigid body at a point in time. Relative to the global frame, the
rigid body is at position `(x, y) = pos`, traveling at velocity `(vx, vy) = vel`, oriented
at angle `angle`, and rotating at rate `angular_vel`.
"""
@kwdef struct RigidBodyTransform
    pos::SVector{2,Float64}
    vel::SVector{2,Float64}
    angle::Float64
    angular_vel::Float64
end

"""
    MovingRigidBody(panels::Panels, motion) :: PresetBody

A non-deforming body with displacement and rotation over time. With zero displacement and
zero angle, the body is at `panels`. `motion(t)` is the [`RigidBodyTransform`](@ref) of the
body at time `t`.
"""
struct MovingRigidBody <: PresetBody
    panels::PanelsF64
    motion::FunctionWrapper{RigidBodyTransform,Tuple{Float64}}
end

n_panels(body::MovingRigidBody) = n_panels(body.panels)
bodypanels(body::MovingRigidBody) = body.panels

"""
    CartesianGrid(h, xlims, ylims)
    CartesianGrid(; h, xlims, ylims)

A Cartesian grid with grid step `h`.
"""
struct CartesianGrid
    x0::NTuple{2,Float64}
    n::NTuple{2,Int}
    h::Float64
end

function CartesianGrid(h::Real, lims::Vararg{NTuple{2},2})
    x0 = map(first, lims)
    n = map(lim -> round(Int, (lim[2] - lim[1]) / h), lims)
    CartesianGrid(x0, n, h)
end

CartesianGrid(; h, xlims, ylims) = CartesianGrid(h, xlims, ylims)

"""
    gridstep(::CartesianGrid)

The spacing between grid vertices.
"""
gridstep(grid::CartesianGrid) = grid.h

"""
    extents(::CartesianGrid) -> ((xmin, xmax), (ymin, ymax))

Minimum and maximum coordinates of the grid.
"""
extents(grid::CartesianGrid) = map((x0, n) -> (x0, x0 + n * grid.h), grid.x0, grid.n)

abstract type GridPoints end
struct GridVertices <: GridPoints end
struct GridU <: GridPoints end
struct GridV <: GridPoints end
struct GridΓ <: GridPoints end

"""
    gridsize(grid::CartesianGrid, pts::GridPoints=GridVertices())

Returns the dimensions of an array of points on the grid.
"""
gridsize(grid::CartesianGrid) = gridsize(grid, GridVertices())
gridsize(grid::CartesianGrid, ::GridVertices) = grid.n

"""
    coords(grid::CartesianGrid, pts::GridPoints=GridVertices())

Returns a tuple of ranges `(x, y)` such that `(x[i], y[j])` is the coordinate of the point
at index `(i, j)`.
"""
coords(grid::CartesianGrid) = coords(grid, GridVertices())
function coords(grid::CartesianGrid, ::GridVertices)
    map(grid.x0, grid.n) do x0, n
        x0 .+ grid.h .* (0:n)
    end
end

function coords(grid::CartesianGrid, ::GridU)
    xs, ys = coords(grid)
    (xs, _midpoints(ys))
end
gridsize(grid::CartesianGrid, ::GridU) = (grid.n[1], grid.n[2] - 1)

function coords(grid::CartesianGrid, ::GridV)
    xs, ys = coords(grid)
    (_midpoints(xs), ys)
end
gridsize(grid::CartesianGrid, ::GridV) = (grid.n[1] - 1, grid.n[2])

function coords(grid::CartesianGrid, ::GridΓ)
    xs, ys = coords(grid)
    (xs[2:(end - 1)], ys[2:(end - 1)])
end
gridsize(grid::CartesianGrid, ::GridΓ) = grid.n .- 2

function _midpoints(r::AbstractRange)
    range(; start=first(r) + step(r) / 2, step=step(r), length=length(r) - 1)
end

"""
    GridIndices

Indices for the fluid grid.

# Fields
- `nx`: Number of grid cells along x.
- `ny`: Number of grid cells along y.
- `nu`: Number of x flux points.
- `nv`: Number of y flux points.
- `nq`: Total number of x and y flux points.
- `nΓ`: Number of circulation (or streamfunction) points.
- `L`: Offset of the left boundary in the boundary array.
- `R`: Offset of the right boundary in the boundary array.
- `B`: Offset of the bottom boundary in the boundary array.
- `T`: Offset of the top boundary in the boundary array.
"""
struct GridIndices
    nx::Int
    ny::Int
    nu::Int
    nv::Int
    nq::Int
    nΓ::Int
    L::Int
    R::Int
    B::Int
    T::Int
    function GridIndices(grid::CartesianGrid)
        nx, ny = grid.n

        nu = (nx + 1) * ny
        nv = nx * (ny + 1)
        nq = nu + nv
        nΓ = (nx - 1) * (ny - 1)

        L = 0
        R = ny + 1
        B = 2 * (ny + 1)
        T = 2 * (ny + 1) + nx + 1

        new(nx, ny, nu, nv, nq, nΓ, L, R, B, T)
    end
end

"""
    MultiDomainGrid(; h, xlims, ylims, nlevel)

Multiple [`CartesianGrid`](@ref)s with different scales, centered at the same point.

See also [`fluid_grid`](@ref).
"""
struct MultiDomainGrid
    base::CartesianGrid
    nlevel::Int
    inds::GridIndices
    function MultiDomainGrid(base::CartesianGrid, nlevel)
        new(base, nlevel, GridIndices(base))
    end
end

function MultiDomainGrid(; h, xlims, ylims, nlevel)
    MultiDomainGrid(CartesianGrid(h, xlims, ylims), nlevel)
end

"""
    gridstep(::MultiDomainGrid, level=1)

Grid step on the `level`th multi domain level.
"""
gridstep(grid::MultiDomainGrid) = grid.base.h
gridstep(grid::MultiDomainGrid, level::Integer) = 2.0^(level - 1) * grid.base.h

"""
    subdomain(grid::MultiDomainGrid, level) :: CartesianGrid

Return the `level`th multi domain level of `grid`.
"""
function subdomain(grid::MultiDomainGrid, level::Integer)
    base = grid.base
    h = 2.0^(level - 1) * base.h
    x0 = @. base.x0 + base.n * (base.h - h) / 2
    CartesianGrid(x0, base.n, h)
end

"""
    coords(grid::MultiDomainGrid, pts::GridPoints=GridVertices(), levels=:)

Return the coordinates of each [`CartesianGrid`](@ref) subdomain in `levels`.
"""
function coords(grid::MultiDomainGrid, pts::GridPoints=GridVertices(), levels=:)
    map((1:(grid.nlevel))[levels]) do level
        coords(subdomain(grid, level), pts)
    end
end

"""
    gridsize(grid::MultiDomainGrid, pts::GridPoints=GridVertices(), levels=:)

The dimensions of the multi domain grid.
"""
function gridsize(grid::MultiDomainGrid, pts::GridPoints=GridVertices(), levels=:)
    (gridsize(grid.base, pts)..., length((1:(grid.nlevel))[levels]))
end

"""
    MultiDomainExtents(; xlims, ylims, nlevel)

The region defined by a [`MultiDomainGrid`](@ref), without the grid step.

See also [`fluid_grid`](@ref).
"""
@kwdef struct MultiDomainExtents
    xlims::NTuple{2,Float64}
    ylims::NTuple{2,Float64}
    nlevel::Int
end

"""
    fluid_grid(; [h], xlims, ylims, nlevel)

A cartesian grid with step `h`, limits `xlims` and `ylims`, and `nlevel` total grids that
expand outward. Each grid has the same center point, and grid step `h * 2 ^ (level - 1)` for
`level` in `1:nlevel`. If `h` is included, return a [`MultiDomainGrid`](@ref). Otherwise,
return a [`MultiDomainExtents`](@ref) and the grid step will be automatically computed when
passed to [`Fluid`](@ref).
"""
fluid_grid(; h=nothing, xlims, ylims, nlevel) = fluid_grid(h, xlims, ylims, nlevel)
fluid_grid(h::Nothing, xlims, ylims, nlevel) = MultiDomainExtents(; xlims, ylims, nlevel)
fluid_grid(h, xlims, ylims, nlevel) = MultiDomainGrid(; h, xlims, ylims, nlevel)

"""
    GridVelocity(; center, vel, angle, angular_vel)

The current motion and orientation of the grid relative to the global frame.

# Keywords
- `center`: The center of rotation.
- `vel`: The grid's velocity in the global frame at the center of rotation.
- `angle`: The grid's orientation relative to the global frame in radians.
- `angular_vel`: The time rate of change of `angle`.
"""
@kwdef struct GridVelocity
    center::SVector{2,Float64}
    vel::SVector{2,Float64}
    angle::Float64
    angular_vel::Float64
end

"""
    GridMotion

The motion of the fluid discretization relative to the global frame in which the freestream
velocity is expressed. Can be one of:
- [`StaticGrid`](@ref)
- [`MovingGrid`](@ref)
"""
abstract type GridMotion end

"""
    StaticGrid() :: GridMotion

A grid that is static in the global frame.
"""
struct StaticGrid <: GridMotion end

"""
    MovingGrid(f) :: GridMotion

A grid that moves in global frame. `f(t)` should return a [`GridVelocity`](@ref) that
expresses the current motion of the grid.
"""
struct MovingGrid <: GridMotion
    motion::TimeFunc{GridVelocity}
end

(m::MovingGrid)(t) = m.motion(t)

"""
    Fluid(; grid, Re, freestream_vel, [grid_motion])

# Keywords
- `grid::Union{MultiDomainGrid,MultiDomainExtents}`: The discretization of the fluid. See
  [`fluid_grid`](@ref).
- `Re`: Reynolds number.
- `freestream_vel(t) -> (vx, vy)`: Freestream velocity as a function of time.
- `grid_motion::GridMotion`: Whether the grid is moving relative to the global frame.
"""
@kwdef struct Fluid{M<:GridMotion}
    grid::MultiDomainGrid
    Re::Float64
    freestream_vel::TimeFunc{SVector{2,Float64}} = t -> (0.0, 0.0)
    grid_motion::M = StaticGrid()
    function Fluid(grid, Re, freestream_vel, grid_motion::M) where {M}
        new{M}(grid, Re, freestream_vel, grid_motion)
    end
end

function Fluid(extents::MultiDomainExtents, Re::Float64, args...)
    h = default_gridstep(Re)

    (; xlims, ylims, nlevel) = extents
    grid = MultiDomainGrid(; h, xlims, ylims, nlevel)
    Fluid(grid, Re, args...)
end

default_gridstep(Re) = floor(2 / Re; sigdigits=1)

"""
    AbstractScheme

A time-stepping scheme.
"""
abstract type AbstractScheme end

"""
    CNAB(; dt, n=2)

An `n`-step Crank-Nicolson/Adams-Bashforth timestepping scheme with time step `dt`.
"""
struct CNAB <: AbstractScheme
    dt::Float64
    β::Vector{Float64}
    function CNAB(; dt, n=2)
        if n != 2
            throw(DomainError("only 2-step CNAB is currently supported"))
        end
        new(dt, [1.5, -0.5])
    end
end

timestep(scheme::CNAB) = scheme.dt

"""
    default_scheme([T,] grid; Umax, [cfl])

A default [`AbstractScheme`](@ref) to hit a target `cfl` number.

# Arguments
- `T::Type{<:AbstractScheme}`: The type of scheme to return.
- `grid::Union{CartesianGrid,MultiDomainGrid}`: The discretization of the fluid.
- `Umax`: The maximum fluid velocity relative to the discretization.
- `cfl`: Target CFL number.
"""
default_scheme(grid; kw...) = default_scheme(CNAB, grid; kw...)

function default_scheme(::Type{CNAB}, grid; Umax, cfl=0.1)
    h = gridstep(grid)
    dt = floor(cfl * h / Umax; sigdigits=1)
    CNAB(; dt)
end

"""
    Problem(; fluid::Fluid, bodies::Bodies, scheme::AbstractScheme)
    Problem(fluid::Fluid, bodies::Bodies, scheme::AbstractScheme)

A description of the immersed boundary problem to solve.
"""
@kwdef struct Problem{S<:AbstractScheme,M}
    fluid::Fluid{M}
    bodies::Bodies
    scheme::S
end

"""
    timestep(prob::Problem)
    timestep(scheme::AbstractScheme)

The time between consecutive time steps.
"""
timestep(prob::Problem) = timestep(prob.scheme)

gridstep(prob::Problem) = gridstep(prob.fluid.grid)

"""
    State(prob::Problem; t0=0.0)

The state of the `prob` system at a certain time step. `t0` specifies the time at
`state.index == 1`.
"""
mutable struct State{S,M}
    prob::Problem{S,M}
    index::Int  # Index of the current time step
    t::Float64  # Current state time
    t0::Float64  # Initial state time when index == 1
    q::Matrix{Float64}  # Velocity flux
    q0::Matrix{Float64}  # Base velocity flux
    Γ::Matrix{Float64}  # Circulation
    ψ::Matrix{Float64}  # Streamfunction
    nonlin::Vector{Matrix{Float64}}  # Memory of nonlinear terms
    panels::PanelStateF64  # Structural panels
    F̃b::Vector{Float64}  # FIXME: Document correctly
    fb::Matrix{Float64}
    freestream_vel::SVector{2,Float64}  # [ux, uy] freestream velocity
    function State(prob::Problem{S,M}; t0=0.0) where {S,M}
        state = new{S,M}(prob)

        grid = prob.fluid.grid
        nlevel = grid.nlevel
        (; nx, ny, nu, nq, nΓ) = grid.inds

        npanel = n_panels(prob.bodies)

        state.prob = prob
        state.t0 = t0
        state.q = zeros(nq, nlevel)
        state.q0 = zeros(nq, nlevel)
        state.Γ = zeros(nΓ, nlevel)
        state.ψ = zeros(nΓ, nlevel)
        state.nonlin = _nonlin(prob)
        state.panels = PanelState(npanel)
        state.F̃b = zeros(2 * npanel)
        state.fb = zeros(npanel, 2)
        state.freestream_vel = zero(SVector{2})

        set_time_index!(state, 0)

        state
    end
end

function _nonlin(prob::Problem{CNAB})
    grid = prob.fluid.grid
    (; nΓ) = grid.inds
    nlevel = grid.nlevel
    [zeros(nΓ, nlevel) for _ in 1:length(prob.scheme.β)]
end

_nonlinear(_) = Matrix{Float64}[]

function update_vars!(state::State)
    prob = state.prob
    dt = timestep(prob)
    h = gridstep(prob)
    F̃b = reshape(state.F̃b, :, 2)

    @. state.fb = F̃b * h / dt

    nothing
end

function Base.:(==)(a::State, b::State)
    a.prob === b.prob &&
        a.index === b.index &&
        a.t === b.t &&
        a.t0 === b.t0 &&
        a.q == b.q &&
        a.q0 == b.q0 &&
        a.Γ == b.Γ &&
        a.ψ == b.ψ &&
        a.nonlin == b.nonlin &&
        a.panels == b.panels &&
        a.F̃b == b.F̃b &&
        a.fb == b.fb &&
        a.freestream_vel == b.freestream_vel
end

inc_time_index!(state::State) = set_time_index!(state, state.index + 1)

function set_time_index!(state::State, index::Int)
    state.index = index
    state.t = state.t0 + timestep(state.prob) * (index - 1)
end

"""
    x_velocity(state, [levels])

Return the x velocity array of `state` on subdomain `levels`.
"""
x_velocity(args...) = _velocity(Val(1), args...)
x_velocity!(out, args...) = _velocity!(out, Val(1), args...)
GridPoints(::typeof(x_velocity)) = GridU()
GridPoints(::typeof(x_velocity!)) = GridU()

"""
    y_velocity(state, [levels])

Return the y velocity array of `state` on subdomain `levels`.
"""
y_velocity(args...) = _velocity(Val(2), args...)
y_velocity!(out, args...) = _velocity!(out, Val(2), args...)
GridPoints(::typeof(y_velocity)) = GridV()
GridPoints(::typeof(y_velocity!)) = GridV()

function _velocity(::Val{axis}, state::State, args...) where {axis}
    grid = state.prob.fluid.grid
    pts = (GridU(), GridV())[axis]
    u = Array{Float64}(undef, gridsize(grid, pts))
    _velocity!(u, Val(axis), state, args...)
end

function _velocity!(u::AbstractArray, ::Val{axis}, state::State, levels=:) where {axis}
    grid = state.prob.fluid.grid
    q = split_flux(state.q, grid.inds)[axis]
    q0 = split_flux(state.q0, grid.inds)[axis]

    for lev in (1:(grid.nlevel))[levels]
        hc = gridstep(grid, lev)
        @. u[:, :, lev] = (q[:, :, lev] + q0[:, :, lev]) / hc
    end

    u
end

"""
    vorticity(state, [levels])

Return the vorticity array of `state` on subdomain `levels`.
"""
function vorticity(state::State, args...)
    grid = state.prob.fluid.grid
    ω = Array{Float64}(undef, gridsize(grid, GridΓ()))
    vorticity!(ω, state, args...)
end

function vorticity!(ω::AbstractArray, state::State, levels=:)
    grid = state.prob.fluid.grid
    Γ = unflatten_circ(state.Γ, grid.inds, levels)

    for lev in (1:(grid.nlevel))[levels]
        hc = gridstep(grid, lev)
        @. ω[:, :, lev] = Γ[:, :, lev] / hc^2
    end

    ω
end

GridPoints(::typeof(vorticity)) = GridΓ()
GridPoints(::typeof(vorticity!)) = GridΓ()

"""
    coords(grid, func, ...)

Coordinates of the quantity `func` on `grid`. `func` can be one of the following (or the
in-place `!` versions):
- [`x_velocity`](@ref)
- [`y_velocity`](@ref)
- [`vorticity`](@ref)
"""
coords(grid, func, args...) = coords(grid, GridPoints(func), args...)

"""
    boundary_pos(state)
    boundary_pos(state, inds)
    boundary_pos(state, body::Symbol)

Return the positions of each panel point. In the first form, return all of the points. In
the second form, return the points at indices `inds`. In the last form, return the points
for the body named `body`.
"""
boundary_pos(args...) = _boundary_val(boundary_pos, args...)
boundary_pos!(out, args...) = _boundary_val!(out, boundary_pos, args...)
_get_bndry(::typeof(boundary_pos), state::State) = state.panels.xb
_get_bndry(::typeof(boundary_pos), state::State, inds) = @view state.panels.xb[inds, :]

"""
    boundary_len(state)
    boundary_len(state, inds)
    boundary_len(state, body::Symbol)

Return the lengths of each panel. See [`boundary_pos`](@ref) for the meaning of the parameters.
"""
boundary_len(args...) = _boundary_val(boundary_len, args...)
boundary_len!(out, args...) = _boundary_val!(out, boundary_len, args...)
_get_bndry(::typeof(boundary_len), state::State) = state.panels.ds
_get_bndry(::typeof(boundary_len), state::State, inds) = @view state.panels.ds[inds]

"""
    boundary_vel(state)
    boundary_vel(state, inds)
    boundary_vel(state, body::Symbol)

Return the velocities of each panel. See [`boundary_pos`](@ref) for the meaning of the parameters.
"""
boundary_vel(args...) = _boundary_val(boundary_vel, args...)
boundary_vel!(out, args...) = _boundary_val!(out, boundary_vel, args...)
_get_bndry(::typeof(boundary_vel), state::State) = state.panels.ub
_get_bndry(::typeof(boundary_vel), state::State, inds) = @view state.panels.ub[inds, :]

"""
    boundary_(state)
    boundary_force(state, inds)
    boundary_force(state, body::Symbol)

Return the force on each panel by the fluid. See [`boundary_pos`](@ref) for the meaning of
the parameters.
"""
boundary_force(args...) = _boundary_val(boundary_force, args...)
boundary_force!(out, args...) = _boundary_val!(out, boundary_force, args...)
_get_bndry(::typeof(boundary_force), state::State) = state.fb
_get_bndry(::typeof(boundary_force), state::State, inds) = @view state.fb[inds, :]

_boundary_val(func, state::State) = _get_bndry(func, state)
_boundary_val(func, state::State, inds) = _get_bndry(func, state, inds)
function _boundary_val(func, state::State, body::Symbol)
    _get_bndry(func, state, panel_range(state.prob.bodies, body))
end

_boundary_val!(out, args...) = copy!(out, _boundary_val(args...))

"""
    boundary_total_force(state)
    boundary_total_force(state, inds)
    boundary_total_force(state, body::Symbol)

The total `[x, y]` force. See [`boundary_pos`](@ref) for the meaning of the parameters.
"""
function boundary_total_force(state::State, args...)
    fb = boundary_force(state, args...)
    SVector{2}(sum(fb; dims=1))
end

function boundary_total_force!(fb, state::State, args...)
    copy!(fb, boundary_total_force(state, args...))
end

_arraysize(prob::Problem, ::typeof(boundary_pos!), npanel) = (npanel, 2)
_arraysize(prob::Problem, ::typeof(boundary_len!), npanel) = (npanel,)
_arraysize(prob::Problem, ::typeof(boundary_vel!), npanel) = (npanel, 2)
_arraysize(prob::Problem, ::typeof(boundary_force!), npanel) = (npanel, 2)
_arraysize(prob::Problem, ::typeof(boundary_total_force!), npanel) = (2,)

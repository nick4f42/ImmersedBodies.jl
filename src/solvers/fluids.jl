struct FreestreamFlow <: FluidConditions
    velocity::FunctionWrapper{SVector{2,Float64},Tuple{Float64}} # t -> [ux, uy]
    Re::Float64
end

"""
    FreestreamFlow([velocity]; Re) :: FluidConditions

A flow with velocity `velocity(t) = (ux, uy)` (defaults to zero) and Reynold's number `Re`.
"""
FreestreamFlow(velocity=t -> (0.0, 0.0); Re) = FreestreamFlow(velocity, Re)

function Base.show(io::IO, ::MIME"text/plain", flow::FreestreamFlow)
    print(io, "FreestreamFlow with Re=", flow.Re)
    return nothing
end

"""
    UniformGrid(h, xspan, yspan) :: FluidDiscretization

A uniform grid with spacing `h` on the specified x and y limits.

# Arguments
- `h`: The grid step.
- `xspan = (xmin, xmax)`: Extents of the grid along the x axis.
- `yspan = (ymin, ymax)`: Extents of the grid along the x axis.
"""
struct UniformGrid <: FluidDiscretization
    h::Float64 # Grid cell size
    xs::LinRange{Float64,Int} # x coordinates
    ys::LinRange{Float64,Int} # y coordinates
    function UniformGrid(h::Float64, span::Vararg{NTuple{2,Number},2})
        xs, ys = (x0:h:x1 for (x0, x1) in span)
        return new(h, xs, ys)
    end
    function UniformGrid(
        h::Float64, corner::NTuple{2,AbstractFloat}, counts::NTuple{2,Integer}
    )
        xs, ys = (x0 .+ LinRange(0, h * (n - 1), n) for (x0, n) in zip(corner, counts))
        return new(h, xs, ys)
    end
end

# Default to grid Re of 2
default_gridstep(flow::FreestreamFlow) = 2 / flow.Re

"""
    UniformGrid(flow::FreestreamFlow, xspan, yspan) :: FluidDiscretization

A uniform grid with a default gridstep based on `flow`.
"""
function UniformGrid(flow::FreestreamFlow, span...)
    h = floor(default_gridstep(flow); sigdigits=1)
    return UniformGrid(h, span...)
end

gridstep(grid::UniformGrid) = grid.h

"""
    xycoords(grid::UniformGrid) -> (xs, ys)

Return `AbstractRange`s of the x and y coordinates in `grid`.
"""
xycoords(grid::UniformGrid) = (grid.xs, grid.ys)

function Base.show(io::IO, ::MIME"text/plain", grid::UniformGrid)
    (x1, x2), (y1, y2) = map(extrema, xycoords(grid))
    print(io, "UniformGrid:")

    xspan, yspan = map(extrema, xycoords(grid))
    if get(io, :compact, false)
        print(io, " gridstep=", gridstep(grid), ", xspan=", xspan, ", yspan=", yspan)
    else
        print(io, "\n  gridstep = ", grid.h)
        print(io, "\n     xspan = ", xspan)
        print(io, "\n     yspan = ", yspan)
    end
    return nothing
end

"""
    scale(grid::UniformGrid, k) :: UniformGrid

Scale a uniform grid about its center by a factor `k`.
"""
function scale(grid::UniformGrid, k)
    coords = xycoords(grid)

    # Negative most coordinate
    corner = map(coords) do r
        x0, x1 = extrema(r)
        c = (x0 + x1) / 2

        k * (x0 - c) + c
    end

    counts = map(length, coords)

    h = k * grid.h
    return UniformGrid(h, corner, counts)
end

"""
    MultiLevelGrid(base::UniformGrid, nlevels::Int) :: FluidDiscretization

A combination of `base` scaled at `nlevels` different factors of two.
"""
struct MultiLevelGrid <: FluidDiscretization
    base::UniformGrid
    nlevels::Int
end

"""
    baselevel(grid::MultiLevelGrid) :: UniformGrid

The base discretization of the multi-level discretization.
"""
baselevel(grid::MultiLevelGrid) = grid.base

"""
    nlevels(grid::MultiLevelGrid)

The amount of levels in the multi-level discretization.
"""
nlevels(grid::MultiLevelGrid) = grid.nlevels

"""
    sublevel(grid::MultiLevelGrid, lev::Int) :: UniformGrid

The `lev`th level of the multi-level discretization.
"""
sublevel(grid::MultiLevelGrid, lev::Int) = scale(grid.base, 2.0^(lev - 1))

"""
    gridstep(domain::MultiUniformGrid, lev=1)

The grid step of the `lev`th level.
"""
gridstep(grid::MultiLevelGrid) = gridstep(baselevel(grid))
gridstep(grid::MultiLevelGrid, lev::Int) = gridstep(grid.base) * 2.0^(lev - 1)

function Base.show(io::IO, ::MIME"text/plain", grids::MultiLevelGrid)
    nlev = nlevels(grids)
    print(io, nlev, "-level MultiLevelGrid:")

    if get(io, :compact, false)
        basestep = gridstep(baselevel(grids))
        xspan, yspan = map(extrema, xycoords(sublevel(grids, nlev)))
        print(io, " base_gridstep=", basestep, ", xspan=", xspan, ", yspan=", yspan)
        return nothing
    end

    let grid = baselevel(grids), (xspan, yspan) = map(extrema, xycoords(grid))
        print(io, "\n  base level:")
        print(io, " gridstep=", gridstep(grid), ", xspan=", xspan, ", yspan=", yspan)
    end
    if nlev > 1
        let grid = sublevel(grids, nlev), (xspan, yspan) = map(extrema, xycoords(grid))
            print(io, "\n  last level:")
            print(io, " gridstep=", gridstep(grid), ", xspan=", xspan, ", yspan=", yspan)
        end
    end

    return nothing
end

struct PsiOmegaGridIndexing
    nx::Int # Number of x grid cells
    ny::Int # Number of y grid cells
    nu::Int # Number of x flux points
    nq::Int # Number of total flux points
    nΓ::Int # Number of circulation points

    # Boundary index offsets
    L::Int # Left (-x)
    R::Int # Right (+x)
    B::Int # Bottom (-y)
    T::Int # Top (+y)

    function PsiOmegaGridIndexing(grid::UniformGrid)
        nx = length(grid.xs) - 1
        ny = length(grid.ys) - 1

        nu = (nx + 1) * ny
        nv = nx * (ny + 1)
        nq = nu + nv
        nΓ = (nx - 1) * (ny - 1)

        L = 0
        R = ny + 1
        B = 2 * (ny + 1)
        T = 2 * (ny + 1) + nx + 1

        return new(nx, ny, nu, nq, nΓ, L, R, B, T)
    end
end

"""
    PsiOmegaFluidGrid(flow::FreestreamFlow, grid; [scheme], [frame]) :: AbstractFluid

Fluid simulated with the streamfunction-vorticity formulation.

# Arguments
- `flow::FreestreamFlow`: The prescribed flow conditions.
- `grid::MultiLevelGrid`: The fluid domain and discretization.
- `scheme::AbstractScheme`: The timestepping scheme to use.
- `frame::AbstractFrame`: The reference frame that grid coordinates are resolved in.
"""
struct PsiOmegaFluidGrid{S<:AbstractScheme,F<:AbstractFrame} <: AbstractFluid
    flow::FreestreamFlow
    grid::MultiLevelGrid
    scheme::S
    frame::F
    gridindex::PsiOmegaGridIndexing
    function PsiOmegaFluidGrid(
        flow::FreestreamFlow, grid::MultiLevelGrid; scheme::S, frame::F=GlobalFrame()
    ) where {S,F}
        if frame isa DiscretizationFrame
            throw(ArgumentError("invalid reference frame given (circular definition)"))
        end
        gridindex = PsiOmegaGridIndexing(baselevel(grid))
        return new{S,F}(flow, grid, scheme, frame, gridindex)
    end
end

const DEFAULT_LEVEL_COUNT = 5

"""
    PsiOmegaFluidGrid(flow::FreestreamFlow, grid::UniformGrid; kw...) :: AbstractFluid

Create a fluid grid from a base `grid` and $DEFAULT_LEVEL_COUNT levels.
"""
function PsiOmegaFluidGrid(flow::FreestreamFlow, grid::UniformGrid; kw...)
    grids = MultiLevelGrid(grid, DEFAULT_LEVEL_COUNT)
    return PsiOmegaFluidGrid(flow, grids; kw...)
end

conditions(fluid::PsiOmegaFluidGrid) = fluid.flow
discretized(fluid::PsiOmegaFluidGrid) = fluid.grid
timestep_scheme(fluid::PsiOmegaFluidGrid) = fluid.scheme

function _show(io::IO, fluid::PsiOmegaFluidGrid, prefix)
    if get(io, :compact, false)
        print(io, prefix, "PsiOmegaFluidGrid: Re=", fluid.Re, " frame=", fluid.frame)
        return nothing
    end

    ioc = IOContext(io, :compact => true)
    mime = MIME("text/plain")
    print(io, prefix, "PsiOmegaFluidGrid:")

    print(ioc, '\n', prefix, "    flow = ")
    show(ioc, mime, fluid.flow)

    print(ioc, '\n', prefix, "    grid = ")
    show(ioc, mime, fluid.grid)

    print(ioc, '\n', prefix, "  scheme = ")
    show(ioc, mime, fluid.scheme)

    print(ioc, '\n', prefix, "   frame = ")
    show(ioc, mime, fluid.frame)

    return nothing
end

# The x and y coordinates of the gridpoints where x-flux is stored.
xflux_ranges(grid::UniformGrid) = (grid.xs, midpoints(grid.ys))

# The x and y coordinates of the gridpoints where y-flux is stored.
yflux_ranges(grid::UniformGrid) = (midpoints(grid.xs), grid.ys)

# The x and y coordinates of the gridpoints where circulation and streamfunction are stored.
circ_ranges(grid::UniformGrid) = (grid.xs[2:(end - 1)], grid.ys[2:(end - 1)])

midpoints(r::AbstractVector) = @views @. (r[begin:(end - 1)] + r[(begin + 1):end]) / 2

function flatten_circ(Γ::AbstractArray, gridindex::PsiOmegaGridIndexing)
    # Flatten the x and y dimensions of the circulation array into one
    return reshape(Γ, gridindex.nΓ, size(Γ)[3:end]...)
end

function unflatten_circ(Γ::AbstractArray, gridindex::PsiOmegaGridIndexing)
    # Expand the x and y dimensions of a flattened circulation array
    (; nx, ny) = gridindex
    return reshape(Γ, nx - 1, ny - 1, size(Γ)[2:end]...)
end

function unflatten_circ(Γ::AbstractMatrix, gridindex::PsiOmegaGridIndexing, lev)
    # Unflatten the circulation on the lev'th sublevel
    return unflatten_circ(view(Γ, :, lev), gridindex)
end

function split_flux(q::AbstractArray, gridindex::PsiOmegaGridIndexing)
    # Extract the x and y flux components from a flattened flux array
    (; nx, ny, nu) = gridindex

    uflat = @view q[1:nu, ..]
    vflat = @view q[(nu + 1):end, ..]

    dims = size(q)[2:end]

    u = @views reshape(uflat, nx + 1, ny, dims...)
    v = @views reshape(vflat, nx, ny + 1, dims...)
    return (u, v)
end

function split_flux(q::AbstractMatrix, gridindex::PsiOmegaGridIndexing, lev)
    # Extract the x and y flux components on the lev'th sublevel
    return split_flux(view(q, :, lev), gridindex)
end

function body_segment_length(fluid::PsiOmegaFluidGrid)
    return 2 * (gridstep ∘ baselevel ∘ discretized)(fluid)
end

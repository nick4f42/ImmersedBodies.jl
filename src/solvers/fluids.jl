"""
    UniformGrid((xlims, ylims), h) :: FluidDiscretization

A uniform grid with spacing `h` on the specified x and y limits.

# Arguments
- `xlims = (xmin, xmax)`: Extents of the grid along the x axis.
- `ylims = (ymin, ymax)`: Extents of the grid along the x axis.
- `h`: The grid step.
"""
struct UniformGrid <: FluidDiscretization
    h::Float64 # Grid cell size
    xs::LinRange{Float64,Int} # x coordinates
    ys::LinRange{Float64,Int} # y coordinates
    function UniformGrid(h::Float64, lims::Vararg{NTuple{2,AbstractFloat},2})
        xs, ys = (x0:h:x1 for (x0, x1) in lims)
        return new(h, xs, ys)
    end
    function UniformGrid(
        h::Float64, corner::NTuple{2,AbstractFloat}, counts::NTuple{2,Integer}
    )
        xs, ys = (x0 .+ LinRange(0, h * (n - 1), n) for (x0, n) in zip(corner, counts))
        return new(h, xs, ys)
    end
end

"""
    gridstep(grid::UniformGrid)

The grid cell spacing.
"""
gridstep(grid::UniformGrid) = grid.h

"""
    xycoords(grid::UniformGrid) -> (xs, ys)

Return `AbstractRange`s of the x and y coordinates in `grid`.
"""
xycoords(grid::UniformGrid) = (grid.xs, grid.ys)

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
    gridstep(domain::MultiUniformGrid, lev::Int)

The grid step of the `lev`th level.
"""
gridstep(grid::MultiLevelGrid, lev::Int) = gridstep(grid.base) * 2.0^(lev - 1)

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
    PsiOmegaFluidGrid(scheme, grid, freestream, frame=GlobalFrame(); Re) :: AbstractFluid

Fluid simulated with the streamfunction-vorticity formulation.

# Arguments
- `scheme::AbstractScheme`: The timestepping scheme to use.
- `grid::MultiLevelGrid`: The fluid domain to simulate.
- `freestream`: The freestream velocity `t -> [ux, uy]` in the lab frame.
- `frame::AbstractFrame`: The reference frame that grid coordinates are resolved in.
- `Re::Float64`: The Reynold's number of the flow.
"""
struct PsiOmegaFluidGrid{S<:AbstractScheme,F<:AbstractFrame} <: AbstractFluid
    scheme::S
    grid::MultiLevelGrid
    freestream::FunctionWrapper{SVector{2,Float64},Tuple{Float64}} # t -> [ux, uy]
    frame::F
    Re::Float64
    gridindex::PsiOmegaGridIndexing
    function PsiOmegaFluidGrid(
        scheme::S, grid::MultiLevelGrid, freestream, frame::F=GlobalFrame(); Re
    ) where {S,F}
        if frame isa DiscretizationFrame
            throw(ArgumentError("invalid reference frame given (circular definition)"))
        end
        gridindex = PsiOmegaGridIndexing(grid.base)
        return new{S,F}(scheme, grid, freestream, frame, Re, gridindex)
    end
end

discretized(fluid::PsiOmegaFluidGrid) = fluid.grid

timestep_scheme(fluid::PsiOmegaFluidGrid) = fluid.scheme

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

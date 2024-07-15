abstract type GridKind end
struct Primal <: GridKind end
struct Dual <: GridKind end

abstract type GridLocation{K<:GridKind} end
struct Node{K} <: GridLocation{K} end
struct Edge{K} <: GridLocation{K}
    i::Int
end

const Loc_u = Edge{Primal}
const Loc_ω = Edge{Dual}

"""
    Grid(; h, n, x0, levels)

A Cartesian grid with uniform spacing `h`, `n` cells along each dimension, and lower corner
position `x0`.  `n` is rounded up to the next multiple of 4.
"""
@kwdef struct Grid{N,T<:AbstractFloat}
    h::T
    n::SVector{N,Int}
    x0::SVector{N,T}
    levels::Int
    function Grid(h::T, n, x0, levels) where {T}
        let n = @. 4 * cld(n, 4)
            new{length(n),T}(h, n, x0, levels)
        end
    end
end

gridcorner(grid::Grid) = grid.x0
gridcorner((; x0, h, n)::Grid, level::Integer) = x0 + h * n * (1 - 2^(level - 1)) / 2

gridstep(grid::Grid) = grid.h
gridstep(grid::Grid, level::Integer) = grid.h * 2^(level - 1)

function coord(grid::Grid, loc, I::SVector{N,<:Integer}, args...) where {N}
    x0 = gridcorner(grid, args...)
    h = gridstep(grid, args...)
    x0 + h * (I + _cellcoord(loc, Val(N)))
end

coord(grid, loc, I, args...) = coord(grid, loc, SVector(Tuple(I)), args...)

_cellcoord((; i)::Edge{Primal}, ::Val{N}) where {N} = SVector(ntuple(≠(i), N)) / 2
_cellcoord((; i)::Edge{Dual}, ::Val{N}) where {N} = SVector(ntuple(==(i), N)) / 2

struct IncludeBoundary end
struct ExcludeBoundary end

function cell_axes(n::SVector{N}, loc, ::IncludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (0:n[j]) : (0:n[j]-1), Val(N))
end

function cell_axes(n::SVector{N}, loc, ::ExcludeBoundary) where {N}
    ntuple(j -> _on_bndry(loc, j) ? (1:n[j]-1) : (0:n[j]-1), Val(N))
end

cell_axes(grid::Grid, args...) = cell_axes(grid.n, args...)

_on_bndry((; i)::Edge{Primal}, j) = i == j
_on_bndry((; i)::Edge{Dual}, j) = i ≠ j

function boundary_axes(n::SVector{N}, loc::Edge) where {N}
    a = cell_axes(n, loc, IncludeBoundary())
    (SArray ∘ map)(CartesianIndices(SOneTo.((2, N)))) do index
        dir, j = Tuple(index)
        if _on_bndry(loc, j)
            let Iⱼ = (a[j][begin], a[j][end])[dir]
                setindex(a, Iⱼ:Iⱼ, j)
            end
        else
            ntuple(_ -> 1:0, N)
        end
    end
end

function boundary_axes(n::SVector{N}, loc::Type{<:Edge}; dims=ntuple(identity, N)) where {N}
    map(i -> boundary_axes(n, loc(i)), dims)
end

boundary_axes(grid::Grid, args...; kw...) = boundary_axes(grid.n, args...; kw...)

"""
    IrrotationalFlow

Specifies a flow velocity with zero discrete curl.
"""
abstract type IrrotationalFlow end

"""
    UniformFlow(u)

A flow with uniform freestream velocity `u(t)`.
"""
struct UniformFlow{U} <: IrrotationalFlow
    u::U
end

"""
    AbstractBody

A body that interacts with the fluid.  Bodies specify a set of points, and prescribe the
flow velocity in a small region near each point.
"""
abstract type AbstractBody end

"""
    PrescribedBody

A body whose shape is prescribed independently from the fluid flow.
"""
abstract type PrescribedBody <: AbstractBody end

"""
    StaticBody(xb::AbstractVector{<:SVector})

A body of time-constant points `xb`.
"""
struct StaticBody{A<:AbstractVector{<:SVector}} <: PrescribedBody
    xb::A
end

struct IBProblem{N,T,B<:AbstractBody,U<:IrrotationalFlow}
    grid::Grid{N,T}
    bodies::Vector{B}
    Re::T
    u0::U
end

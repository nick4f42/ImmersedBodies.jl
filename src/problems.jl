abstract type GridKind end
struct Primal <: GridKind end
struct Dual <: GridKind end

struct AllAxes end

abstract type GridLocation{K<:GridKind} end
struct Node{K} <: GridLocation{K} end
struct Edge{K,I<:Union{Int,AllAxes}} <: GridLocation{K}
    i::I
end
Edge{K}(i::I) where {K,I} = Edge{K,I}(i)
Edge{K,AllAxes}() where {K} = Edge{K,AllAxes}(AllAxes())

axisindex(x::Edge) = x.i
setaxis(x::Edge{K}, i) where {K} = Edge{K}(i)

const Loc_u = Edge{Primal,Int}
const LocVec_u = Edge{Primal,AllAxes}
const Loc_ω = Edge{Dual,Int}
const LocVec_ω = Edge{Dual,AllAxes}

const OneArrayOrTuple{T,N} = Union{AbstractArray{T,N},NTuple{N,AbstractArray{T,N}}}
struct GridArray{L<:GridLocation,T,N,A<:OneArrayOrTuple{T,N}}
    loc::L
    a::A
end

unwrap(a::GridArray) = a.a
array_axis(a::GridArray) = (a.a, axisindex(a))
axisindex(a::GridArray) = axisindex(a.loc)

axispairs(a::GridArray{<:Any,NTuple{N}}) where {N} = ntuple(i -> (i, a.a[i]), N)
axispairs(a::GridArray{LocVec_ω,<:AbstractArray}) = ((3, a.a))

Base.similar(a::GridArray) = GridArray(a.loc, similar(a.a))
Base.similar(a::GridArray{L,T,N,Tuple}) where {L,T,N} = GridArray(a.loc, similar.(a.a))

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

function coord(grid::Grid, loc, I::SVector{N}, args...) where {N}
    x0 = gridcorner(grid, args...)
    h = gridstep(grid, args...)
    x0 + h * (I + _cellcoord(loc, Val(N)))
end

coord(grid, loc, I, args...) = coord(grid, loc, SVector(Tuple(I)), args...)

_cellcoord((; i)::Edge{Primal,Int}, ::Val{N}) where {N} = SVector(ntuple(≠(i), N))//2
_cellcoord((; i)::Edge{Dual,Int}, ::Val{N}) where {N} = SVector(ntuple(==(i), N))//2

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

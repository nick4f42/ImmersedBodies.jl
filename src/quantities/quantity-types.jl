struct ArrayQuantity{F} <: Quantity
    f::F
end

const NumberOrArray = Union{Number,AbstractArray{<:Number}}

(qty::ArrayQuantity)(state) = qty.f(state)

struct ArrayValues{S<:AbstractVector,A<:NumberOrArray} <: AbstractVector{A}
    times::S
    values::Vector{A}
end

Base.size(vals::ArrayValues) = size(vals.values)
Base.getindex(vals::ArrayValues, i) = vals.values[i]
timevalue(vals::ArrayValues) = vals.times

const CoordRange = LinRange{Float64,Int}

struct GridQuantity{F,N} <: Quantity
    f::F
    coords::NTuple{N,CoordRange}
end

coordinates(qty::GridQuantity) = qty.coords

struct GridValue{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    array::A
    coords::NTuple{N,CoordRange}
end

Base.size(val::GridValue) = size(val.array)
Base.getindex(val::GridValue, i...) = val.array[i...]
coordinates(val::GridValue) = val.coords

function (qty::GridQuantity)(state)
    array = qty.f(state)::AbstractArray
    return GridValue(array, qty.coords)
end

struct GridValues{S,N,A<:AbstractArray,V<:GridValue} <: AbstractVector{V}
    times::S
    values::Vector{A}
    coords::NTuple{N,CoordRange}
    function GridValues(
        times::S, values::AbstractVector{A}, coords::NTuple{N}
    ) where {S,T,N,A<:AbstractArray{T,N}}
        V = GridValue{T,N,A}
        return new{S,N,A,V}(times, values, coords)
    end
end

Base.size(vals::GridValues) = size(vals.values)
Base.getindex(vals::GridValues, i) = GridValue(vals.values[i], vals.coords)
timevalue(vals::GridValues) = vals.times
coordinates(vals::GridValues) = vals.coords

struct MultiLevelGridQuantity{F,N} <: Quantity
    f::F
    coords::Vector{NTuple{N,CoordRange}}
end

coordinates(qty::MultiLevelGridQuantity) = qty.coords

struct MultiLevelGridValue{N,T,M,A<:AbstractArray{T,M}} <: AbstractArray{T,M}
    array::A
    coords::Vector{NTuple{N,CoordRange}}
end

Base.size(val::MultiLevelGridValue) = size(val.array)
Base.getindex(val::MultiLevelGridValue, i...) = val.array[i...]
coordinates(val::MultiLevelGridValue) = val.coords

function (qty::MultiLevelGridQuantity)(state)
    array = qty.f(state)::AbstractArray
    return MultiLevelGridValue(array, qty.coords)
end

struct MultiLevelGridValues{S,A<:AbstractArray,N,V<:MultiLevelGridValue} <:
       AbstractVector{V}
    times::S
    values::Vector{A}
    coords::Vector{NTuple{N,CoordRange}}
    function MultiLevelGridValues(
        times::S, values::AbstractVector{A}, coords::AbstractVector{<:NTuple{N}}
    ) where {S,T,M,A<:AbstractArray{T,M},N}
        @assert M == N + 1 # one dimension for grid sublevel
        V = MultiLevelGridValue{N,T,M,A}
        return new{S,A,N,V}(times, values, coords)
    end
end

Base.size(vals::MultiLevelGridValues) = size(vals.values)
function Base.getindex(vals::MultiLevelGridValues, i)
    return MultiLevelGridValue(vals.values[i], vals.coords)
end
timevalue(vals::MultiLevelGridValues) = vals.times
coordinates(val::MultiLevelGridValues) = val.coords

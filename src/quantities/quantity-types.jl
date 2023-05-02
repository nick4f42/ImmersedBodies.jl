struct ArrayQuantity{F} <: Quantity
    f::F
end

const NumberOrArray = Union{Number,AbstractArray{<:Number}}

(qty::ArrayQuantity)(state) = qty.f(state)

struct ArrayValues{S<:AbstractVector,A<:NumberOrArray,X<:AbstractVector{A}} <:
       AbstractVector{A}
    times::S
    values::X
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

struct GridValues{S,N,A<:AbstractArray,X<:AbstractVector{A},V<:GridValue} <:
       AbstractVector{V}
    times::S
    values::X
    coords::NTuple{N,CoordRange}
    function GridValues(
        times::S, values::X, coords::NTuple{N}
    ) where {S,T,N,A<:AbstractArray{T,N},X<:AbstractVector{A}}
        V = GridValue{T,N,A}
        return new{S,N,A,X,V}(times, values, coords)
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

struct MultiLevelGridValues{
    S,A<:AbstractArray,N,X<:AbstractVector{A},V<:MultiLevelGridValue
} <: AbstractVector{V}
    times::S
    values::X
    coords::Vector{NTuple{N,CoordRange}}
    function MultiLevelGridValues(
        times::S, values::X, coords::AbstractVector{<:NTuple{N}}
    ) where {S,T,M,A<:AbstractArray{T,M},X<:AbstractVector{A},N}
        @assert M == N + 1 # one dimension for grid sublevel
        V = MultiLevelGridValue{N,T,M,A}
        return new{S,A,N,X,V}(times, values, coords)
    end
end

Base.size(vals::MultiLevelGridValues) = size(vals.values)
function Base.getindex(vals::MultiLevelGridValues, i)
    return MultiLevelGridValue(vals.values[i], vals.coords)
end
timevalue(vals::MultiLevelGridValues) = vals.times
coordinates(val::MultiLevelGridValues) = val.coords

struct ConcatArrayQuantity{F} <: Quantity
    f::F
    dim::Int
end

struct ConcatArrayValue{A<:AbstractArray} <: AbstractVector{A}
    arrays::Vector{A}
    dim::Int
end

Base.size(val::ConcatArrayValue) = size(val.arrays)
Base.getindex(val::ConcatArrayValue, i) = val.arrays[i]

(qty::ConcatArrayQuantity)(state) = ConcatArrayValue(qty.f(state), qty.dim)

struct ConcatArrayValues{S,X,V<:ConcatArrayValue} <: AbstractVector{V}
    times::S
    values::X
    dim::Int
    function ConcatArrayValues(
        times::S, values::X, dim::Int
    ) where {S,T,X<:AbstractVector{<:AbstractVector{T}}}
        V = ConcatArrayValue{T}
        return new{S,X,V}(times, values, dim)
    end
end

Base.size(vals::ConcatArrayValues) = size(vals.values)
Base.getindex(vals::ConcatArrayValues, i) = ConcatArrayValue(vals.values[i], vals.dim)
timevalue(vals::ConcatArrayValues) = vals.times

struct BodyArrayQuantity{F} <: Quantity
    f::F
    dim::Int # Dimension to concatenate along
    bodies::Vector{Int} # Corresponding body indices
end

function BodyArrayQuantity(f, dim::Int, bodies)
    return BodyArrayQuantity(f, dim, convert(Vector{Int}, bodies))
end

bodyindices(qty::BodyArrayQuantity) = qty.bodies

struct BodyArrayValue{A<:AbstractArray} <: AbstractVector{A}
    arrays::Vector{A}
    dim::Int
    bodies::Vector{Int}
end

Base.size(val::BodyArrayValue) = size(val.arrays)
Base.getindex(val::BodyArrayValue, i) = val.arrays[i]
bodyindices(val::BodyArrayValue) = val.bodies

(qty::BodyArrayQuantity)(state) = BodyArrayValue(qty.f(state), qty.dim, qty.bodies)

struct BodyArrayValues{S,X,V<:BodyArrayValue} <: AbstractVector{V}
    times::S
    values::X
    dim::Int
    bodies::Vector{Int}
    function BodyArrayValues(
        times::S, values::X, dim::Int, bodies::AbstractVector{Int}
    ) where {S,T,X<:AbstractVector{<:AbstractVector{T}}}
        V = BodyArrayValue{T}
        return new{S,X,V}(times, values, dim, bodies)
    end
end

Base.size(vals::BodyArrayValues) = size(vals.values)
function Base.getindex(vals::BodyArrayValues, i)
    return BodyArrayValue(vals.values[i], vals.dim, vals.bodies)
end
timevalue(vals::BodyArrayValues) = vals.times
bodyindices(vals::BodyArrayValues) = vals.bodies

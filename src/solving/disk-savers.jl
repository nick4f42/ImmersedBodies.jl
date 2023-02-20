const QUANTITY_TYPE_ATTR = "quantity_type"
const TIME_ATTR = "time"
const COORDS_ATTR = "coords"

abstract type QuantityDiskSaver end

function quantity_values(data::Union{HDF5.Group,HDF5.Dataset})
    @assert haskey(attributes(data), QUANTITY_TYPE_ATTR)
    attr = attributes(data)[QUANTITY_TYPE_ATTR]
    @assert eltype(attr) <: HDF5.FixedString && ndims(attr) == 0

    type = Symbol(read_attribute(data, QUANTITY_TYPE_ATTR))
    return quantity_values(Val(type), data)
end

last_dim_slices(a::AbstractArray) = eachslice(a; dims=ndims(a))
last_dim_slices(v::AbstractVector) = v

"""
    DatasetArray{T,N}

Type-stable wrapper of `HDF5.Dataset` for array types.
"""
struct DatasetArray{T,N} <: AbstractArray{T,N}
    dset::HDF5.Dataset
    DatasetArray(dset::HDF5.Dataset) = new{eltype(dset),ndims(dset)}(dset)
end

Base.size(array::DatasetArray) = size(array.dset)::NTuple{ndims(array),Int}
function Base.getindex(array::DatasetArray{T,N}, indices::Vararg{Any,N}) where {T,N}
    n = sum(i -> !isa(i, Integer), indices; init=0)
    return array.dset[indices...]::(n == 0 ? T : Array{T,n})
end
function Base.getindex(array::DatasetArray, indices...)
    n = ndims(array)
    @assert all(==(1), indices[(n + 1):end])
    return array[indices[1:n]...]
end

struct ArrayDiskSaver <: QuantityDiskSaver
    dset::HDF5.Dataset
end

# size but anything beisdes an array is 0 dimensional
value_shape(_) = ()
value_shape(a::AbstractArray) = size(a)

function create_disk_saver(parent, name, ::ArrayQuantity, timeref::HDF5.Reference, value)
    maxlen = length(parent[timeref])

    dtype = datatype(value)
    space = (value_shape(value)..., maxlen)

    dset = create_dataset(parent, name, dtype, space)
    attributes(dset)[QUANTITY_TYPE_ATTR] = string(nameof(ArrayQuantity))
    attributes(dset)[TIME_ATTR] = timeref

    return ArrayDiskSaver(dset)
end

function update_saver(saver::ArrayDiskSaver, timeindex::Int, value)
    dims = (Colon() for _ in 1:(ndims(saver.dset) - 1))
    saver.dset[dims..., timeindex] = value
    return nothing
end

function quantity_values(::Val{nameof(ArrayQuantity)}, dset::HDF5.Dataset)
    time = read(dset[read_attribute(dset, TIME_ATTR)])
    data = DatasetArray(dset)
    arrays = collect(last_dim_slices(data))
    return ArrayValues(time, arrays)
end

to_range_tuple(r::AbstractRange) = (min=minimum(r), step=step(r), length=length(r))
from_range_tuple(r) = range(r.min; step=r.step, length=r.length)

function create_disk_saver(
    parent, name, qty::GridQuantity, timeref::HDF5.Reference, value::GridValue
)
    maxlen = length(parent[timeref])
    dtype = datatype(value)
    space = (value_shape(value)..., maxlen)

    dset = create_dataset(parent, name, dtype, space)
    attributes(dset)[QUANTITY_TYPE_ATTR] = string(nameof(GridQuantity))
    attributes(dset)[TIME_ATTR] = timeref
    attributes(dset)[COORDS_ATTR] = collect(map(to_range_tuple, qty.coords))

    return ArrayDiskSaver(dset)
end

function quantity_values(::Val{nameof(GridQuantity)}, dset::HDF5.Dataset)
    time = read(dset[read_attribute(dset, TIME_ATTR)])
    data = DatasetArray(dset)
    arrays = collect(last_dim_slices(data))
    coords = map(from_range_tuple, read_attribute(dset, COORDS_ATTR))
    return GridValues(time, arrays, Tuple(coords))
end

function create_disk_saver(
    parent,
    name,
    qty::MultiLevelGridQuantity,
    timeref::HDF5.Reference,
    value::MultiLevelGridValue,
)
    maxlen = length(parent[timeref])
    dtype = datatype(value)
    space = (value_shape(value)..., maxlen)

    dset = create_dataset(parent, name, dtype, space)
    attributes(dset)[QUANTITY_TYPE_ATTR] = string(nameof(MultiLevelGridQuantity))
    attributes(dset)[TIME_ATTR] = timeref

    ranges = reinterpret(reshape, LinRange{Float64,Int}, qty.coords)
    attributes(dset)[COORDS_ATTR] = map(to_range_tuple, ranges)

    return ArrayDiskSaver(dset)
end

function quantity_values(::Val{nameof(MultiLevelGridQuantity)}, dset::HDF5.Dataset)
    time = read(dset[read_attribute(dset, TIME_ATTR)])
    data = DatasetArray(dset)
    arrays = collect(last_dim_slices(data))
    coord_array = map(from_range_tuple, read_attribute(dset, COORDS_ATTR))
    coords = reinterpret(reshape, NTuple{ndims(data) - 2,eltype(coord_array)}, coord_array)
    return MultiLevelGridValues(time, arrays, coords)
end

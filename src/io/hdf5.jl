"""
    HDF5Saver(file, paths...; [mode])

Saves to the HDF5 file, filename, or group `file`. `paths` is pairs of `path => val` that
specifies to save a representation of `val` at `path` in the HDF5 file. If `file` is a
filename, open the file with `mode`. `val` can be an instance of:
- [`Problem`](@ref)
- [`SolutionValues`](@ref)
"""
struct SaveHDF5{F<:Union{String,HDF5.File,HDF5.Group}} <: SolutionSaver
    file::F
    mode::String
    paths::Vector{Pair{String,Any}}  # Pair{String,<:SolutionValues}
end

SaveHDF5(file, paths::AbstractVector; mode="cw") = SaveHDF5(file, mode, paths)
function SaveHDF5(file, paths::Pair...; kw...)
    SaveHDF5(file, collect(Pair{String,Any}, paths); kw...)
end

function fluid_group(prob::Problem, times::Timesteps, funcs)
    SolutionValues(times, map(funcs) do f
        name = rstrip((string ∘ nameof)(f), '!')
        name => MultiDomainValue(f, coords(prob.fluid.grid, f))
    end...)
end

function body_group(
    prob::Problem,
    times::Timesteps,
    funcs;
    bodies::Vector{Symbol}=collect(keys(prob.bodies.byname)),
)
    SolutionValues(
        times,
        collect(
            Pair{String,Any},
            (Iterators.flatten ∘ Iterators.map)(bodies) do body
                panels = panel_range(prob.bodies, body)
                map(funcs) do f
                    dims = _arraysize(prob, f, length(panels))
                    func_name = rstrip((string ∘ nameof)(f), '!')
                    "$body/$func_name" => ArrayValue(Float64, dims) do out, state
                        f(out, state, panels)
                    end
                end
            end,
        ),
    )
end

struct HDF5Saver
    file::HDF5.File
    toclose::Bool  # Whether to close the file after we're done
    savers::Vector{Any}  # Functions f(state) that update the file
end

function HDF5Saver(save::SaveHDF5, file::HDF5.File; toclose, max_timestep::Int)
    # Create a buffer for functions to write their data to.
    buffer = Vector{UInt8}(undef, _buffer_size(save))

    savers = Iterators.map(save.paths) do (path, val)
        _saver(file, path, val; buffer, max_timestep)
    end
    saver_funcs = collect(Iterators.filter(!isnothing, savers))

    HDF5Saver(file, toclose, saver_funcs)
end

function _buffer_size(save::SaveHDF5)
    maximum(save.paths) do (_path, val)
        _buffer_size(val)
    end
end

_buffer_size(::Problem) = 0

function _saver(file, path, prob::Problem; _kw...)
    prob_group = _create_or_get_group(file, path)
    _saver(prob_group, "fluid", prob.fluid)
    _saver(prob_group, "bodies", prob.bodies)
    _saver(prob_group, "scheme", prob.scheme)
    nothing
end

_buffer_size(::Fluid) = 0

function _saver(file, path, fluid::Fluid; _kw...)
    grid = fluid.grid
    h = grid.base.h
    xlims, ylims = map(NamedTuple{(:min, :max)}, extents(grid.base))
    nlevel = grid.nlevel

    _write_or_check_dataset(
        file, path, fill((; grid=(; h, xlims, ylims, nlevel), Re=fluid.Re))
    )
end

_buffer_size(::Bodies) = 0

function _saver(file, path, bodies::Bodies; _kw...)
    bodies_group = _create_or_get_group(file, path)
    for (name, part) in bodies.byname
        body_group = _create_or_get_group(bodies_group, string(name))
        _save(body_group, section_body(part))
    end
    nothing
end

_buffer_size(::PanelSection) = nothing

function _saver(file, path, part::PresetBodySection; _kw...)
    body_group = _create_or_get_group(file, path)
    _save(body_group, part.body)
    _write_or_check_dataset(body_group, "panel_range", fill(_range_tuple(part.panel_range)))
    nothing
end

_buffer_size(::AbstractBody) = nothing

function _saver(file, path, body::AbstractBody; _kw...)
    body_group = _create_or_get_group(file, path)
    _save(body_group, body)
    nothing
end

function _save(group::HDF5.Group, body::PresetBody)
    panels = bodypanels(body)
    _write_or_check_attribute(group, "type", (string ∘ nameof ∘ typeof)(body))
    _write_or_check_dataset(group, "positions", panels.xb)
    _write_or_check_dataset(group, "lengths", panels.ds)
    nothing
end

_buffer_size(::AbstractScheme) = 0

function _saver(file, path, scheme::AbstractScheme; _kw...)
    _write_or_check_dataset(file, path, fill((; dt=timestep(scheme))))
    _write_or_check_attribute(file[path], "type", (string ∘ nameof ∘ typeof)(scheme))
    nothing
end

function _buffer_size(vals::SolutionValues)
    maximum(vals.vals) do (_name, val)
        _buffer_size(val::SolutionValue)
    end
end
_buffer_size(a::ArrayValue) = sizeof(a.type) * prod(a.dims)
_buffer_size(a::MultiDomainValue) = sizeof(Float64) * prod(a.dims)

function _saver(file, path, vals::SolutionValues; buffer, max_timestep::Int)
    group = _create_or_get_group(file, path)
    n_times = _timestep_count(vals.times, max_timestep)

    val_savers = map(vals.vals) do (name, val)
        _val_saver(group, name, val; n_times, buffer)
    end

    time_dspace = dataspace((n_times,), (-1,))
    time_chunk = (100_000,)

    time_dset = _create_or_get_time_dataset(
        group, "_timestep_time", Float64, time_dspace; chunk=time_chunk
    )
    index_dset = _create_or_get_time_dataset(
        group, "_timestep_index", Int, time_dspace; chunk=time_chunk
    )

    function (state::State)
        (; inside, index) = _timestep_in(vals.times, state.index)
        if inside
            time_dset[index] = state.t
            index_dset[index] = state.index
            for saver in val_savers
                saver(state, index)
            end
        end
    end
end

function _val_saver(
    group::HDF5.Group,
    name::String,
    val::ArrayValue;
    n_times::Int,
    buffer::AbstractVector{UInt8},
)
    (; saver) = _array_saver(group, name, val; n_times, buffer)
    saver
end

function _val_saver(
    group::HDF5.Group,
    name::String,
    val::MultiDomainValue;
    n_times::Int,
    buffer::AbstractVector{UInt8},
)
    array = ArrayValue(val.f!, Float64, val.dims)
    (; dset, saver) = _array_saver(group, name, array; n_times, buffer)
    _write_or_check_attribute(dset, "coords", _coord_matrix(val.coords))
    saver
end

function _array_saver(
    group::HDF5.Group,
    name::String,
    val::ArrayValue;
    n_times::Int,
    buffer::AbstractVector{UInt8},
)
    dspace = _unbounded_dataspace((val.dims..., n_times))
    dset = _create_or_get_time_dataset(
        group, name, val.type, dspace; chunk=_good_chunk(val.type, val.dims)
    )

    temp = _buffer_view(buffer, val.type, val.dims)

    slice = ntuple(_ -> :, length(val.dims))
    function saver(state::State, i_time)
        val.f!(temp, state)
        dset[slice..., i_time] = temp
    end

    (; dset, saver)
end

function _buffer_view(buffer::AbstractVector{UInt8}, T, dims::Dims)
    a = @views reshape(buffer[1:(prod(dims) * sizeof(T))], sizeof(T), dims...)
    reinterpret(reshape, T, a)
end

# Heuristic for HDF5 chunk size
function _good_chunk(T, dims::Dims)
    n = cld(100_000, prod(dims) * sizeof(T))
    (dims..., n)
end

function _coord_matrix(coords::AbstractVector{Coords})
    [_range_tuple(coords[i][j]) for j in 1:2, i in eachindex(coords)]
end
_range_tuple(r::AbstractRange) = (; start=first(r), step=step(r), length=length(r))

function _create_or_get_group(parent, path; kw...)
    if !haskey(parent, path)
        create_group(parent, path; kw...)
    else
        group = parent[path]
        if !(group isa HDF5.Group)
            error("object at $(repr(HDF5.name(group))) is not a group")
        end
        group
    end
end

function _write_or_check_dataset(parent, name, data; kw...)
    if !haskey(parent, name)
        write_dataset(parent, name, data; kw...)
    else
        dset = parent[name]

        if !(dset isa HDF5.Dataset)
            error("object at $(repr(HDF5.name(dset))) is not a dataset")
        end

        # Reading a 0D array to the HDF5 file unwraps the value from the array
        if read(dset) != _unwrap_0d(data)
            error("dataset $(repr(HDF5.name(dset))) has incorrect value")
        end
    end

    nothing
end

_unwrap_0d(x::AbstractArray{<:Any,0}) = x[]
_unwrap_0d(x) = x

function _create_or_get_time_dataset(parent, path, dtype, dspace; kw...)
    # The last dimension of the dataset is for the timestep

    if !haskey(parent, path)
        create_dataset(parent, path, dtype, dspace; kw...)
    else
        dset = parent[path]

        if !(dset isa HDF5.Dataset)
            error("object at $(repr(HDF5.name(dset))) is not a dataset")
        end

        if datatype(dset) != datatype(dtype)
            error("dataset $(repr(HDF5.name(dset))) has incorrect datatype")
        end

        dims, maxdims = HDF5.get_extent_dims(dataspace(dspace))
        actual_dims, actual_maxdims = HDF5.get_extent_dims(dset)

        if actual_dims[1:(end - 1)] != dims[1:(end - 1)]
            error("dataset $(repr(HDF5.name(dset))) has incorrect dataspace")
        end

        # If the duration of the simulation changed, the maximum dimension might increase.
        # Make sure to resize the dataset here if possible.
        n = dims[end]
        actual_n = actual_dims[end]
        if actual_n < n
            nmax = maxdims[end]
            actual_nmax = actual_maxdims[end]
            if actual_nmax != -1 && (nmax == -1 || nmax > actual_nmax)
                error(
                    "last dimension of dataset $(repr(HDF5.name(dset))) cannot be expanded"
                )
            end

            HDF5.set_extent_dims(dset, dims)
        end

        dset
    end
end

function _write_or_check_attribute(parent, name, data)
    if !haskey(attributes(parent), name)
        write_attribute(parent, name, data)
    else
        actual_data = read_attribute(parent, name)
        if actual_data != data
            error(
                "attribute $name of dataset $(repr(HDF5.name(parent))) has incorrect value"
            )
        end
    end
    nothing
end

# dataspace where the last dimension is unbounded
_unbounded_dataspace(dims::Tuple) = dataspace(dims, (dims[1:(end - 1)]..., -1))

function _init(save::SaveHDF5{String}, max_timestep)
    file = h5open(save.file, save.mode)
    HDF5Saver(save, file; toclose=true, max_timestep)
end
function _init(save::SaveHDF5, max_timestep)
    HDF5Saver(save, save.file; toclose=false, max_timestep)
end

function _update(saver::HDF5Saver, state::State)
    for saver in saver.savers
        saver(state)
    end
    nothing
end

function _finalize(saver::HDF5Saver)
    saver.toclose && close(saver.file)
    nothing
end

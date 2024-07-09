unit(::Val{N}, i) where {N} = CartesianIndex(ntuple(==(i), N))
unit(n) = Base.Fix1(unit, Val(n))

struct OffsetTuple{O,T<:Tuple}
    x::T
    OffsetTuple{O}(x::T) where {O,T} = new{O,T}(x)
end

OffsetTuple(a::Tuple) = OffsetTuple{1}(a)
OffsetTuple(a::OffsetTuple) = a

function n_offset_tuple(f, ::Val{N}, ::Val{O}) where {N,O}
    OffsetTuple{O}(ntuple(i -> f(i - 1 + O), Val(N)))
end

n_offset_tuple(f, a::Tuple) = n_offset_tuple(f, Val(length(a)), Val(1))
n_offset_tuple(f, a::OffsetTuple{O}) where {O} = n_offset_tuple(f, Val(length(a)), Val(O))

Base.length(a::OffsetTuple) = length(a.x)
Base.eachindex(a::OffsetTuple{O}) where {O} = (1:length(a)) .+ (O - 1)
Base.getindex(a::OffsetTuple{O}, i::Integer) where {O} = a.x[i-O+1]
Base.pairs(a::OffsetTuple{O}) where {O} = Base.Pairs(a, ntuple(i -> i - 1 + O, length(a)))

Base.map(f, a::OffsetTuple{O}) where {O} = OffsetTuple{O}(map(f, a.x))

function Adapt.adapt_structure(to, a::OffsetTuple{O}) where {O}
    OffsetTuple{O}(Adapt.adapt_structure(to, a.x))
end

struct Vec end
struct VecZ end
vec_kind(x::Tuple) = Vec()
vec_kind(x::OffsetTuple{3,<:NTuple{1}}) = VecZ()

function otheraxes(i)
    j = i % 3 + 1
    k = (i + 1) % 3 + 1
    (j, k)
end

each_other_axes(i) = (otheraxes(i), reverse(otheraxes(i)))

function permute(f, i::Int)
    (j, k) = otheraxes(i)
    f(j, k) - f(k, j)
end

permute(f, i, ::Vec, ::Vec) = permute(f, i)

function permute(f, i::Int, ::Vec, ::VecZ)
    @assert i in (1, 2)
    i == 1 ? f(2, 3) : -f(1, 3)
end

permute(f, i, a::VecZ, b::Vec) = permute((x, y) -> f(y, x), i, b, a)

macro loop(backend, inds, ex)
    if !(
        inds isa Expr &&
        inds.head == :call &&
        length(inds.args) == 3 &&
        inds.args[1] == :(in) &&
        inds.args[2] isa Symbol
    )
        throw(ArgumentError("second argument must be in the form `I in R`"))
    end

    I = esc(inds.args[2])
    R = esc(inds.args[3])
    kern = esc(gensym("kern"))
    I0 = esc(gensym("I0"))
    quote
        @kernel function $kern(@Const($I0))
            $I = @index(Global, Cartesian)
            $I += $I0
            $(esc(ex))
        end
        R = _cartesianindices($R)
        backend = get_backend($(esc(backend)))
        $kern(backend, 64)(R[1] - oneunit(R[1]); ndrange=size(R))
    end
end

_cartesianindices(I::CartesianIndices) = I
_cartesianindices(I) = CartesianIndices(I)

# Non-allocating sum(map(f, a, b)) for arrays.
function sum_map(f, a, b)
    s = zero(promote_type(eltype(a), eltype(b)))
    # b not included in eachindex to work on the GPU.
    for i in eachindex(a)
        s += f(a[i], b[i])
    end
    s
end

# Version of LinearAlgebra.dot that works on the GPU.
dot(a, b) = sum_map(*, a, b)

unit(::Val{N}, i) where {N} = CartesianIndex(ntuple(==(i), N))
unit(n) = Base.Fix1(unit, Val(n))

z_vector(a) = OffsetArray(SVector((a,)), 3:3)

ensure_3d(a::Tuple) = a
ensure_3d(a::AbstractArray) = z_vector(a)

struct Vec end
struct VecZ end
vec_kind(x::Tuple) = Vec()
vec_kind(x::AbstractArray) = VecZ()

function permute(f, i::Int, ::Vec, ::Vec)
    j = i % 3 + 1
    k = (i + 1) % 3 + 1
    f(j, k) - f(k, j)
end

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

δ(i, ::Val{N}) where {N} = CartesianIndex(ntuple(==(i), N))
δ_for(::CartesianIndex{N}) where {N} = Base.Fix2(δ, Val(N))

function permute(f, i)
    j = i % 3 + 1
    k = (i + 1) % 3 + 1
    f(j, k) - f(k, j)
end

macro loop(inds::Expr, ex)
    if !(
        inds.head == :call &&
        length(inds.args) == 3 &&
        inds.args[1] == :(in) &&
        inds.args[2] isa Symbol
    )
        throw(ArgumentError("first argument must be in the form `I in R`"))
    end

    I_ex = inds.args[2]
    I = esc(I_ex)
    R = esc(inds.args[3])

    syms = OrderedDict()
    ex2 = esc(_collect_arguments!(syms, ex, (I_ex,)))

    sym_pairs = collect(pairs(syms))
    exprs = @. esc(first(sym_pairs))
    args = @. esc(last(sym_pairs))

    kern = esc(gensym("kern"))
    I0 = esc(gensym("I0"))
    r = esc(gensym("r"))
    quote
        @kernel function $kern($(args...), @Const($I0))
            $I = @index(Global, Cartesian)
            $I += $I0
            $ex2
        end
        $r = _cartesianindices($R)
        $kern(get_backend($(exprs[1])), 64)(
            $(exprs...), $r[1] - oneunit($r[1]); ndrange=size($r)
        )
    end
end

_cartesianindices(I::CartesianIndices) = I
_cartesianindices(I) = CartesianIndices(I)

function _collect_arguments!(syms, ex::Expr, exclude)
    if ex.head == :.
        if ex in exclude
            ex
        else
            get!(syms, ex) do
                gensym(ex.args[2].value)
            end
        end
    else
        start = ex.head == :(call) ? 2 : 1
        Expr(
            ex.head,
            ex.args[1:(start-1)]...,
            (_collect_arguments!(syms, a, exclude) for a in ex.args[start:end])...,
        )
    end
end

function _collect_arguments!(syms, ex::Symbol, exclude)
    if ex in exclude
        ex
    else
        get!(syms, ex) do
            gensym(ex)
        end
    end
end

_collect_arguments!(syms, ex, _) = ex

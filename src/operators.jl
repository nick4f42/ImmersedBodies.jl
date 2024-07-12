function nonlinear!(nonlin, u, ω)
    for (i, nonlinᵢ) in pairs(nonlin)
        @loop nonlinᵢ (I in nonlinᵢ) nonlinᵢ[I] = nonlinear(i, u, ω, I)
    end
    nonlin
end

function nonlinear(i, u, ω, I)
    δ = unit(length(I))
    permute(i, vec_kind(u), vec_kind(ω)) do j, k
        uI = (u[j][I] + u[j][I-δ(i)] + u[j][I+δ(j)] + u[j][I-δ(i)+δ(j)]) / 4
        ωI = (ω[k][I] + ω[k][I+δ(j)]) / 2
        uI * ωI
    end
end

function rot!(ω, u; h)
    for (i, ωᵢ) in pairs(ω)
        @loop ωᵢ (I in ωᵢ) ωᵢ[I] = rot(i, u, I; h)
    end
    ω
end

function rot(i, u, I; h)
    δ = unit(length(I))
    permute(i) do j, k
        (u[k][I] - u[k][I-δ(j)]) / h
    end
end

function curl!(u, ψ; h)
    for (i, uᵢ) in pairs(u)
        @loop uᵢ (I in uᵢ) uᵢ[I] = curl(i, ψ, I; h)
    end
    u
end

function curl(i, ψ, I; h)
    δ = unit(length(I))
    permute(i, Vec(), vec_kind(ψ)) do j, k
        (ψ[k][I+δ(j)] - ψ[k][I]) / h
    end
end

struct LaplacianPlan{P1,P2,L<:AbstractArray}
    λ::L
    fwd::P1
    inv::P2
    n_logical::Int
end

function LaplacianPlan(ωᵢ, i, n::SVector{N}) where {N}
    R = inner_cell_axes(n, Loc_ω(i))
    nω = length.(R)
    λ = OffsetArray(similar(ωᵢ, nω), R)
    laplacian_eigvals!(λ, i)

    kind = laplacian_fft_kind(i, N)
    fwd = FFT_R2R.plan_r2r!(ωᵢ, kind)
    inv = FFT_R2R.plan_r2r!(ωᵢ, map(k -> FFTW.inv_kind[k], kind))
    n_logical = prod(map(FFTW.logical_size, nω, kind))

    LaplacianPlan(λ, fwd, inv, n_logical)
end

laplacian_fft_kind(i, nd) = ntuple(j -> i == j ? FFTW.REDFT01 : FFTW.RODFT00, nd)

function laplacian_eigvals!(λ, i)
    nd = ndims(λ)
    R = CartesianIndices(λ)
    n = size(λ)
    @loop λ (I in R) begin
        I₁ = Tuple(I - first(R)) .+ 1
        s = zero(eltype(λ))
        for j in 1:nd
            s += if (i == j)
                -4 * sin(π * (I₁[j] - 1) / (2n[j]))^2
            else
                -4 * sin(π * I₁[j] / (2(n[j] + 1)))^2
            end
        end
        λ[I] = s
    end
    λ
end

laplacian_plans(ω, n) = map(i -> LaplacianPlan(ω[i], i, n), tupleindices(ω))

struct EigenbasisTransform{F,O,P<:Tuple{Vararg{LaplacianPlan}}}
    f::F
    plan::OffsetTuple{O,P}
end

EigenbasisTransform(f, plan::Tuple) = EigenbasisTransform(f, OffsetTuple(plan))

function (X::EigenbasisTransform)(y, ω)
    for i in eachindex(ω)
        X(y[i], ω[i], i)
    end
    y
end

function (X::EigenbasisTransform)(yᵢ, ωᵢ, i)
    plan = X.plan[i]
    let yᵢ = no_offset_view(yᵢ), ωᵢ = no_offset_view(ωᵢ), λ = no_offset_view(plan.λ)
        mul!(yᵢ, plan.inv, ωᵢ)
        @. yᵢ *= X.f(λ) / plan.n_logical
        mul!(yᵢ, plan.fwd, yᵢ)
    end
    yᵢ
end

function multidomain_coarsen!(ω², ω¹; n)
    for i in eachindex(ω²)
        R = _coarse_indices(Tuple(n), Loc_ω(i))
        @loop ω²[i] (I in R) ω²[i][I] = multidomain_coarsen(i, ω¹[i], I; n)
    end
    ω²
end

function _coarse_indices(n::NTuple{N}, loc::Edge{Dual}) where {N}
    ntuple(N) do i
        n4 = n[i] .÷ 4
        i == loc.i ? (n4:3n4-1) : (n4+1:3n4-1)
    end
end

function multidomain_coarsen(i, ωᵢ, I²; n)
    T = eltype(ωᵢ)
    stencil = _coarsen_stencil(T)
    s = zero(T)
    indices = _fine_indices(i, Tuple(n), Tuple(I²))
    for I¹ in indices
        s += dot(SMatrix{3,3}(@view ωᵢ[I¹]), stencil)
    end
    s / length(indices)
end

function _coarsen_stencil(T)
    (@SMatrix [
        1 2 1
        2 4 2
        1 2 1
    ]) / T(16)
end

_fine_indices(_, n::NTuple{2}, I::NTuple{2}) = (CartesianIndices(_fine_range.(n, I)),)

function _fine_indices(i, n::NTuple{3}, I::NTuple{3})
    plane1 = 2(I[i] - (n[i] ÷ 4))
    r = _fine_range.(n, I)
    ntuple(2) do plane
        j = plane1 + plane - 1
        CartesianIndices(setindex(r, j:j, i))
    end
end

function _fine_range(n::Int, I::Int)
    2(I - (n ÷ 4)) .+ (-1:1)
end

function multidomain_interpolate!(ω_b, ω; n)
    for i in eachindex(ω), (j, k) in each_other_axes(i), dir in 1:2
        b = ω_b[i][dir, k]
        @loop b (I in b) b[I] = multidomain_interpolate(ω[i], (i, j, k), dir, I; n)
    end
    ω_b
end

function multidomain_interpolate(ωᵢ, (i, j, k), dir, I¹::CartesianIndex{2}; n)
    δ = unit(2)
    I² = CartesianIndex(ntuple(dim -> n[dim] ÷ 4 + fld(I¹[dim], 2), 2))
    if iseven(I¹[j])
        ωᵢ[I²]
    else
        (ωᵢ[I²] + ωᵢ[I²+δ(j)]) / 2
    end
end

function multidomain_interpolate(ωᵢ, (i, j, k), dir, I¹::CartesianIndex{3}; n)
    δ = unit(3)
    n4 = Tuple(n) .÷ 4
    I² = CartesianIndex(
        ntuple(3) do dim
            if dim == i
                n4[dim] + fld(I¹[dim] - 1, 2)
            else
                n4[dim] + fld(I¹[dim], 2)
            end
        end,
    )
    a = (1 + 2mod(I¹[i] + 1, 2)) / 4
    if iseven(I¹[j])
        (1 - a) * ωᵢ[I²] + a * ωᵢ[I²+δ(i)]
    else
        ((1 - a) * (ωᵢ[I²] + ωᵢ[I²+δ(j)]) + a * (ωᵢ[I²+δ(i)] + ωᵢ[I²+δ(i)+δ(j)])) / 2
    end
end

abstract type AbstractDeltaFunc end

(delta::AbstractDeltaFunc)(r::AbstractVector) = prod(delta, r)

struct DeltaYang3S <: AbstractDeltaFunc end
support(::DeltaYang3S) = 2

function (::DeltaYang3S)(r::AbstractFloat)
    u = one(r)
    a = abs(r)
    if a < 1
        17u / 48 + sqrt(3u) * π / 108 + a / 4 - r^2 / 4 +
        (1 - 2 * a) / 16 * sqrt(-12 * r^2 + 12 * a + 1) -
        sqrt(3u) / 12 * asin(sqrt(3u) / 2 * (2 * a - 1))
    elseif a < 2
        55u / 48 - sqrt(3u) * π / 108 - 13 * a / 12 +
        r^2 / 4 +
        (2 * a - 3) / 48 * sqrt(-12 * r^2 + 36 * a - 23) +
        sqrt(3u) / 36 * asin(sqrt(3u) / 2 * (2 * a - 3))
    else
        zero(r)
    end
end

struct Reg{D<:AbstractDeltaFunc,T,N,A<:AbstractArray{SVector{N,Int},2},M,W<:AbstractArray{T,M}}
    delta::D
    I::A
    weights::W
end

Adapt.@adapt_structure Reg

function Reg(backend, T, delta, nb, ::Val{N}) where {N}
    I = KernelAbstractions.zeros(backend, SVector{N,Int}, nb, N)

    s = support(delta)
    r = ntuple(_ -> length(-s:s), N)
    weights = KernelAbstractions.zeros(backend, T, r..., nb, N)

    Reg(delta, I, weights)
end

function update_weights!(reg::Reg, grid::Grid{N}, ibs, xbs) where {N}
    @assert ndims(ibs) == 1 && axes(ibs) == axes(xbs)
    for i in 1:N
        @loop reg.weights (J in ibs) begin
            ib = J[1]
            xb = xbs[ib]

            xu0 = coord(grid, Loc_u(i), zeros(SVector{N,Int}))
            reg.I[ib, i] = I = @. round(Int, (xb - xu0) / grid.h)

            for k in CartesianIndices(axes(reg.weights)[1:N])
                ΔI = (-support(reg.delta) - 1) .+ SVector(Tuple(k))
                xu = coord(grid, Loc_u(i), I + ΔI)
                reg.weights[k, ib, i] = reg.delta((xb - xu) / grid.h)
            end
        end
    end
    reg
end

function interpolate_body!(ub::AbstractMatrix, reg::Reg, u)
    s = support(reg.delta)
    @loop ub (J in ub) begin
        ib, i = Tuple(J)
        w = @view reg.weights[.., ib, i]
        Ib = reg.I[ib, i]
        I = CartesianIndices(map(i -> i .+ (-s:s), Tuple(Ib)))
        uᵢ = @view u[i][I]
        ub[J] = dot(w, uᵢ)
    end
end

function regularize!(fu, reg::Reg{<:Any,<:Any,N}, fb) where {N}
    R = axes(reg.weights)[1:N]

    for fuᵢ in fu
        fuᵢ .= 0
    end

    for J in CartesianIndices(fb)
        ib, i = Tuple(J)
        fuᵢ = fu[i]
        @loop fuᵢ (K in R) begin
            I0 = CartesianIndex(Tuple(reg.I[ib, i] .- (support(reg.delta) + 1)))
            I = I0 + K
            fuᵢ[I] += fb[J] * reg.weights[K, ib, i]
        end
    end

    fu
end

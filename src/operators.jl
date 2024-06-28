function nonlinear!(nonlin, u, ω)
    for (i, nonlinᵢ) in pairs(nonlin)
        @loop nonlinᵢ (I in nonlinᵢ) nonlinᵢ[I] = nonlinear(i, u, ω, I)
    end
    nonlin
end

function nonlinear(i, u, ω, I)
    δ = unit(length(I))
    permute(i, vec_kind(u), vec_kind(ω)) do j, k
        let ω = ensure_3d(ω)
            uI = (u[j][I] + u[j][I-δ(i)] + u[j][I+δ(j)] + u[j][I-δ(i)+δ(j)]) / 4
            ωI = (ω[k][I] + ω[k][I+δ(j)]) / 2
            uI * ωI
        end
    end
end

function rot!(ω, u; h)
    for (i, ωᵢ) in axispairs(ω)
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
        let ψ = ensure_3d(ψ)
            (ψ[k][I+δ(j)] - ψ[k][I]) / h
        end
    end
end

struct DeltaFunc{F}
    f::F
    support::Int
end

(delta::DeltaFunc)(r::Real) = delta.f(r)
(delta::DeltaFunc)(r) = prod(delta.f, r)

const delta_yang3_smooth1 = DeltaFunc(2) do r::AbstractFloat
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

struct Reg{F,T,N,A<:AbstractArray{SVector{N,Int},2},M,W<:AbstractArray{T,M}}
    delta::DeltaFunc{F}
    I::A
    weights::W
end

function Reg(backend, T, delta::DeltaFunc, nb, ::Val{N}) where {N}
    I = KernelAbstractions.zeros(backend, SVector{N,Int}, nb, N)

    s = delta.support
    support = ntuple(_ -> length(-s:s), N)
    weights = KernelAbstractions.zeros(backend, T, support..., nb, N)

    Reg(delta, I, weights)
end

function update_weights!(reg::Reg, grid::Grid{N}, ibs, xbs) where {N}
    @assert ndims(ibs) == 1 && axes(ibs) == axes(xbs)
    weights = reg.weights
    Is = reg.I
    delta = reg.delta
    s = delta.support

    for i in 1:N
        @loop weights (J in ibs) begin
            ib = J[1]
            xb = xbs[ib]

            xu0 = coord(grid, Loc_u(i), zeros(SVector{N,Int}))
            Is[ib, i] = I = @. round(Int, (xb - xu0) / grid.h)

            for k in CartesianIndices(axes(weights)[1:N])
                ΔI = (-s - 1) .+ SVector(Tuple(k))
                xu = coord(grid, Loc_u(i), I + ΔI)
                weights[k, ib, i] = delta((xb - xu) / grid.h)
            end
        end
    end
    reg
end

function interpolate!(ub::AbstractMatrix, reg::Reg, u)
    weights = reg.weights
    Is = reg.I
    delta = reg.delta
    @loop ub (J in ub) begin
        ib, i = Tuple(J)
        w = @view weights[.., ib, i]
        I = support_range(Is[ib, i], delta.support)
        uᵢ = @view u[i][I]
        ub[J] = dot(w, uᵢ)
    end
end

support_range(I, s) = CartesianIndices(map(i -> i .+ (-s:s), Tuple(I)))

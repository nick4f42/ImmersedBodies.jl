using ImmersedBodies
using ImmersedBodies:
    @loop,
    unit,
    each_other_axes,
    tupleindices,
    cell_axes,
    inner_cell_axes,
    boundary_axes,
    OffsetTuple,
    nonlinear!,
    rot!,
    curl!,
    LaplacianPlan,
    EigenbasisTransform,
    laplacian_plans,
    multidomain_coarsen!,
    multidomain_interpolate!,
    AbstractDeltaFunc,
    support,
    DeltaYang3S,
    Reg,
    update_weights!,
    interpolate_body!,
    regularize!
using KernelAbstractions
using GPUArrays
using OffsetArrays: OffsetArray, no_offset_view
using StaticArrays
using StaticArrays: SOneTo
using LinearAlgebra
using Test
using Random

import FFTW
import ImmersedBodies: FFT_R2R

import CUDA, AMDGPU

_backend(array) = get_backend(convert(array, [0]))

function _gridarray(f, array, grid, loc, R::Tuple{Vararg{AbstractRange}}; level=1)
    a = map(CartesianIndices(R)) do I
        x = coord(grid, loc, I, level)
        f(x)
    end
    OffsetArray(convert(array, a), R)
end

_loc_axes(::Val{N}, loc::Type{<:Edge}) where {N} = ntuple(identity, N)
_loc_axes(::Val{2}, loc::Type{Edge{Dual}}) = OffsetTuple{3}((3,))

function _gridarray(f, array, grid::Grid{N}, loc::Type{<:Edge}, R; kw...) where {N}
    map(_loc_axes(Val(N), loc)) do i
        _gridarray(x -> f(x)[i], array, grid, loc(i), R[i]; kw...)
    end
end

function _boundary_array(f, array, grid::Grid{N}, loc; kw...) where {N}
    Rb = boundary_axes(grid.n, loc; dims=ntuple(identity, 3))
    map(_loc_axes(Val(N), loc)) do i
        (SArray ∘ map)(CartesianIndices(Rb[i])) do index
            dir, j = Tuple(index)
            _gridarray(x -> f(x)[i], array, grid, loc(i), Rb[i][dir, j]; kw...)
        end
    end
end

struct LinearFunc{N,T,M}
    u0::SVector{N,T}
    du::SMatrix{N,3,T,M}
end
LinearFunc{N,T}(u0, du) where {N,T} = LinearFunc{N,T,3N}(u0, du)

(f::LinearFunc)(x::SVector{3}) = f.u0 + f.du * x
(f::LinearFunc)(x::SVector{2}) = f([x; 0])

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{LinearFunc{N,T}}) where {N,T}
    u0 = rand(rng, SVector{N,T})
    du = rand(rng, SMatrix{N,3,T})
    LinearFunc{N,T}(u0, du)
end

_rand_xy(T) = _rand_xy(Random.default_rng(), T)

function _rand_xy(rng::AbstractRNG, ::Type{LinearFunc{3,T}}) where {T}
    u0 = [@SArray(rand(T, 2)); 0]
    du = [
        @SArray(rand(T, 2, 2)) @SArray(zeros(T, 2, 1))
        @SArray(zeros(T, 1, 3))
    ]
    LinearFunc{3,T}(u0, du)
end

_rand_z(T) = _rand_z(Random.default_rng(), T)

function _rand_z(rng::AbstractRNG, ::Type{LinearFunc{3,T}}) where {T}
    ω0 = [@SArray(zeros(T, 2)); rand(T)]
    dω = [
        @SArray(zeros(T, 2, 3))
        @SArray(rand(T, 1, 2)) 0
    ]
    LinearFunc{3,T}(ω0, dω)
end

_is_xy(f::LinearFunc{3}) = iszero(f.u0[3]) && iszero(f.du[3, :]) && iszero(f.du[:, 3])
_is_z(f::LinearFunc{3}) = iszero(f.u0[1:2]) && iszero(f.du[1:2, :]) && iszero(f.du[3, 3])

function _with_divergence(f::LinearFunc{3,T}, d) where {T}
    i = diagind(f.du)
    du = setindex(f.du, d - sum(@view f.du[i[2:end]]), i[1])
    LinearFunc{3,T}(f.u0, du)
end

_div(f::LinearFunc{3}) = sum(diag(f.du))

function _curl(f::LinearFunc{3})
    A = f.du
    SVector(A[3, 2] - A[2, 3], A[1, 3] - A[3, 1], A[2, 1] - A[1, 2])
end

_kind_str(kind::Tuple) = string("(", join(FFTW.kind2string.(kind), ", "), ")")
_kind_str(kind) = FFTW.kind2string(kind)

function test_utils()
    @test unit(Val(2), 1) == CartesianIndex((1, 0))
    @test unit(Val(3), 1) == CartesianIndex((1, 0, 0))
    @test unit(Val(3), 3) == CartesianIndex((0, 0, 1))
    @test unit(4)(2) == CartesianIndex((0, 1, 0, 0))

    @test_throws "I in R" @macroexpand1 @loop x (2 in R) x[I] = y[I]
    @test_throws "I in R" @macroexpand1 @loop x (in(I, R, S)) x[I] = y[I]
    @test_throws ArgumentError @macroexpand1 @loop x I x[I] = y[I]
    @test_throws MethodError @macroexpand1 @loop x (I in R) x[I] = y[I] extra
end

function test_loop(array)
    let
        cmap(f, s...) = OffsetArray(map(f, CartesianIndices(s)), s...)
        asarray(T, a) = OffsetArray(T(no_offset_view(a)), axes(a)...)
        a1 = cmap(I -> 100 .+ float.(Tuple(I)), 2:5, 1:3, -4:-2)
        b1 = cmap(I -> float.(Tuple(I)), 2:4, 1:3, -4:-4)
        a2 = asarray(array, a1)
        b2 = asarray(array, b1)

        R = CartesianIndices((2:4, 1:2, -4:-4))

        @views a1[R] = b1[R]
        @loop a2 (I in R) a2[I] = b2[I]

        # Drop the offset indexing and check equality on the CPU.
        @test no_offset_view(a1) == Array(no_offset_view(a2))
    end

    let
        a = array([1.0, 5.0, 2.5])
        b = array([3, 7, -4])
        c = array(zeros(3))
        @loop c (I in (2:3,)) c[I] = b[I] - 2 * a[I]
        @test Array(c) ≈ [0, -3, -9]
    end

    let
        a = array([1.0, 2.0, 3.0])
        @test_throws MethodError @loop a (I in +) a[I] = 0
    end
end

function test_problems()
    let grid = Grid(; h=0.05, n=(7, 12, 5), x0=(0, 1, 0.5), levels=3)
        @test grid.n == [8, 12, 8]
    end

    let grid = Grid(; h=0.05, n=(7, 12), x0=(0, 1), levels=3)
        @test grid.n == [8, 12]
    end

    let h = 0.25, n = SVector(8, 4), x0 = SVector(10, 19), grid = Grid(; h, n, x0, levels=5)
        @test gridcorner(grid) == gridcorner(grid, 1) == x0
        @test gridcorner(grid, 2) ≈ x0 - n * h / 2
        @test gridcorner(grid, 3) ≈ x0 - n * h * 3 / 2

        @test gridstep(grid) == gridstep(grid, 1) == h
        @test gridstep(grid, 2) ≈ 2 * h
        @test gridstep(grid, 3) ≈ 4 * h

        @test coord(grid, Edge{Dual}(3), (1, 3)) ≈ x0 + h * SVector(1, 3)
        @test coord(grid, Edge{Primal}(2), (1, 3)) ≈ x0 + h * SVector(1.5, 3)
        @test coord(grid, Edge{Dual}(2), (1, 3)) ≈ x0 + h * SVector(1, 3.5)
        @test coord(grid, Edge{Primal}(2), (1, 3), 2) ≈
            (x0 - n * h / 2) + 2h * SVector(1.5, 3)
        @test coord(grid, Edge{Dual}(2), (1, 3), 2) ≈
            (x0 - n * h / 2) + 2h * SVector(1, 3.5)

        @test cell_axes(grid.n, Edge{Dual}(3)) == (0:8, 0:4)
        @test inner_cell_axes(grid.n, Edge{Dual}(3)) == (1:7, 1:3)

        @test cell_axes(grid.n, Edge{Primal}(1)) == (0:8, 0:3)
        @test inner_cell_axes(grid.n, Edge{Primal}(1)) == (1:7, 0:3)
    end
    let h = 0.25,
        n = SVector(8, 4, 12),
        x0 = SVector(10, 19, 5),
        grid = Grid(; h, n, x0, levels=5)

        @test cell_axes(grid.n, Edge{Dual}(2)) == (0:8, 0:3, 0:12)
        @test inner_cell_axes(grid.n, Edge{Dual}(2)) == (1:7, 0:3, 1:11)

        @test cell_axes(grid.n, Edge{Primal}(2)) == (0:7, 0:4, 0:11)
        @test inner_cell_axes(grid.n, Edge{Primal}(2)) == (0:7, 1:3, 0:11)
    end
end

function test_fft_r2r(array)
    params = [
        (FFTW.RODFT00, (8, 7), 1:2),
        (FFTW.REDFT10, (9, 6), 1:2),
        (FFTW.REDFT01, (7, 8), 1:2),
        ((FFTW.RODFT00, FFTW.REDFT01), (5, 9), [(1, 2)]),
        ((FFTW.RODFT00, FFTW.REDFT10, FFTW.REDFT01), (3, 6, 4), [(1, 2, 3)]),
    ]
    @testset "$(_kind_str(kind)) size=$sz" for (kind, sz, dimss) in params
        test_fft_r2r(array, kind, sz, dimss)
    end
end

function test_fft_r2r(array, kind, sz, dimss)
    for dims in dimss
        x1 = rand(sz...)
        x2 = array(x1)

        p1 = FFTW.plan_r2r!(x1, kind, dims)
        p2 = FFT_R2R.bad_plan_r2r!(x2, Val.(kind), dims)

        mul!(x1, p1, x1)
        mul!(x2, p2, x2)
        @test x1 ≈ convert(Array, x2)
    end
end

function test_delta_func(δ::AbstractDeltaFunc)
    s = support(δ)
    let r = s .+ 0.5 .+ [0.0, 1e-3, 0.5, 1.0, 100.0]
        @test all(@. δ(r) ≈ 0)
        @test all(@. δ(-r) ≈ 0)
    end

    let n = 1000
        @test 2s / (n - 1) * sum(δ, range(-s, s, n)) ≈ 1
    end
end

function test_nonlinear(
    array, grid::Grid{N}, u_true::LinearFunc{3}, ω_true::LinearFunc{3}, R
) where {N}
    if N == 2
        @assert _is_xy(u_true)
        @assert _is_z(ω_true)
    end

    nonlin_true(x) = u_true(x) × ω_true(x)

    Ru = map(r -> first(r)-1:last(r)+1, R)
    Rω = map(r -> first(r):last(r)+1, R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))
    ω = _gridarray(ω_true, array, grid, Loc_ω, ntuple(_ -> Rω, 3))

    nonlin_expect = _gridarray(nonlin_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    nonlin_got = nonlinear!(map(zero, nonlin_expect), u, ω)

    @test all(@. no_offset_view(nonlin_got) ≈ no_offset_view(nonlin_expect))

    (; nonlin_true, Ru, Rω, u, ω, nonlin_expect, nonlin_got)
end

function test_nonlinear(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        u = _rand_xy(LinearFunc{3,Float64}),
        ω = _rand_z(LinearFunc{3,Float64}),
        R = (1:5, 3:8)

        test_nonlinear(array, grid, u, ω, R)
    end
    nothing
end

function test_nonlinear(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        u = rand(LinearFunc{3,Float64}),
        ω = rand(LinearFunc{3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_nonlinear(array, grid, u, ω, R)
    end
    nothing
end

function test_rot(array, grid::Grid{N}, u_true::LinearFunc{3}, R) where {N}
    if N == 2
        @assert _is_xy(u_true)
    end

    ω_true(x) = _curl(u_true)

    Ru = map(r -> first(r)-1:last(r), R)

    u = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> Ru, 3))

    ω_expect = _gridarray(ω_true, array, grid, Loc_ω, ntuple(_ -> R, 3))
    ω_got = rot!(map(zero, ω_expect), u; h=grid.h)

    @test all(i -> no_offset_view(ω_got[i]) ≈ no_offset_view(ω_expect[i]), eachindex(ω_got))

    (; ω_true, Ru, u, ω_expect, ω_got)
end

function test_rot(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        u = _rand_xy(LinearFunc{3,Float64}),
        R = (2:4, 0:3)

        test_rot(array, grid, u, R)
    end
    nothing
end

function test_rot(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        u = rand(LinearFunc{3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_rot(array, grid, u, R)
    end
    nothing
end

function test_curl(array, grid::Grid{N}, ψ_true::LinearFunc{3}, R) where {N}
    if N == 2
        @assert _is_z(ψ_true)
    end

    u_true(x) = _curl(ψ_true)

    Rψ = map(r -> first(r):last(r)+1, R)

    ψ = _gridarray(ψ_true, array, grid, Loc_ω, ntuple(_ -> Rψ, 3))

    u_expect = _gridarray(u_true, array, grid, Loc_u, ntuple(_ -> R, 3))
    u_got = curl!(map(zero, u_expect), ψ; h=grid.h)

    @test all(@. no_offset_view(u_got) ≈ no_offset_view(u_expect))

    (; u_true, Rψ, ψ, u_expect, u_got)
end

function test_curl(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ψ = _rand_z(LinearFunc{3,Float64}),
        R = (2:4, 0:3)

        test_curl(array, grid, ψ, R)
    end
    nothing
end

function test_curl(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ψ = rand(LinearFunc{3,Float64}),
        R = (2:4, 0:3, -1:1)

        test_curl(array, grid, ψ, R)
    end
    nothing
end

function test_laplacian_inv(array, grid::Grid{N}, ψ_true::LinearFunc{3,T}) where {N,T}
    @assert _div(ψ_true) < eps(T)

    if N == 2
        @assert _is_z(ψ_true)
    end

    Rψ = ntuple(i -> inner_cell_axes(grid.n, Loc_ω(i)), 3)
    Rψb = ntuple(i -> cell_axes(grid.n, Loc_ω(i)), 3)
    Ru = ntuple(i -> inner_cell_axes(grid.n, Loc_u(i)), 3)

    ψ = _gridarray(ψ_true, array, grid, Loc_ω, Rψb)
    for i in eachindex(ψ),
        (j, _) in each_other_axes(i),
        Iⱼ in (Rψb[i][j][begin], Rψb[i][j][end])

        R = CartesianIndices(setindex(Rψb[i], Iⱼ:Iⱼ, j))
        @loop ψ[i] (I in R) ψ[i][I] = 0
    end

    ψ_expect = map(i -> OffsetArray(ψ[i][Rψ[i]...], Rψ[i]), tupleindices(ψ))
    ψ_got = map(similar, ψ_expect)
    u = ntuple(N) do i
        dims = Ru[i]
        OffsetArray(
            KernelAbstractions.zeros(_backend(array), Float64, length.(dims)...), dims
        )
    end

    plan = laplacian_plans(ψ_got, grid.n)

    curl!(u, ψ; h=grid.h)
    rot!(ψ_got, u; h=grid.h)
    EigenbasisTransform(λ -> -1 / (λ / grid.h^2), plan)(ψ_got, ψ_got)

    @test all(i -> no_offset_view(ψ_got[i]) ≈ no_offset_view(ψ_expect[i]), eachindex(ψ_got))

    (; Rψ, Rψb, Ru, ψ, ψ_expect, ψ_got, u, plan)
end

function test_laplacian_inv(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ψ = _rand_z(LinearFunc{3,Float64})

        test_laplacian_inv(array, grid, ψ)
    end
    nothing
end

function test_laplacian_inv(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ψ = _with_divergence(rand(LinearFunc{3,Float64}), 0)

        test_laplacian_inv(array, grid, ψ)
    end
    nothing
end

function test_multidomain_coarsen(array, grid::Grid{N}, ω_true::LinearFunc{3}) where {N}
    R = ntuple(i -> inner_cell_axes(grid.n, Loc_ω(i)), 3)
    ω¹ = _gridarray(ω_true, array, grid, Loc_ω, R; level=1)
    ω²_expect = _gridarray(ω_true, array, grid, Loc_ω, R; level=2)
    ω²_got = map(copy, ω²_expect)

    for i in eachindex(ω²_got)
        R_inner = CartesianIndices(
            ntuple(N) do j
                n4 = grid.n[j] ÷ 4
                i == j ? (n4:3n4-1) : (n4+1:3n4-1)
            end,
        )
        @loop ω²_got[i] (I in R_inner) ω²_got[i][I] = 0
    end

    multidomain_coarsen!(ω²_got, ω¹; n=grid.n)

    @test all(
        i -> no_offset_view(ω²_got[i]) ≈ no_offset_view(ω²_expect[i]), eachindex(ω²_got)
    )

    (; R, ω¹, ω²_expect, ω²_got)
end

function test_multidomain_coarsen(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ω = _rand_z(LinearFunc{3,Float64})

        test_multidomain_coarsen(array, grid, ω)
    end
    nothing
end

function test_multidomain_coarsen(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ω = rand(LinearFunc{3,Float64})

        test_multidomain_coarsen(array, grid, ω)
    end
    nothing
end

function test_multidomain_interpolate(array, grid::Grid{N}, ω_true::LinearFunc{3}) where {N}
    R = ntuple(i -> inner_cell_axes(grid.n, Loc_ω(i)), 3)
    Rb = boundary_axes(grid.n, Loc_ω; dims=ntuple(identity, 3))

    ω = _gridarray(ω_true, array, grid, Loc_ω, R; level=2)

    ω_b_expect = _boundary_array(ω_true, array, grid, Loc_ω; level=1)
    ω_b_got = map(a -> map(zero, a), ω_b_expect)

    multidomain_interpolate!(ω_b_got, ω; n=grid.n)

    @test all(
        i -> all(@. no_offset_view(ω_b_got[i]) ≈ no_offset_view(ω_b_expect[i])),
        eachindex(ω_b_got),
    )

    (; R, ω, ω_b_expect, ω_b_got)
end

function test_multidomain_interpolate(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3),
        ω = _rand_z(LinearFunc{3,Float64})

        test_multidomain_interpolate(array, grid, ω)
    end
    nothing
end

function test_multidomain_interpolate(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3),
        ω = rand(LinearFunc{3,Float64})

        test_multidomain_interpolate(array, grid, ω)
    end
    nothing
end

function test_regularization(
    array, grid::Grid{N}, u_true::LinearFunc{3}, xb::AbstractVector{<:SVector}
) where {N}
    backend = _backend(array)
    T = Float64
    nb = length(xb)

    reg = Reg(backend, T, DeltaYang3S(), nb, Val(N))
    update_weights!(reg, grid, 1:nb, xb)

    R = ntuple(i -> inner_cell_axes(grid.n, Loc_u(i)), N)

    u = _gridarray(u_true, array, grid, Loc_u, R)

    ub_expect = (permutedims ∘ stack ∘ map)(x -> u_true(x)[1:N], Array(xb))
    ub_got = KernelAbstractions.zeros(backend, T, nb, N)
    interpolate_body!(ub_got, reg, u)

    @test Array(ub_got) ≈ ub_expect

    fu = _gridarray(x -> zero(SVector{N}), array, grid, Loc_u, R)
    fb = KernelAbstractions.ones(backend, T, nb, N)
    regularize!(fu, reg, fb)

    @test all(@. sum(no_offset_view(fu)) ≈ nb)

    (; reg, R, u, ub_expect, ub_got, fu, fb)
end

function test_regularization(array, ::Val{2})
    let grid = Grid(; h=0.05, n=(80, 80), x0=(-2.0, -1.95), levels=3),
        u = _rand_xy(LinearFunc{3,Float64}),
        nb = 20,
        xb = (array ∘ map)(range(0, 2π, nb)) do t
            SVector(cos(t), sin(t))
        end

        test_regularization(array, grid, u, xb)
    end
    nothing
end

function test_regularization(array, ::Val{3})
    let grid = Grid(; h=0.05, n=(80, 80, 80), x0=(-2.0, -1.95, -2.05), levels=3),
        u = rand(LinearFunc{3,Float64}),
        nb = 20,
        xb = (array ∘ map)(range(0, 1, nb)) do t
            a = 2π * t
            SVector(cos(a), sin(a), 2t - 1)
        end

        test_regularization(array, grid, u, xb)
    end
    nothing
end

using ImmersedBodies
using ImmersedBodies:
    @loop,
    unit,
    nonlinear!,
    rot!,
    curl!,
    delta_yang3_smooth1,
    Reg,
    update_weights!,
    interpolate!,
    regularize!
using KernelAbstractions
using GPUArrays
using OffsetArrays: OffsetArray, no_offset_view
using StaticArrays
using LinearAlgebra
using Test

import CUDA, AMDGPU

_backend(array) = get_backend(array([0]))

function _gridarray(
    f, grid, loc, R::NTuple{N,CartesianIndices}; array=identity, dims=ntuple(identity, N)
) where {N}
    map(dims, R) do i, r
        OffsetArray(array(
            map(r) do I
                x = coord(grid, loc(i), I)
                f(x)[i]
            end,
        ), r)
    end
end
function _gridarray(f, grid, loc, R::Tuple{Vararg{Tuple}}; kw...)
    _gridarray(f, grid, loc, CartesianIndices.(R); kw...)
end

# curl of Ax
_curl(A) = SVector(A[3, 2] - A[2, 3], A[1, 3] - A[3, 1], A[2, 1] - A[1, 2])

arrays = [Array]
CUDA.functional() && push!(arrays, CUDA.CuArray)
AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)

@info "Functional backends: $arrays"

@testset "ImmersedBodies.jl" verbose = true begin
    @testset "utils" begin
        @test unit(Val(2), 1) == CartesianIndex((1, 0))
        @test unit(Val(3), 1) == CartesianIndex((1, 0, 0))
        @test unit(Val(3), 3) == CartesianIndex((0, 0, 1))
        @test unit(4)(2) == CartesianIndex((0, 1, 0, 0))

        @test_throws "I in R" @macroexpand1 @loop x (2 in R) x[I] = y[I]
        @test_throws "I in R" @macroexpand1 @loop x (in(I, R, S)) x[I] = y[I]
        @test_throws ArgumentError @macroexpand1 @loop x I x[I] = y[I]
        @test_throws MethodError @macroexpand1 @loop x (I in R) x[I] = y[I] extra
    end

    @testset "utils $array" for array in arrays
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

    @testset "problems" begin
        let grid = Grid(; h=0.05, n=(7, 12, 5), x0=(0, 1, 0.5), levels=3)
            @test grid.n == [8, 12, 8]
        end

        let grid = Grid(; h=0.05, n=(7, 12), x0=(0, 1), levels=3)
            @test grid.n == [8, 12]
        end

        let h = 0.25,
            n = SVector(8, 4),
            x0 = SVector(10, 19),
            grid = Grid(; h, n, x0, levels=5)

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
        end
    end

    @testset "operators" begin
        @testset "$δ" for δ in (delta_yang3_smooth1,)
            s = δ.support
            let r = s .+ 0.5 .+ [0.0, 1e-3, 0.5, 1.0, 100.0]
                @test all(@. δ(r) ≈ 0)
                @test all(@. δ(-r) ≈ 0)
            end

            let n = 1000
                @test 2s / (n - 1) * sum(δ, range(-s, s, n)) ≈ 1
            end
        end
    end

    @testset "operators $array" for array in arrays
        backend = _backend(array)

        @testset "2D nonlinear" begin
            grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3)

            u0 = [@SArray(rand(2)); 0]
            du = [@SArray(rand(2, 2)); @SArray(zeros(1, 2))]
            ω0 = [@SArray(zeros(2)); rand()]
            dω = [@SArray(zeros(2, 2)); @SArray(rand(1, 2))]
            u_true(x) = u0 + du * x
            ω_true(x) = ω0 + dω * x
            nonlin_true(x) = u_true(x) × ω_true(x)

            R = (2:4, 0:3)
            nonlin_expect = _gridarray(nonlin_true, grid, Loc_u, (R, R); array)
            Ru = map(r -> first(r)-1:last(r)+1, R)
            u = _gridarray(u_true, grid, Loc_u, (Ru, Ru); array)
            Rω = map(r -> first(r):last(r)+1, R)
            ω = _gridarray(ω_true, grid, Loc_ω, (Rω,); array, dims=3)[1]

            nonlin_got = nonlinear!(similar.(nonlin_expect), u, ω)

            @test all(@. no_offset_view(nonlin_got) ≈ no_offset_view(nonlin_expect))
        end
        @testset "3D nonlinear" begin
            grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3)

            u0 = @SArray rand(3)
            du = @SArray rand(3, 3)
            ω0 = @SArray rand(3)
            dω = @SArray rand(3, 3)
            u_true(x) = u0 + du * x
            ω_true(x) = ω0 + dω * x
            nonlin_true(x) = u_true(x) × ω_true(x)

            R = (2:4, 0:3, -1:1)
            nonlin_expect = _gridarray(nonlin_true, grid, Loc_u, (R, R, R); array)
            Ru = map(r -> first(r)-1:last(r)+1, R)
            u = _gridarray(u_true, grid, Loc_u, (Ru, Ru, Ru); array)
            Rω = map(r -> first(r):last(r)+1, R)
            ω = _gridarray(ω_true, grid, Loc_ω, (Rω, Rω, Rω); array)

            nonlin_got = nonlinear!(similar.(nonlin_expect), u, ω)

            @test all(@. no_offset_view(nonlin_got) ≈ no_offset_view(nonlin_expect))
        end
        @testset "2D rot" begin
            grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3)

            u0 = @SArray(rand(2))
            du = [
                @SArray(rand(2, 2)) @SArray(zeros(2))
                @SArray(zeros(1, 3))
            ]
            u_true(x) = u0 + du[1:2, 1:2] * x
            ω_true(x) = _curl(du)

            R = (2:4, 0:3)
            ω_expect = _gridarray(ω_true, grid, Loc_ω, (R,); array, dims=3)[1]
            Ru = map(r -> first(r)-1:last(r), R)
            u = _gridarray(u_true, grid, Loc_u, (Ru, Ru); array)

            ω_got = rot!(similar(ω_expect), u; h=grid.h)

            @test no_offset_view(ω_got) ≈ no_offset_view(ω_expect)
        end
        @testset "3D rot" begin
            grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3)

            u0 = @SArray rand(3)
            du = @SArray rand(3, 3)
            u_true(x) = u0 + du * x
            ω_true(x) = _curl(du)

            R = (2:4, 0:3, -1:1)
            ω_expect = _gridarray(ω_true, grid, Loc_ω, (R, R, R); array)
            Ru = map(r -> first(r)-1:last(r), R)
            u = _gridarray(u_true, grid, Loc_u, (Ru, Ru, Ru); array)

            ω_got = rot!(similar.(ω_expect), u; h=grid.h)

            @test all(@. no_offset_view(ω_got) ≈ no_offset_view(ω_expect))
        end
        @testset "2D curl" begin
            grid = Grid(; h=0.05, n=(8, 16), x0=(-0.3, 0.4), levels=3)

            ψ0 = SVector(0, 0, rand())
            dψ = [
                @SArray(zeros(2, 3))
                @SArray(rand(1, 2)) 0
            ]
            ψ_true(x) = ψ0 + dψ[:, 1:2] * x
            u_true(x) = _curl(dψ)

            R = (2:4, 0:3)
            u_expect = _gridarray(u_true, grid, Loc_u, (R, R); array)
            Rψ = map(r -> first(r):last(r)+1, R)
            ψ = _gridarray(ψ_true, grid, Loc_ω, (Rψ,); array, dims=3)[1]

            u_got = curl!(similar.(u_expect), ψ; h=grid.h)

            @test all(@. no_offset_view(u_got) ≈ no_offset_view(u_expect))
        end
        @testset "3D curl" begin
            grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3)

            ψ0 = @SArray rand(3)
            dψ = @SArray rand(3, 3)
            ψ_true(x) = ψ0 + dψ * x
            u_true(x) = _curl(dψ)

            R = (2:4, 0:3, -1:1)
            u_expect = _gridarray(u_true, grid, Loc_u, (R, R, R); array)
            Rψ = map(r -> first(r):last(r)+1, R)
            ψ = _gridarray(ψ_true, grid, Loc_ω, (Rψ, Rψ, Rψ); array)

            u_got = curl!(similar.(u_expect), ψ; h=grid.h)

            @test all(@. no_offset_view(u_got) ≈ no_offset_view(u_expect))
        end
        @testset "regularize/interpolate" begin
            nb = 20
            xb = array([SVector(cos(t), sin(t)) for t in range(0, 2π, nb)])
            T = Float64

            grid = Grid(; h=0.05, n=(80, 80), x0=(-2.0, -2.0), levels=3)
            reg = Reg(backend, T, delta_yang3_smooth1, nb, Val(2))
            update_weights!(reg, grid, 1:nb, xb)

            u0 = @SArray rand(2)
            du = @SArray rand(2, 2)
            u_true(x) = u0 + du * x

            R = map(n -> 0:n, Tuple(grid.n))

            let
                u = _gridarray(u_true, grid, Loc_u, (R, R); array)

                ub_expect = permutedims(stack(u_true.(Array(xb))))
                ub_got = KernelAbstractions.zeros(backend, T, nb, 2)
                interpolate!(ub_got, reg, u)

                @test Array(ub_got) ≈ ub_expect
            end

            let
                fu = _gridarray(zero, grid, Loc_u, (R, R); array)
                fb = KernelAbstractions.ones(backend, T, nb, 2)
                regularize!(fu, reg, fb)

                # `sum` not forwarded to GPU array by `OffsetArray`.
                @test all(@. sum(no_offset_view(fu)) ≈ nb)
            end
        end
    end
end

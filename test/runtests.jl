using ImmersedBodies
using ImmersedBodies: @loop, δ, δ_for
using GPUArrays
using OffsetArrays: OffsetArray, no_offset_view
using StaticArrays
using LinearAlgebra
using Test

import CUDA, AMDGPU

arrays = [Array]
CUDA.functional() && push!(arrays, CUDA.CuArray)
AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)

@info "Functional backends: $arrays"

@testset "ImmersedBodies.jl" verbose = true begin
    @testset "utils" begin
        @test δ(1, Val(2)) == CartesianIndex((1, 0))
        @test δ(1, Val(3)) == CartesianIndex((1, 0, 0))
        @test δ(3, Val(3)) == CartesianIndex((0, 0, 1))
        @test δ_for(CartesianIndex((9, 9, 9, 9)))(2) == CartesianIndex((0, 1, 0, 0))

        @test_throws "I in R" @macroexpand1 @loop (2 in R) x[I] = y[I]
        @test_throws "I in R" @macroexpand1 @loop in(I, R, S) x[I] = y[I]
        @test_throws MethodError @macroexpand1 @loop I x[I] = y[I]
        @test_throws MethodError @macroexpand1 @loop (I in R) x[I] = y[I] extra
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
            @loop (I in R) a2[I] = b2[I]

            # Drop the offset indexing and check equality on the CPU.
            @test no_offset_view(a1) == Array(no_offset_view(a2))
        end

        let
            a = array([1.0, 5.0, 2.5])
            b = array([3, 7, -4])
            c = array(zeros(3))
            @loop (I in (2:3,)) c[I] = b[I] - 2 * a[I]
            @test Array(c) ≈ [0, -3, -9]
        end

        let
            a = array([1.0, 2.0, 3.0])
            @test_throws MethodError @loop (I in +) a[I] = 0
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

    @testset "operators $array" for array in arrays
        @testset "nonlinear" begin
            grid = Grid(; h=0.05, n=(8, 16, 12), x0=(-0.3, 0.4, 0.1), levels=3)

            u0 = @SVector rand(3)
            du = @SMatrix rand(3, 3)
            ω0 = @SVector rand(3)
            dω = @SMatrix rand(3, 3)
            u_true(x) = u0 + du * x
            ω_true(x) = ω0 + dω * x
            nonlin_true(x) = u_true(x) × ω_true(x)

            R = (2:4, 0:3, -1:1)
            Ru = map(r -> first(r)-1:last(r)+1, R)
            Rω = map(r -> first(r):last(r)+1, R)
        end
    end
end

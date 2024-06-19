using ImmersedBodies
using ImmersedBodies: @loop
using GPUArrays
using OffsetArrays: OffsetArray, no_offset_view
using Test

import CUDA, AMDGPU

arrays = [Array]
CUDA.functional() && push!(arrays, CUDA.CuArray)
AMDGPU.functional() && push!(arrays, AMDGPU.ROCArray)

@info "Functional backends: $arrays"

@testset "ImmersedBodies.jl" verbose = true begin
    @testset "utils" begin
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
            R = CartesianIndices((2:3,))
            @loop (I in R) c[I] = b[I] - 2 * a[I]
            @test Array(c) ≈ [0, -3, -9]
        end
    end

    @testset "grids" begin
        let grid = Grid(; h=0.05, n=(7, 12, 5), x0=(0, 1, 0.5), levels=3)
            @test grid.n == [8, 12, 8]
        end
    end
end

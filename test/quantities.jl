module TestQuantities

using ImmersedBodies.Quantities
using Test
using Plots

@testset "quantities" begin
    @testset "plotting" begin
        let xs = LinRange(0, 1, 3), ys = LinRange(0, 2, 4)
            z = xs .+ ys'
            val = GridValue(z, (xs, ys))

            p, = plot(val)
            @test p[1][:seriestype] == :heatmap # Make sure the recipe was applied
        end

        let coords = [(LinRange(-t, t, 6), LinRange(-0.5t, 0.5t, 3)) for t in 2.0 .^ (0:2)]
            z = cat((xs .+ ys' for (xs, ys) in coords)...; dims=3)
            val = MultiLevelGridValue(z, coords)

            p, = plot(val)
            @test p[1][:seriestype] == :heatmap # Make sure the recipe was applied
            @test length(p.series_list) == size(z, 3) # One heatmap per sublevel
        end
    end

    @testset "values" begin
        # Use a placeholder state for testing
        state = 0

        let
            array = reshape(1:6, 3, 2)
            f = ArrayQuantity(state -> array)
            v = f(state)

            @test v == array
            @test eltype(v) <: Int
        end

        let
            coords = (LinRange(0, 1, 2), LinRange(0, 1, 3))
            array = [1 2 3; 4 5 6]
            f = GridQuantity(state -> array, coords)
            v = f(state)

            @test coordinates(f) == coords

            @test v isa GridValue
            @test eltype(v) <: Int
            @test v == array
            @test coordinates(v) == coords
        end

        let
            coords = [(LinRange(0, 10i, 2), LinRange(0, 30i, 3)) for i in 1:4]
            array = reshape(1:(2 * 3 * 4), 2, 3, 4)
            f = MultiLevelGridQuantity(state -> array, coords)
            v = f(state)

            @test coordinates(f) == coords

            @test v isa MultiLevelGridValue
            @test eltype(v) <: Int
            @test v == array
            @test coordinates(v) == coords
        end
    end
end

end # module

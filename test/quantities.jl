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
end

end # module

module TestCurves

using ImmersedBodies.Curves
using StaticArrays
using Test

@testset "curves" begin
    let
        # Evenly spaced parameterization
        function f(t)
            θ = 2 * pi * t
            return (cos(θ), sin(θ))
        end

        # Unevenly space the parameterization by transforming the input
        t(s) = s^2

        curve = ParameterizedCurve(f ∘ t)

        @test isclosed(curve)
        @test curve(0) ≈ SVector(f(0))

        n = 20
        segments = partition(curve, n)

        # The partitioned points should be roughly the same as the evenly spaced ones
        points = zeros(n, 2)
        for (i, t) in enumerate((0:(n - 1)) ./ n)
            points[i, :] .= f(t)
        end

        @test segments.points ≈ points rtol = 0.001

        ds = 2 * pi / n
        @test all(isapprox(ds; rtol=0.01), segments.lengths)
    end

    let
        f(t) = (t^2, 2t)

        curve = ParameterizedCurve(f)

        @test !isclosed(curve)
        @test curve(1) ≈ SVector(f(1))
    end
end

end # module

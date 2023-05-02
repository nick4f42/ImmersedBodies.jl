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

        @test Curves.points(segments) ≈ points rtol = 0.001

        ds = 2 * pi / n
        @test all(isapprox(ds; rtol=0.01), Curves.lengths(segments))
    end

    let
        f(t) = (t^2, 2t)

        curve = ParameterizedCurve(f)

        @test !isclosed(curve)
        @test curve(1) ≈ SVector(f(1))
    end

    let t = translate((1, 2)) |> rotate(deg2rad(90)) |> scale(2)
        @test t((-1, 2)) ≈ [-8, 0]
        @test t((-1, -2)) ≈ [0, 0]
        @test t([0, 0]) ≈ [-4, 2]
    end

    @test translate((2, 3))((1, 5)) ≈ [3, 8]
    @test rotate(1.0)((1, 0)) ≈ [cos(1.0), sin(1.0)]
    @test scale(3)((5, 10)) ≈ [15, 30]

    let
        t = range(0, 2π, 13)[1:12]
        points = @. [cos(t) sin(t)]

        cut_at = map((5, 45, 181, -35)) do θ
            Curves.LineSegment((cosd(θ), sind(θ)), (0, 0))
        end

        ranges = Curves.cut_indices(points, cut_at)
        @test ranges == (1:1, 2:2, 3:7, 8:11, 12:12)

        segments = Segments(points, ones(size(points, 1)))
        parts = Curves.separate(segments, cut_at)

        part_points = reduce(vcat, Iterators.map(Curves.points, parts))
        @test part_points == Curves.points(segments)

        part_counts = map(length ∘ Curves.lengths, parts)
        @test part_counts == (1, 1, 5, 4, 1)

        # Cut segments must be in order
        let cut = (cut_at[2], cut_at[1])
            # Cuts the 2nd segment, doesn't find the 1st segment
            @test_throws "Cannot find intersection" Curves.cut_indices(points, cut)
        end
        # Cuts the 4th segment, no more points to check the 2nd segment
        let cut = (cut_at[4], cut_at[2])
            @test_throws "Cannot find intersection" Curves.cut_indices(points, cut)
        end
        # Cut segments must intersect somewhere
        let cut = (Curves.LineSegment((2.0, 3.0), (2.0, 4.0)),)
            @test_throws "Cannot find intersection" Curves.cut_indices(points, cut)
        end
    end

    let
        line = LineSegment((0, 1), (1, 1)) |> translate((1, 0)) |> rotate(π) |> scale(2)
        @test line(0) ≈ [-2, -2]
        @test line(1) ≈ [-4, -2]

        points = Curves.points(partition(line, 3))
        @test points[1, :] ≈ [-2, -2]
        @test points[end, :] ≈ [-4, -2]
    end

    for curve in (LineSegment((1, 2), (-5, 3)), Circle())
        transform = translate((1, 0)) |> rotate(deg2rad(90)) |> scale(3)
        curve2 = transform(curve)

        @test isclosed(curve2) == isclosed(curve)
        @test arclength(curve2) ≈ 3 * arclength(curve)
        @test curve2(0.3) ≈ transform(curve(0.3))
        @test curve2(0.6) ≈ transform(curve(0.6))
    end

    @test_throws "only contain digits" NacaParams("hi")
    @test_throws "airfoil not implemented" NacaParams("001")

    let
        params = NacaParams("3412")
        @test params isa Curves.NACA4

        (; m, p, t) = params
        @test m ≈ 0.03
        @test p ≈ 0.40
        @test t ≈ 0.12

        @test isclosed(NACA(params))
    end

    let
        airfoil = Curves.naca"2412"

        s_le = Curves.leading_edge(airfoil)
        s_te = Curves.trailing_edge(airfoil)

        @test airfoil(s_le) ≈ [0, 0] atol = 1e-8
        @test airfoil(s_te) ≈ [1, 0] atol = 1e-8

        n = 50
        segments = partition(airfoil, n)
        points = Curves.points(segments)
        lengths = Curves.lengths(segments)
        arclen = arclength(airfoil)

        @test all(≈(arclen / n; rtol=0.1), lengths)

        let i1 = 1 + floor(Int, n * s_le), i2 = i1 + 1
            # Airfoil y coordinate swaps sign at leading edge
            @test sign(points[i1, 2]) != sign(points[i2, 2])
        end
    end
end

end # module

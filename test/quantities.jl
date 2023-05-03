module TestQuantities

using ImmersedBodies
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

        let arrays = [[1 2 3; 4 5 6], [7 8; 9 10], [11; 12;;]],
            f = ConcatArrayQuantity(state -> arrays, 2),
            v = f(state)

            @test v isa ConcatArrayValue
            @test eltype(v) <: AbstractMatrix{Int}
            @test v == arrays
            @test v.dim == 2
        end

        let array = [1, 2, 5], f = ConcatArrayQuantity(state -> array, 1), v = f(state)
            @test v isa ConcatArrayValue
            @test eltype(v) <: Int
            @test v == array
            @test v.dim == 1
        end

        let bodies = 2:4,
            arrays = [[1.0 2.0], [3.0 4.0; 5.0 6.0], [7.0 8.0]],
            f = BodyArrayQuantity(state -> arrays, 1, bodies),
            v = f(state)

            @test bodyindices(f) == bodies

            @test v isa BodyArrayValue
            @test v == arrays
            @test v.dim == 1
            @test bodyindices(v) == bodies
        end

        let bodies = [1, 3, 4],
            array = [0.1, 0.3, 0.4],
            f = BodyArrayQuantity(state -> array, 1, bodies),
            v = f(state)

            @test bodyindices(f) == bodies

            @test v isa BodyArrayValue
            @test v == array
            @test v.dim == 1
            @test bodyindices(v) == bodies
        end
    end

    @testset "functions" begin
        flow = FreestreamFlow(t -> (1.0, 0.0); Re=50.0)
        basegrid = UniformGrid(0.05, (-1.0, 1.0), (-1.0, 1.0))
        grids = MultiLevelGrid(basegrid, 2)

        fluid = PsiOmegaFluidGrid(flow, grids; scheme=CNAB(0.005))

        curves = [
            Curves.Circle(0.2, (0.0, 0.3)),
            Curves.LineSegment((-0.3, 0.0), (0.3, 0.0)),
            Curves.Circle(0.2, (0.0, -0.3)),
        ]
        segments = map(curve -> partition(curve, fluid), curves)
        bodies = BodyGroup([
            RigidBody(segments[1]),
            RigidBody(segments[2]),
            EulerBernoulliBeam(
                LinearModel, segments[3], [ClampBC(BodyPointIndex(1))]; m=1.0, kb=1.0
            ),
        ])

        prob = Problem(fluid, bodies)

        state = initstate(prob)

        # TODO: Test the contents of the quantity, coordinates, etc
        # Right now, we just test for type to make sure the right methods are called

        let vx = flow_velocity(XAxis(DiscretizationFrame()), prob)(state)
            @test vx isa MultiLevelGridValue
            @test coordinates(vx) isa AbstractVector{<:Tuple}
        end

        let vy = flow_velocity(YAxis(DiscretizationFrame()), prob)(state)
            @test vy isa MultiLevelGridValue
            @test coordinates(vy) isa AbstractVector{<:Tuple}
        end

        let ψ = streamfunction(prob)(state)
            @test ψ isa MultiLevelGridValue
            @test coordinates(ψ) isa AbstractVector{<:Tuple}
        end

        let ω = vorticity(prob)(state)
            @test ω isa MultiLevelGridValue
            @test coordinates(ω) isa AbstractVector{<:Tuple}
        end

        let xb = body_point_pos(prob; bodyindex=1)(state)
            @test xb isa AbstractMatrix{Float64}

            y = @view xb[:, 2]
            @test all(>(0), y)
        end

        let xb = body_point_pos(prob; bodyindex=2:3)(state)
            @test xb isa BodyArrayValue

            y = @view xb[2][:, 2]
            @test all(<(0), y)
            @test bodyindices(xb) == 2:3
        end

        let vb = body_point_vel(prob; bodyindex=1)(state)
            @test vb isa AbstractMatrix{Float64}
        end

        let vb = body_point_vel(prob; bodyindex=2:3)(state)
            @test vb isa BodyArrayValue
            @test bodyindices(vb) == 2:3
        end

        let f = body_traction(prob; bodyindex=1)(state)
            @test f isa AbstractMatrix{Float64}
        end

        let f = body_traction(prob; bodyindex=2:3)(state)
            @test f isa BodyArrayValue
            @test bodyindices(f) == 2:3
        end

        let ds = body_lengths(prob; bodyindex=1)(state)
            @test ds isa AbstractVector{Float64}
            @test all(>(0), ds)
        end

        let ds = body_lengths(prob; bodyindex=2:3)(state)
            @test ds isa BodyArrayValue
            @test all(>(0), ds[1])
            @test all(>(0), ds[2])
            @test bodyindices(ds) == 2:3
        end

        @test_throws "not a deforming body" body_deformation(prob; bodyindex=2)
        @test_throws "not a deforming body" body_deformation(prob; bodyindex=1:2)
        @test_throws "Derivative must be" body_deformation(prob; bodyindex=1, deriv=-1)
        @test_throws "Derivative must be" body_deformation(prob; bodyindex=1, deriv=3)

        let χ = body_deformation(prob; bodyindex=3, deriv=0)(state),
            χd = body_deformation(prob; bodyindex=3, deriv=1)(state),
            χdd = body_deformation(prob; bodyindex=3, deriv=2)(state)

            @test size(χ) == size(χd) == size(χdd)
            @test eltype(χ) <: eltype(χd) <: eltype(χdd) <: Float64
        end

        let χ = body_deformation(prob; bodyindex=3:3)(state)
            @test χ isa BodyArrayValue
            @test bodyindices(χ) == 3:3
        end
    end
end

end # module

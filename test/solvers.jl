module TestSolvers

using ImmersedBodies
using ImmersedBodies.Solvers
using ImmersedBodies.Solvers: StaticBodyProblem
using Test

@testset "solvers" begin
    let flow = FreestreamFlow(; Re=123.4)
        @test flow.velocity(0.3) == [0.0, 0.0]
        @test flow.Re == 123.4
    end

    flow = FreestreamFlow(t -> (1.0, -0.1); Re=100.0)

    dx = 0.02
    xspan = (-1, 3)
    yspan = (-2.0, 2.0)
    basegrid = UniformGrid(dx, xspan, yspan)
    grids = MultiLevelGrid(basegrid, 5)

    @test gridstep(basegrid) ≈ dx
    @test gridstep(grids) ≈ dx
    @test gridstep(grids, 3) ≈ 4 * dx

    dt = 0.004
    scheme = CNAB(dt)

    @test timestep(scheme) ≈ dt

    let Umax = 1.5, scheme = default_scheme(grids; Umax, cfl=0.1)
        cfl = Umax * timestep(scheme) / gridstep(grids)
        @test cfl < 0.2
    end

    function max_vel(state)
        qty = state.qty
        nq, nlev = size(qty.q)
        return maximum(1:nlev) do lev
            h = gridstep(grids, lev)
            maximum(1:nq) do i
                abs((qty.q[i, lev] + qty.q0[i, lev]) / h)
            end
        end
    end

    let flow = FreestreamFlow(t -> (t, t); Re=200.0)
        @test default_gridstep(flow) ≈ 0.01 # Grid Re of 2

        grid = UniformGrid(flow, (0.0, 1.0), (-2.0, 2.0))
        @test gridstep(grid) ≈ 0.01 # Floored to 1 significant digit
    end

    @test_throws "invalid reference frame" begin
        PsiOmegaFluidGrid(flow, grids; scheme, frame=DiscretizationFrame())
    end

    let
        fluid = PsiOmegaFluidGrid(flow, basegrid; scheme)
        @test nlevels(discretized(fluid)) == Solvers.DEFAULT_LEVEL_COUNT
    end

    let
        fluid = PsiOmegaFluidGrid(flow, grids; scheme)

        # Body outside of innermost fluid grid
        curve = Curves.LineSegment((0.0, 1.5), (0.0, 2.1))
        body = RigidBody(partition(curve, fluid))
        bodies = BodyGroup([body])

        prob = Problem(fluid, bodies)

        @test_throws "outside innermost fluid grid" solve(prob, (0.0, 5 * dt))
    end

    @testset "static rigid bodies" begin
        fluid = PsiOmegaFluidGrid(flow, grids; scheme)

        @test conditions(fluid) isa FreestreamFlow
        @test discretized(fluid) isa MultiLevelGrid

        curve = Curves.Circle(0.5)
        body = RigidBody(partition(curve, fluid))
        bodies = BodyGroup([body])

        prob = Problem(fluid, bodies)
        @test prob isa StaticBodyProblem

        state = initstate(prob)

        solve!(state, prob, 10 * dt)
        @test max_vel(state) < 20
    end

    @testset "moving grid" begin
        frame = OffsetFrame(GlobalFrame()) do t
            r = 0.1 .* (cos(t), sin(t))
            v = 0.1 .* (-sin(t), cos(t))
            θ = 0.2 * sin(t)
            Ω = 0.2 * cos(t)
            return OffsetFrameInstant(r, v, θ, Ω)
        end

        fluid = PsiOmegaFluidGrid(flow, grids; scheme, frame)

        curve = Curves.Circle(0.5)
        body = RigidBody(partition(curve, fluid))
        bodies = BodyGroup([body])

        prob = Problem(fluid, bodies)
        @test prob isa StaticBodyProblem

        state = initstate(prob)

        solve!(state, prob, 10 * dt)
        @test max_vel(state) < 20
    end

    @testset "moving rigid bodies" begin
        fluid = PsiOmegaFluidGrid(flow, grids; scheme)

        function offset(t)
            r = 0.1 .* (cos(t), sin(t))
            v = 0.1 .* (-sin(t), cos(t))
            θ = 0.2 * sin(t)
            Ω = 0.2 * cos(t)
            return OffsetFrameInstant(r, v, θ, Ω)
        end
        body1 = RigidBody(
            partition(Curves.LineSegment((-0.3, 0.5), (0.3, 0.5)), fluid),
            OffsetFrame(offset, DiscretizationFrame()),
        )
        body2 = RigidBody(
            partition(Curves.LineSegment((-0.3, -0.5), (0.3, -0.5)), fluid),
            OffsetFrame(offset, GlobalFrame()),
        )
        bodies = BodyGroup([body1, body2])

        prob = Problem(fluid, bodies)
        @test !(prob isa StaticBodyProblem)

        state = initstate(prob)

        solve!(state, prob, 10 * dt)
        @test max_vel(state) < 20
    end

    @testset "linear deforming body" begin
        fluid = PsiOmegaFluidGrid(flow, grids; scheme)

        segments1 = partition(Curves.LineSegment((-0.2, 0.5), (0.2, 0.5)), fluid)
        body1 = RigidBody(segments1)

        bcs = ClampParameterBC.([0.0, 1.0])
        segments2 = partition(Curves.LineSegment((-0.3, -0.5), (0.3, -0.5)), fluid)
        body2 = EulerBernoulliBeamBody(
            segments2, bcs; model=LinearModel(), kb=1.0, ke=1.0, m=1.0
        )

        bodies = BodyGroup([body1, body2])

        prob = Problem(fluid, bodies)
        @test !(prob isa StaticBodyProblem)

        state = initstate(prob)

        solve!(state, prob, 10 * dt)
        @test max_vel(state) < 20
    end
end

end # module

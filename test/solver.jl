@testset "solver" begin
    motion_choices = [
        "static grid" => StaticGrid(),
        "moving grid" => MovingGrid() do t
            GridVelocity(;
                center=(-1.0, 0.0),
                vel=(0.0, 0.2),
                angle=0.1 * sin(t),
                angular_vel=0.1 * cos(t),
            )
        end,
    ]

    body_choices = [
        "single cyl" =>
            fluid -> Bodies(;
                cyl=StaticBody(
                    Panels(fluid.grid, Curves.Circle(; r=0.5, center=(0.0, 0.0)))
                ),
            ),
        "double line" =>
            fluid -> Bodies(;
                line1=StaticBody(
                    Panels(fluid.grid, Curves.LineSegment((-0.2, 0.4), (0.2, 0.1)))
                ),
                line2=StaticBody(
                    Panels(fluid.grid, Curves.LineSegment((-0.2, -0.1), (0.2, -0.4)))
                ),
            ),
        "moving cyl" =>
            fluid -> Bodies(;
                cyl=MovingRigidBody(
                    Panels(fluid.grid, Curves.Circle(; r=0.5, center=(0.0, 0.0)))
                ) do t
                    RigidBodyTransform(;
                        pos=(0.3 * cos(t), 0.1 * sin(t)),
                        vel=(-0.3 * sin(t), 0.1 * cos(t)),
                        angle=0.1 * t,
                        angular_vel=0.1,
                    )
                end,
            ),
    ]

    @testset "solving ($i, $j)" for (i, motion) in motion_choices,
        (j, bodies_fn) in body_choices

        let fluid = Fluid(;
                grid=fluid_grid(; xlims=(-1.0, 2.0), ylims=(-1.5, 1.5), nlevel=3),
                Re=100.0,
                freestream_vel=t -> (1.0, 0.0),
                grid_motion=motion,
            ),
            bodies = bodies_fn(fluid),
            scheme = default_scheme(fluid.grid; Umax=2.0, cfl=0.2),
            prob = Problem(; fluid, bodies, scheme),
            state = State(prob)

            solver = create_solver(state)

            solve!(solver; t=3 * timestep(prob))
            @test state.t ≈ 3 * timestep(prob)
        end
    end

    let fluid = Fluid(;
            grid=fluid_grid(; h=0.1, xlims=(-1.0, 0.5), ylims=(-1.0, 1.0), nlevel=3),
            Re=20.0,
            freestream_vel=t -> (1.0, 0.0),
        ),
        bodies = Bodies(;
            cyl=StaticBody(Panels(fluid.grid, Curves.Circle(; r=0.5, center=(0.0, 0.0))))
        ),
        scheme = default_scheme(fluid.grid; Umax=2.0, cfl=0.2),
        prob = Problem(; fluid, bodies, scheme),
        state = State(prob)

        @test_throws "outside allowed region" create_solver(state)
    end
end

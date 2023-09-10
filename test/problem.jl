@testset "problem" begin
    using ImmersedBodies: coords, gridstep, subdomain, timestep

    let grid = fluid_grid(; h=0.1, xlims=(-2, 2), ylims=(0, 1), nlevel=3),
        basegrid = grid.base

        @test gridstep(grid) == gridstep(basegrid) == 0.1
        @test subdomain(grid, 1) == basegrid
        @test coords(subdomain(grid, 1)) == (-2:0.1:2, 0:0.1:1)
        @test coords(subdomain(grid, 2)) == (-4:0.2:4, -0.5:0.2:1.5)

        foreach((GridU(), x_velocity, x_velocity!)) do pts
            @test coords(basegrid, pts) == (-2:0.1:2, 0.05:0.1:0.95)
        end
        foreach((GridV(), y_velocity, y_velocity!)) do pts
            @test coords(basegrid, pts) == (-1.95:0.1:1.95, 0:0.1:1)
        end
        foreach((GridΓ(), vorticity, vorticity!)) do pts
            @test coords(basegrid, pts) == (-1.9:0.1:1.9, 0.1:0.1:0.9)
        end
    end

    let Re = 123.0,
        fluid = Fluid(; grid=fluid_grid(; xlims=(0, 4), ylims=(0, 4), nlevel=1), Re)

        @test gridstep(fluid.grid) ≈ ImmersedBodies.default_gridstep(Re)
    end

    let line1 = StaticBody(Panels(; xb=[0:0.1:1 fill(0, 11)], ds=fill(0.1, 11))),
        line2 = StaticBody(Panels(; xb=[fill(0, 21) 0:0.1:2], ds=fill(0.1, 21))),
        bodies = Bodies(; line1, line2)

        @test !any_fsi(bodies)
        @test panel_range(bodies, :line1) == 1:11
        @test panel_range(bodies, :line2) == 12:32
        @test bodies[:line1] == line1
        @test bodies[:line2] == line2
    end

    let grid = MultiDomainGrid(; h=0.1, xlims=(0, 1), ylims=(0, 1), nlevel=2),
        target_cfl = 0.2,
        Umax = 1.0,
        scheme = default_scheme(grid; Umax, cfl=target_cfl),
        cfl = Umax * timestep(scheme) / gridstep(grid)

        @test cfl <= target_cfl
    end

    let fluid = Fluid(;
            grid=fluid_grid(; h=0.2, xlims=(-1.0, 2.0), ylims=(-1.5, 1.5), nlevel=5),
            Re=10.0,
            freestream_vel=t -> (1.0, 0.0),
            grid_motion=MovingGrid() do t
                GridVelocity(;
                    center=(0.0, 1.0), vel=(-1.0, 0.0), angle=t, angular_vel=1.0
                )
            end,
        ),
        bodies = Bodies(;
            cyl=StaticBody(Panels(fluid.grid, Curves.Circle(; r=0.5, center=(0.0, 0.0)))),
            line=StaticBody(
                Panels(fluid.grid, Curves.LineSegment((1.0, -0.3), (1.0, 0.3)))
            ),
        ),
        scheme = CNAB(; dt=0.1),
        prob = Problem(; fluid, bodies, scheme)

        @test gridstep(prob) == 0.2
        @test timestep(prob) == 0.1
    end
end

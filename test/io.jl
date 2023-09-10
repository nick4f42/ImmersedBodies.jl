# A dummy problem so we can create a small state for testing
function dummy_problem()
    fluid = Fluid(;
        grid=fluid_grid(; h=0.5, xlims=(-1.0, 1.0), ylims=(-1.0, 1.0), nlevel=2),
        Re=100.0,
        freestream_vel=t -> (1.0, 0.0),
    )
    bodies = Bodies(;
        cyl=StaticBody(Panels(fluid.grid, Curves.Circle(; r=0.5, center=(0.0, 0.0))))
    )
    scheme = CNAB(; dt=1.0)
    Problem(; fluid, bodies, scheme)
end

@testset "io" begin
    using ImmersedBodies: update_vars!, set_time_index!, inc_time_index!, extents

    prob = dummy_problem()
    dt = ImmersedBodies.timestep(prob)

    # A minimal, dummy solver for just testing IO
    struct TestSolver{S<:State} <: AbstractSolver
        state::S
    end
    ImmersedBodies.advance!(solver::TestSolver) = inc_time_index!(solver.state)
    ImmersedBodies.getstate(solver::TestSolver) = solver.state

    @testset "state" begin
        state = State(prob; t0=32.5)

        set_time_index!(state, 42)
        state.q[:] = 0.01 .+ eachindex(state.q)
        state.q0[:] = 0.02 .+ eachindex(state.q0)
        state.Γ[:] = 0.03 .+ eachindex(state.Γ)
        state.ψ[:] = 0.04 .+ eachindex(state.ψ)
        state.nonlin[1][:] = 0.05 .+ eachindex(state.nonlin[1])
        state.nonlin[2][:] = 0.06 .+ eachindex(state.nonlin[2])
        state.panels.xb[:] = 0.07 .+ eachindex(state.panels.xb)
        state.panels.ds[:] = 0.08 .+ eachindex(state.panels.ds)
        state.panels.ub[:] = 0.09 .+ eachindex(state.panels.ub)
        state.F̃b[:] = 0.10 .+ eachindex(state.F̃b)
        state.freestream_vel = (-1.0, -2.0)
        update_vars!(state)

        let filename = tempname()
            save_state(filename, state)
            @test state == load_state(filename, prob)
        end
    end

    @testset "HDF5" begin
        using HDF5

        function save(filename)
            SaveHDF5(
                filename,
                "prob" => prob,
                "foo/bar" => SolutionValues(
                    TimestepRange(; step=5),
                    "x0" => ArrayValue(Int, ()) do x, state
                        x[] = state.index
                    end,
                    "x1" => ArrayValue(Float64, (2,)) do x, state
                        x[1] = state.index
                        x[2] = 1.23
                    end,
                ),
                "a" => SolutionValues(
                    TimestepRange(; start=4, step=2, length=3),
                    "x2" => ArrayValue(Float64, (2, 3)) do x, state
                        x .= state.index .* reshape(1:6, 2, 3)
                    end,
                    "y" => MultiDomainValue(
                        (y, state) -> y .= state.index,
                        [
                            (0.0:0.1:0.5, 0.0:0.1:0.1),
                            (0.0:0.2:1.0, 0.0:0.2:0.2),
                            (0.0:0.4:2.0, 0.0:0.4:0.4),
                        ],
                    ),
                ),
                "b" => SolutionValues(
                    TimestepRange(; start=20, step=1, length=5),
                    "z0" => ArrayValue(Int, ()) do z, state
                        z[] = 125
                    end,
                ),
            )
        end

        function test_soln(filename, n::Int)
            h5open(filename) do file
                let inds = 1:5:n, times = dt .* (inds .- 1)
                    @test read(file["foo/bar/_timestep_time"]) ≈ times
                    @test read(file["foo/bar/_timestep_index"]) == inds
                    @test read(file["foo/bar/x0"]) == inds
                    @test read(file["foo/bar/x1"]) ≈ hcat(([i, 1.23] for i in inds)...)
                end

                let inds = 4:2:min(8, n), times = dt .* (inds .- 1)
                    @test read(file["a/_timestep_time"]) ≈ times
                    @test read(file["a/_timestep_index"]) == inds
                    @test read(file["a/x2"]) ≈
                        cat((i * reshape(1:6, 2, 3) for i in inds)...; dims=3)
                    @test read(file["a/y"]) ≈
                        [i for _ in 1:6, _ in 1:2, _ in 1:3, i in inds]
                end

                let inds = 20:n, times = dt .* (inds .- 1)
                    @test read(file["b/_timestep_time"]) ≈ times
                    @test read(file["b/_timestep_index"]) == inds
                    @test read(file["b/z0"]) == fill(125, length(inds))
                end

                coords = map(
                    NamedTuple{(:start, :step, :length)},
                    [
                        (0.0, 0.1, 6) (0.0, 0.2, 6) (0.0, 0.4, 6)
                        (0.0, 0.1, 2) (0.0, 0.2, 2) (0.0, 0.4, 2)
                    ],
                )
                @test read_attribute(file["a/y"], "coords") == coords
            end
        end

        # Make sure that solving over new timesteps expands the solution file, and that
        # solving over the same timesteps correctly overwrites the solution
        let filename = tempname(),
            state = State(prob),
            ranges = [(0, 6, 6), (2, 22, 22), (5, 10, 22)]

            h5open(filename, "w") do file
                file["leave_me_alone"] = true
            end

            @testset "solve from $(i + 1) to $n" for (i, n, nmax) in ranges
                state.index = i
                state.t = dt * (i - 1)
                solve!(TestSolver(state); t=(n - 1) * dt, save=save(filename))
                test_soln(filename, nmax)
            end

            h5open(filename) do file
                @test read(file["leave_me_alone"]) === true

                let grid = prob.fluid.grid,
                    h = grid.base.h,
                    (xlims, ylims) = map(NamedTuple{(:min, :max)}, extents(grid.base)),
                    nlevel = grid.nlevel

                    @test read(file["prob/fluid"]) ==
                        (; grid=(; h, xlims, ylims, nlevel), Re=prob.fluid.Re)
                end

                let body = prob.bodies[:cyl], cyl = file["prob/bodies/cyl"]
                    @test read_attribute(cyl, "type") == "StaticBody"
                    @test read(cyl["positions"]) == body.panels.xb
                    @test read(cyl["lengths"]) == body.panels.ds
                end

                let scheme = file["prob/scheme"]
                    @test read_attribute(scheme, "type") == "CNAB"
                    @test read(scheme) == (; dt=prob.scheme.dt)
                end
            end
        end
    end
end

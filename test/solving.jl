module TestSolving

using ImmersedBodies
using ImmersedBodies.Quantities
using Test
using HDF5

function placeholder_problem()
    flow = FreestreamFlow(t -> (1.0, 0.0); Re=50.0)

    dx = 0.05
    xspan = (-1.0, 1.0)
    yspan = (-1.0, 1.0)
    basegrid = UniformGrid(dx, xspan, yspan)
    grids = MultiLevelGrid(basegrid, 2)

    fluid = PsiOmegaFluidGrid(flow, grids; scheme=CNAB(0.005))

    curve = Curves.Circle(0.3)
    body = RigidBody(partition(curve, fluid))
    bodies = BodyGroup([body])

    return Problem(fluid, bodies)
end

@testset "solve function" begin
    prob = placeholder_problem()
    dt = timestep(prob)
    tspan = (0.0, 20 * dt)

    callback_times = (
        t_all=AllTimesteps(),
        i_vec=timestep_indices([-1, 1, 2, 22]),
        i_range=timestep_indices(; start=2, step=5),
        t_vec=timestep_times([-0.1dt, 0.3dt, 0.8dt, 1.2dt, 9.2dt, 20.6dt]),
        t_range=timestep_times(; start=16.4dt, step=0.4dt),
        t_edge=timestep_times([0.5dt]), # could round up or down
    )
    actual_indices = map(_ -> Int[], callback_times)
    callbacks = map(callback_times, actual_indices) do t, i
        Callback(t) do state
            push!(i, timeindex(state))
        end
    end

    function test_callbacks()
        (; t_all, i_vec, i_range, t_vec, t_range, t_edge) = actual_indices
        @test t_all == 1:21
        @test i_vec == [1, 2]
        @test i_range == 2:5:21
        @test t_vec == [1, 2, 2, 10]
        @test t_range == round.(Int, 1.0 .+ (16.4:0.4:20.0))
        @test t_edge == [1] || t_edge == [2]
    end

    value_inds = [2, 5, 10]
    value_times = dt .* (value_inds .- 1)
    value_group = ValueGroup(
        timestep_indices(value_inds);
        a=_ -> 42,
        b=timeindex,
        c=ArrayQuantity(state -> [1.0, 2.0]),
        d=GridQuantity(state -> [1 2; 3 4], (LinRange(0, 1, 2), LinRange(0, 1, 2))),
        e=MultiLevelGridQuantity(
            state -> ones(2, 2, 3), [(LinRange(0, i, 2), LinRange(0, i, 2)) for i in 1:3]
        ),
    )

    function test_values(vals)
        for v in vals
            @test timevalue(v) ≈ value_times atol = 1e-8
        end

        @test vals.a isa ArrayValues
        @test eltype(vals.a) <: Int
        @test vals.a == [42 for _ in 1:3]

        @test vals.b isa ArrayValues
        @test eltype(vals.b) <: Int
        @test vals.b == value_inds

        @test vals.c isa ArrayValues
        @test eltype(vals.c) <: AbstractArray{Float64}
        @test vals.c == [[1.0, 2.0] for _ in 1:3]

        @test vals.d isa GridValues
        @test eltype(vals.d) <: AbstractArray{Int}
        @test vals.d == [[1 2; 3 4] for _ in 1:3]
        @test coordinates(vals.d) == (LinRange(0, 1, 2), LinRange(0, 1, 2))

        @test vals.e isa MultiLevelGridValues
        @test eltype(vals.e) <: AbstractArray{Float64}
        @test vals.e == [ones(2, 2, 3) for _ in 1:3]
        @test coordinates(vals.e) == [(LinRange(0, i, 2), LinRange(0, i, 2)) for i in 1:3]
    end

    fn = tempname()

    save = Dict("nested" => Dict("values" => value_group))
    out = (vals=value_group,)

    soln = redirect_stderr(devnull) do # Prevent progress bar output
        solve(fn, prob, tspan; out=out, call=callbacks, save=save, showprogress=true)
    end

    @testset "callbacks" begin
        test_callbacks()
    end

    @testset "return values" begin
        test_values(soln.vals)
    end

    @testset "HDF5 saving" begin
        h5open(fn, "r+") do file
            file["not-a-quantity"] = 126.125
            @test_throws AssertionError quantity_values(file["not-a-quantity"])

            group = file["nested/values"]
            vals = NamedTuple(
                k => quantity_values(group[string(k)]) for k in keys(value_group.quantities)
            )

            # Ensure indexing a value read from file returns a value in memory
            @test vals.a[1] isa Int
            @test vals.b[1] isa Int
            @test vals.c[1] isa Vector{Float64}
            @test vals.d[1].array isa Array{Int,2}
            @test vals.e[1].array isa Array{Float64,3}

            test_values(vals)
        end
    end

    rm(fn)
end

end # module

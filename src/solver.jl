"""
    AbstractSolver

Solves a [`Problem`](@ref).
"""
abstract type AbstractSolver end

"""
    advance!(::AbstractSolver)

Compute the state at the next time step in place.
"""
function advance! end

"""
    getstate(::AbstractSolver)

The [`State`](@ref) of a solver.
"""
function getstate end

"""
    create_solver(::Problem) :: AbstractSolver
    create_solver(::State) :: AbstractSolver

Return the correct [`AbstractSolver`](@ref) for a problem or state.
"""
create_solver(prob::Problem; kw...) = create_solver(State(prob); kw...)

"""
    solve!([f], solver::AbstractSolver; t, save=SaveNothing(), log=true)

Solve for the state up to time `t`, calling the function `f(state, i, n)` at each timestep
`i` from `1:n`. `save` is a [`SolutionSaver`](@ref) that determins how to write the
solution.
"""
function solve!(
    f, solver::AbstractSolver; t::Real, save::SolutionSaver=SaveNothing(), log::Bool=true
)
    state = getstate(solver)
    prob = state.prob
    n_timestep = round(Int, (t - state.t) / timestep(prob))
    max_timestep = state.index + n_timestep

    saver = _init(save, max_timestep)
    try
        for i in 1:n_timestep
            advance!(solver)
            _update(saver, state)
            f(state, i, n_timestep)

            # TODO: Add more monitoring functionality
            if log && i % 10 == 1
                println("i = ", state.index, "  t = ", state.t)
            end
        end
    finally
        _finalize(saver)
    end

    nothing
end

solve!(solver::AbstractSolver; kw...) = solve!((_...) -> nothing, solver; kw...)

"""
    solve([f], prob::Problem; t, [save], [solver_kw])

Solve the problem from time 0 to `t`. See [`solve!`](@ref).
"""
function solve(f, prob::Problem; solver_kw=(), kw...)
    solver = create_solver(State(prob); solver_kw...)
    solve!(f, solver; kw...)
end

solve(prob::Problem; kw...) = solve((_...) -> nothing, prob; kw...)

struct CnabSolver{F} <: AbstractSolver
    state::State{CNAB}
    qs::Matrix{Float64}  # Trial flux
    Γs::Matrix{Float64}  # Trial circulation
    reg::Regularization
    fn::F
    function CnabSolver(state::State{CNAB}; fftw=(), cg=())
        prob = state.prob
        grid = prob.fluid.grid

        (; nx, ny, nΓ, nq) = grid.inds
        nlevel = grid.nlevel
        npanel = n_panels(prob.bodies)

        # TODO: Overlap more memory if possible
        qs = zeros(nq, nlevel)
        Γs = zeros(nΓ, nlevel)
        Γbc = zeros(2 * (nx + 1) + 2 * (ny + 1))
        Γtmp = zeros(nΓ, nlevel)
        ψtmp = zeros(nΓ, nlevel)
        qtmp = zeros(nq, nlevel)
        Ftmp = zeros(2 * npanel)
        Γtmp1 = zeros(nΓ)
        Γtmp2 = zeros(nΓ)
        Γtmp3 = zeros(nΓ)
        qtmp1 = zeros(nq)
        qtmp2 = zeros(nq)
        Γdst1 = zeros((nx - 1, ny - 1))
        Γdst2 = zeros((nx - 1, ny - 1))

        dst_plan = _dst_plan(
            Γdst1; flags=FFTW.EXHAUSTIVE, num_threads=Threads.nthreads(), fftw...
        )
        lap_eigs = _lap_eigs(grid.inds)
        Δinv! = Δinv_operator(grid.inds, dst_plan, lap_eigs; Γtmp1=Γdst1, Γtmp2=Γdst2)

        vort2flux! = vort2flux(grid; Δinv!, ψ_bc=Γbc, Γ_tmp=Γtmp1)
        rhs_force! = rhs_force(grid; q_tmp=qtmp1)
        nonlinear! = nonlinear(grid; rhs_force!, fq=qtmp2)

        reg = Regularization(prob)

        # Set initial panel state and update reg
        _init_body_motion!(state, reg)

        A, Ainv = A_operators(prob, dst_plan, lap_eigs; Γtmp1=Γdst1, Γtmp2=Γdst2)

        B! = B_operator(
            prob;
            reg,
            vort2flux!,
            (Ainv1!)=Ainv[1],
            Γ1_tmp=Γtmp2,
            q1_tmp=qtmp2,
            Γ_tmp=Γtmp,
            ψ_tmp=ψtmp,
            q_tmp=qtmp,
        )

        Binv! = Binv_operator(prob; B!, cg_kw=(; maxiter=5000, reltol=1e-12, cg...))

        get_trial_state! = get_trial_state(
            prob; nonlinear!, vort2flux!, A, Ainv, rhsbc=Γtmp2, rhs=Γtmp3, bc=Γbc
        )

        couple_surface! = couple_surface_rigid(prob, grid; Binv!, reg, Ftmp=Ftmp, Q=qtmp1)

        project_circ! = project_circ(grid; (Ainv1!)=Ainv[1], reg, Γtmp=Γtmp2, qtmp=qtmp1)

        fn = (; vort2flux!, get_trial_state!, couple_surface!, project_circ!)
        new{typeof(fn)}(state, qs, Γs, reg, fn)
    end
end

create_solver(state::State{CNAB}; kw...) = CnabSolver(state; kw...)
getstate(solver::CnabSolver) = solver.state

function advance!(solver::CnabSolver)
    state = solver.state
    prob = state.prob

    (; fn, qs, Γs) = solver

    # Increment to next time
    inc_time_index!(state)

    # Update freestream velocity
    state.freestream_vel = prob.fluid.freestream_vel(state.t)

    # Update the preset body motion and regularization matrix
    _update_body_motion!(state, solver.reg)

    # Base flux from freestream and grid frame movement
    base_flux!(state)

    # Computes trial circulation Γs and associated strmfcn and vel flux that don't satisfy
    # no-slip (from explicitly treated terms)
    fn.get_trial_state!(qs, Γs, state)

    # Couple the surface between fluid and structure
    fn.couple_surface!(state, qs)

    # Update circulation, vel-flux, and strmfcn on fine grid to satisfy no-slip
    fn.project_circ!(state, Γs)

    # Interpolate values from finer grid to center region of coarse grid
    fn.vort2flux!(state.ψ, state.q, state.Γ)

    # Update solution variables
    update_vars!(state)

    nothing
end

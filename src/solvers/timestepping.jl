struct GetTrialState{F<:PsiOmegaFluidGrid,N<:Nonlinear,V<:Vort2Flux,MA,MAinv}
    fluid::F
    scheme::CNAB
    nonlinear::N
    vort2flux::V
    As::Vector{MA}
    Ainvs::Vector{MAinv}
    rhsbc::Vector{Float64}
    rhs::Vector{Float64}
    bc::Vector{Float64}
end

function GetTrialState(;
    prob::Problem{<:PsiOmegaFluidGrid{CNAB}},
    nonlinear,
    vort2flux,
    As,
    Ainvs,
    rhsbc,
    rhs,
    bc,
)
    return GetTrialState(
        prob.fluid, timestep_scheme(prob), nonlinear, vort2flux, As, Ainvs, rhsbc, rhs, bc
    )
end

function (gettrial::GetTrialState)(
    qs::AbstractMatrix, Γs::AbstractMatrix, state::StatePsiOmegaGridCNAB
)
    (; As, Ainvs, bc, rhs, rhsbc, fluid, scheme) = gettrial
    nonlinear!, vort2flux! = gettrial.nonlinear, gettrial.vort2flux

    grid = discretized(fluid)
    nlevel = nlevels(grid)

    dt = timestep(scheme)
    Re = conditions(fluid).Re

    nonlin = state.nonlin
    qty = quantities(state)

    for lev in nlevel:-1:1
        bc .= 0
        rhsbc .= 0
        hc = gridstep(grid, lev)

        if lev < nlevel
            @views get_bc!(bc, qty.Γ[:, lev + 1], fluid.gridindex)

            fac = 0.25 * dt / (Re * hc^2)
            apply_bc!(rhsbc, bc, fac, fluid.gridindex)
        end

        # Account for scaling between grids
        # Don't need bc's for anything after this, so we can rescale in place
        bc .*= 0.25

        #compute the nonlinear term for the current time step
        @views nonlinear!(nonlin[1][:, lev], qty, bc, lev)

        @views mul!(rhs, As[lev], qty.Γ[:, lev])

        for n in 1:length(scheme.β)
            @. rhs += dt * scheme.β[n] * (@view nonlin[n][:, lev])
        end

        # Include boundary conditions
        rhs .+= rhsbc

        # Trial circulation  Γs = Ainvs * rhs
        @views mul!(Γs[:, lev], Ainvs[lev], rhs)
    end

    # Store nonlinear solution for use in next time step
    # Cycle nonlinear arrays
    cycle!(nonlin)

    vort2flux!(qty.ψ, qs, Γs)
    return nothing
end

# Rotate elements in `a` forward an index
cycle!(a::Vector) = isempty(a) ? a : pushfirst!(a, pop!(a))

Base.@kwdef struct CircProjecter{MAinv,MC,ME}
    Ainv::MAinv
    C::MC
    E::ME
    Γtmp::Vector{Float64}
    qtmp::Vector{Float64}
end

"""
    project_circ!(Γs, state, prob)

Update circulation to satisfy no-slip condition.

This allows precomputing regularization and interpolation where possible.
"""
function (cproj::CircProjecter)(qty::PsiOmegaGridQuantities, Γs::AbstractMatrix)
    # High-level version:
    #     Γ = Γs - Ainv * (E*C)'*F̃b

    Γs1 = @view Γs[:, 1]
    (; Ainv, C, E, Γtmp, qtmp) = cproj

    qty.Γ .= Γs

    mul!(qtmp, E', view(qty.F̃b, :, 1))
    mul!(Γtmp, C', qtmp)
    mul!(Γs1, Ainv, Γtmp) # use Γs as temporary buffer
    @views qty.Γ[:, 1] .-= Γs1

    return nothing
end

struct SolverPsiOmegaGridCNAB{
    P<:Problem{<:PsiOmegaFluidGrid{CNAB}},
    G<:GetTrialState,
    S<:SurfaceCoupler,
    C<:CircProjecter,
    V<:Vort2Flux,
} <: AbstractSolver
    prob::P
    qs::Matrix{Float64} # Trial flux
    Γs::Matrix{Float64} # Trial circulation
    reg::Reg
    get_trial_state!::G
    couple_surface!::S
    project_circ!::C
    vort2flux!::V
end

function SolverPsiOmegaGridCNAB(
    prob::Problem{<:PsiOmegaFluidGrid{CNAB}}, state::StatePsiOmegaGridCNAB
)
    (; nx, ny, nΓ, nq) = prob.fluid.gridindex

    grid = discretized(prob.fluid)
    basegrid = baselevel(grid)
    nlevel = nlevels(grid)

    # TODO: Overlap more memory if possible

    qs = zeros(nq, nlevel)
    Γs = zeros(nΓ, nlevel)
    Γbc = zeros(2 * (nx + 1) + 2 * (ny + 1))
    Γtmp = zeros(nΓ, nlevel)
    ψtmp = zeros(nΓ, nlevel)
    qtmp = zeros(nq, nlevel)
    Ftmp = zeros(2 * npanels(prob.bodies))
    Γtmp1 = zeros(nΓ)
    Γtmp2 = zeros(nΓ)
    Γtmp3 = zeros(nΓ)
    qtmp1 = zeros(nq)
    qtmp2 = zeros(nq)

    C = C_linearmap(prob.fluid.gridindex)

    lap_inv = LaplacianInv(prob.fluid.gridindex)
    Δinv = lap_inv_linearmap(lap_inv)

    vort2flux = Vort2Flux(; fluid=prob.fluid, Δinv=Δinv, ψbc=Γbc, Γtmp=Γtmp1)

    rhs_force = RhsForce(; prob.fluid.gridindex, Q=qtmp1)
    nonlinear = Nonlinear(; rhs_force, grid=discretized(prob.fluid), C, fq=qtmp2)

    # reg must be updated if body points move relative to the grid
    panels = quantities(state).panels
    reg = Reg(prob, panels)
    E = E_linearmap(reg)

    As, Ainvs = A_Ainv_linearmaps(prob, lap_inv)

    B_times = B_Times(;
        vort2flux, Ainv=Ainvs[1], E, C, Γtmp=Γtmp2, qtmp=qtmp2, Γ=Γtmp, ψ=ψtmp, q=qtmp
    )
    B = B_linearmap(prob.bodies, B_times)

    get_trial_state = GetTrialState(;
        prob, nonlinear, vort2flux, As, Ainvs, rhsbc=Γtmp2, rhs=Γtmp3, bc=Γbc
    )

    couple_surface = if prob.bodies isa BodyGroup{<:RigidBody}
        Binv = Binv_linearmap(prob, B)
        RigidSurfaceCoupler(; basegrid=basegrid, Binv, E, Ftmp=Ftmp, Q=qtmp1)
    else
        nb = npanels(prob.bodies)

        deform_body = only(prob.bodies)
        mats = StructuralMatrices(deform_body)
        Itilde = Itilde_linearmap(nb)
        redist = RedistributionWeights(; E, qtmp=zeros(nq))

        nb_deform = nb
        nf_deform = 2 * nb_deform
        χ_k = zeros(nf_deform)
        ζ_k = zeros(nf_deform)
        ζdot_k = zeros(nf_deform)

        EulerBernoulliSurfaceCoupler(;
            prob, qtot=qtmp1, mats, reg, redist, E, B, Itilde, χ_k, ζ_k, ζdot_k
        )
    end

    project_circ = CircProjecter(; Ainv=Ainvs[1], C, E, Γtmp=Γtmp2, qtmp=qtmp1)

    return SolverPsiOmegaGridCNAB(
        prob, qs, Γs, reg, get_trial_state, couple_surface, project_circ, vort2flux
    )
end

solvertype(::Problem{<:PsiOmegaFluidGrid{CNAB}}) = SolverPsiOmegaGridCNAB

function prescribe_motion!(state::StatePsiOmegaGridCNAB, solver::SolverPsiOmegaGridCNAB)
    prob = solver.prob
    bodies = prob.bodies
    fluidframe = prob.fluid.frame
    panels = quantities(state).panels

    for (i, (bodypanels, body)) in enumerate(zip(panels.perbody, bodies))
        if body isa RigidBody && !is_static(body, fluidframe)
            prescribe_motion!(bodypanels, fluidframe, body, state.t)
            update!(solver.reg, bodypanels, i)
        end
    end
end

function advance!(state::StatePsiOmegaGridCNAB, solver::SolverPsiOmegaGridCNAB)
    return advance!(state, solver, state.i + 1)
end

function advance!(state::StatePsiOmegaGridCNAB, solver::SolverPsiOmegaGridCNAB, index::Int)
    prob = solver.prob
    qty = quantities(state)

    # Increment time to next timestep
    state.i = index
    state.t = state.t0 + timestep(prob) * (index - 1) # index starts at 1

    # Update freestream velocity
    qty.u .= conditions(prob.fluid).velocity(state.t)

    # Trial flux and circulation
    (; qs, Γs) = solver

    # If necessary, prescribe rigid-body motion and update the regularization matrix
    prescribe_motion!(state, solver)

    # Base flux from freestream and grid frame movement
    base_flux!(qty, prob.fluid, state.t)

    # Computes trial circulation Γs and associated strmfcn and vel flux that don't satisfy
    # no-slip (from explicitly treated terms)
    solver.get_trial_state!(qs, Γs, state)

    # Couple the surface between fluid and structure
    solver.couple_surface!(state, qs)

    # Set traction to reflect the new F̃b
    update_traction!(qty, prob)

    # Update circulation, vel-flux, and strmfcn on fine grid to satisfy no-slip
    solver.project_circ!(qty, Γs)

    # Interpolate values from finer grid to center region of coarse grid
    solver.vort2flux!(qty)

    return nothing
end

function solve!(
    f, state::StatePsiOmegaGridCNAB, prob::Problem{<:PsiOmegaFluidGrid}, tf::Float64
)
    solver = SolverPsiOmegaGridCNAB(prob, state)
    t0 = timevalue(state)
    dt = timestep(solver.prob)

    for i in 1:timestep_count(prob, (t0, tf))
        advance!(state, solver, i)
        f(state)
    end
end

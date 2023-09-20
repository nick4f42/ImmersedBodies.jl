function split_flux(q::AbstractArray, inds::GridIndices)
    (; nx, ny, nu) = inds

    u_flat = @view q[1:nu, ..]
    v_flat = @view q[(nu + 1):end, ..]

    dims = size(q)[2:end]
    u = @views reshape(u_flat, nx + 1, ny, dims...)
    v = @views reshape(v_flat, nx, ny + 1, dims...)
    (u, v)
end

split_flux(q, inds, lev) = @views split_flux(q[:, lev], inds)

function unflatten_circ(Γ::AbstractArray, inds::GridIndices)
    (; nx, ny) = inds
    reshape(Γ, nx - 1, ny - 1, size(Γ)[2:end]...)
end

unflatten_circ(Γ, inds, lev) = @views unflatten_circ(Γ[:, lev], inds)

function base_flux!(state::State{<:Any,StaticGrid})
    grid = state.prob.fluid.grid
    u, v = state.freestream_vel
    qx, qy = split_flux(state.q0, grid.inds)

    for lev in 1:(grid.nlevel)
        hc = gridstep(grid, lev)  # Coarse grid spacing
        qx[:, :, lev] .= u * hc
        qy[:, :, lev] .= v * hc
    end
end

function base_flux!(state::State{<:Any,MovingGrid})
    fluid = state.prob.fluid
    grid = fluid.grid

    qx, qy = split_flux(state.q0, grid.inds)

    uinf = state.freestream_vel

    motion = fluid.grid_motion(state.t)
    c = motion.center
    u0 = motion.vel
    Ω = motion.angular_vel
    cθ = cos(motion.angle)
    sθ = sin(motion.angle)
    Rx = @SMatrix [cθ -sθ; sθ cθ]  # Basis of relative frame in global frame
    Rv = Ω * @SMatrix [-sθ -cθ; cθ -sθ]  # Affect of rotation on velocity

    for lev in 1:(grid.nlevel)
        subgrid = subdomain(grid, lev)
        hc = gridstep(subgrid)

        # Transform from freestream velocity (ux, uy) in global frame to relative frame
        let (xs, ys) = coords(subgrid, GridU())
            for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
                qx[i, j, lev] = hc * dot(Rx[:, 1], uinf - u0 - Rv * (SVector(x, y) - c))
            end
        end
        let (xs, ys) = coords(subgrid, GridV())
            for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
                qy[i, j, lev] = hc * dot(Rx[:, 2], uinf - u0 - Rv * (SVector(x, y) - c))
            end
        end
    end
end

function curl!(q, ψ_flat, inds::GridIndices)
    (; nx, ny) = inds
    ψ = unflatten_circ(ψ_flat, inds)
    qx, qy = split_flux(q, inds)

    # X fluxes

    for j in 2:(ny - 1), i in 2:nx
        qx[i, j] = ψ[i - 1, j] - ψ[i - 1, j - 1]  # Interior
    end
    for j in 1, i in 2:nx
        qx[i, j] = ψ[i - 1, j]  # Top boundary
    end
    for j in ny, i in 2:nx
        qx[i, j] = -ψ[i - 1, j - 1]  # Bottom boundary
    end

    # Y fluxes

    for j in 2:ny, i in 2:(nx - 1)
        qy[i, j] = ψ[i - 1, j - 1] - ψ[i, j - 1]  # Interior
    end
    for j in 2:ny, i in 1
        qy[i, j] = -ψ[i, j - 1]  # Left boundary
    end
    for j in 2:ny, i in nx
        qy[i, j] = ψ[i - 1, j - 1]  # Right boundary
    end

    q
end

function rot!(Γ_flat, q, inds::GridIndices)
    (; nx, ny) = inds
    Γ = unflatten_circ(Γ_flat, inds)
    qx, qy = split_flux(q, inds)

    for j in 2:ny, i in 2:nx
        Γ[i - 1, j - 1] = (qx[i, j - 1] - qx[i, j]) + (qy[i, j] - qy[i - 1, j])
    end

    Γ_flat
end

function _dst_plan(b::AbstractMatrix{Float64}; flags, num_threads)
    FFTW.plan_r2r(b, FFTW.RODFT00, (1, 2); flags, num_threads)
end

function Δinv_operator(inds, dst_plan, lap_eigs; kw...)
    _lap_inv_operator(inds, dst_plan; Λ=lap_eigs, kw...)
end

function A_operator(prob::Problem, dst_plan, lap_eigs::AbstractMatrix; level, kw...)
    fluid = prob.fluid
    (; grid, Re) = fluid
    hc = gridstep(grid, level)
    dt = timestep(prob)

    Λexpl = @. inv(1 - lap_eigs * dt / (2 * Re * hc^2))  # Explicit eigenvalues
    A = _lap_inv_operator(grid.inds, dst_plan; Λ=Λexpl, kw...)

    Λimpl = @. 1 + lap_eigs * dt / (2 * Re * hc^2)  # Implicit eigenvalues
    Ainv = _lap_inv_operator(grid.inds, dst_plan; Λ=Λimpl, kw...)

    (A, Ainv)
end

function A_operators(prob::Problem, dst_plan, lap_eigs; kw...)
    fluid = prob.fluid
    nlevel = fluid.grid.nlevel
    ops = [A_operator(prob, dst_plan, lap_eigs; level, kw...) for level in 1:nlevel]
    A = [A for (A, _) in ops]
    Ainv = [Ainv for (_, Ainv) in ops]
    (A, Ainv)
end

function _lap_inv_operator(
    inds::GridIndices,
    dst_plan::FFTW.Plan;
    Λ::AbstractMatrix,
    Γtmp1::AbstractMatrix,
    Γtmp2::AbstractMatrix,
)
    (; nx, ny) = inds

    # Include scale to make forward and inverse transforms equal
    scale = 1 / (4 * nx * ny)

    function (x, b)
        # Use temporary arrays to avoid alignment issues with the FFT plan
        Γtmp1[:] = b
        mul!(Γtmp2, dst_plan, Γtmp1)
        @. Γtmp2 *= scale / Λ
        mul!(Γtmp1, dst_plan, Γtmp2)
        x[:] = Γtmp1
        x
    end
end

function _lap_eigs(inds::GridIndices)
    (; nx, ny) = inds
    i = 1:(nx - 1)
    j = permutedims(1:(ny - 1))
    @. -2 * (cos(π * i / nx) + cos(π * j / ny) - 2)
end

function vort2flux(
    grid::MultiDomainGrid; Δinv!, ψ_bc::AbstractVector, Γ_tmp::AbstractVector
)
    function (ψ::AbstractMatrix, q::AbstractMatrix, Γ::AbstractMatrix)
        nlevel = grid.nlevel

        # Interpolate values from finer grid to center region of coarse grids
        for lev in 2:nlevel
            @views coarsify!(Γ[:, lev], Γ[:, lev - 1], grid.inds)
        end

        # Invert Laplacian on largest grid with zero boundary conditions
        ψ .= 0
        ψ_bc .= 0
        @views Δinv!(ψ[:, nlevel], Γ[:, nlevel])  # Δψ = Γ
        @views curl!(q[:, nlevel], ψ[:, nlevel], ψ_bc, grid.inds)  # q = ∇×ψ

        # Telescope in to finer grids, using boundary conditions from coarser
        for lev in (nlevel - 1):-1:1
            @views Γ_tmp .= Γ[:, lev]
            @views get_bc!(ψ_bc, ψ[:, lev + 1], grid.inds)
            apply_bc!(Γ_tmp, ψ_bc, grid.inds; fac=1.0)
            @views Δinv!(ψ[:, lev], Γ_tmp)  # Δψ = Γ

            if lev < nlevel
                @views curl!(q[:, lev], ψ[:, lev], ψ_bc, grid.inds)  # q = ∇×ψ
            end
        end

        nothing
    end
end

function coarsify!(Γc_flat::AbstractVector, Γ_flat::AbstractVector, inds::GridIndices)
    (; nx, ny) = inds
    Γc = unflatten_circ(Γc_flat, inds)
    Γ = unflatten_circ(Γ_flat, inds)

    # Indices
    is = nx ÷ 2 .+ ((-nx ÷ 2 + 2):2:(nx ÷ 2 - 2))
    js = ny ÷ 2 .+ ((-ny ÷ 2 + 2):2:(ny ÷ 2 - 2))
    # Coarse indices
    ics = nx ÷ 2 .+ ((-nx ÷ 4 + 1):(nx ÷ 4 - 1))
    jcs = ny ÷ 2 .+ ((-ny ÷ 4 + 1):(ny ÷ 4 - 1))

    for (j, jc) in zip(js, jcs), (i, ic) in zip(is, ics)
        Γc[ic, jc] =
            Γ[i, j] +
            0.5 * (Γ[i + 1, j] + Γ[i, j + 1] + Γ[i - 1, j] + Γ[i, j - 1]) +
            0.25 * (Γ[i + 1, j + 1] + Γ[i + 1, j - 1] + Γ[i - 1, j - 1] + Γ[i - 1, j + 1])
    end

    Γc_flat
end

function curl!(
    q::AbstractVector, ψ_flat::AbstractVector, ψ_bc::AbstractVector, inds::GridIndices
)
    (; nx, ny, T, B, L, R) = inds

    ψ = unflatten_circ(ψ_flat, inds)
    qx, qy = split_flux(q, inds)

    # X fluxes

    for j in 2:(ny - 1), i in 2:nx
        qx[i, j] = ψ[i - 1, j] - ψ[i - 1, j - 1]  # Interior
    end
    for j in 1, i in 2:nx
        qx[i, j] = ψ[i - 1, j] - ψ_bc[B + i]  # Bottom boundary
    end
    for j in ny, i in 2:nx
        qx[i, j] = ψ_bc[i + T] - ψ[i - 1, j - 1]  # Top boundary
    end

    for j in 1:ny, i in 1
        qx[i, j] = ψ_bc[(L + 1) + j] - ψ_bc[L + j]  # Left boundary
    end
    for j in 1:ny, i in nx + 1
        qx[i, j] = ψ_bc[(R + 1) + j] - ψ_bc[R + j]  # Right boundary
    end

    # Y fluxes

    for j in 2:ny, i in 2:(nx - 1)
        qy[i, j] = ψ[i - 1, j - 1] - ψ[i, j - 1]  # Interior
    end
    for j in 2:ny, i in 1
        qy[i, j] = ψ_bc[L + j] - ψ[i, j - 1]  # Left boundary
    end
    for j in 2:ny, i in nx
        qy[i, j] = ψ[i - 1, j - 1] - ψ_bc[R + j]  # Right boundary
    end
    for j in 1, i in 1:nx
        qy[i, j] = ψ_bc[B + i] - ψ_bc[(B + 1) + i]  # Bottom boundary
    end
    for j in ny + 1, i in 1:nx
        qy[i, j] = ψ_bc[T + i] - ψ_bc[(T + 1) + i]  # Top boundary
    end

    q
end

function get_bc!(rbc::AbstractVector, r_flat::AbstractVector, inds::GridIndices)
    # Given vorticity on a larger, coarser mesh, interpolate it's values to the edge of a
    # smaller, finer mesh.

    (; nx, ny, T, B, L, R) = inds
    r = unflatten_circ(r_flat, inds)

    let i = (nx ÷ 4) .+ (0:(nx ÷ 2)), ibc = 1:2:(nx + 1)
        @views @. rbc[B + ibc] = r[i, ny ÷ 4]
        @views @. rbc[T + ibc] = r[i, 3 * ny ÷ 4]
    end

    let i = (nx ÷ 4) .+ (1:(nx ÷ 2)), ibc = 2:2:nx
        @views @. rbc[B + ibc] = 0.5 * (r[i, ny ÷ 4] + r[i - 1, ny ÷ 4])
        @views @. rbc[T + ibc] = 0.5 * (r[i, 3 * ny ÷ 4] + r[i - 1, 3 * ny ÷ 4])
    end

    let j = (ny ÷ 4) .+ (0:(ny ÷ 2)), jbc = 1:2:(ny + 1)
        @views @. rbc[L + jbc] = r[nx ÷ 4, j]
        @views @. rbc[R + jbc] = r[3 * nx ÷ 4, j]
    end

    let j = (ny ÷ 4) .+ (1:(ny ÷ 2)), jbc = 2:2:ny
        @views @. rbc[L + jbc] = 0.5 * (r[nx ÷ 4, j] + r[nx ÷ 4, j - 1])
        @views @. rbc[R + jbc] = 0.5 * (r[3 * nx ÷ 4, j] + r[3 * nx ÷ 4, j - 1])
    end

    rbc
end

function apply_bc!(
    r_flat::AbstractVector, rbc::AbstractVector, inds::GridIndices; fac::Real
)
    # Given vorticity at edges of domain, rbc, (from larger, coarser mesh), add values to correct
    # laplacian of vorticity  on the (smaller, finer) domain, r.
    # r is a vorticity-like array of size (nx-1)×(ny-1)

    (; nx, ny, T, B, L, R) = inds
    r = unflatten_circ(r_flat, inds)

    # add bc's from coarser grid
    for i in 1:(nx - 1)
        let j = 1
            r[i, j] += fac * rbc[(B + 1) + i]
        end
        let j = ny - 1
            r[i, j] += fac * rbc[(T + 1) + i]
        end
    end

    for j in 1:(ny - 1)
        let i = 1
            r[i, j] += fac * rbc[(L + 1) + j]
        end
        let i = nx - 1
            r[i, j] += fac * rbc[(R + 1) + j]
        end
    end

    r_flat
end

function rhs_force(grid::MultiDomainGrid; q_tmp)
    function (fq::AbstractVector, state::State, Γbc::AbstractVector, lev::Int)
        Γ = unflatten_circ(state.Γ, grid.inds, lev)
        avg_flux!(q_tmp, state, grid.inds, lev)  # Compute average fluxes across cells

        # Call helper function to loop over the arrays and store product in fq
        direct_product!(fq, q_tmp, Γ, Γbc, grid.inds)

        fq
    end
end

function avg_flux!(Q::AbstractVector, state::State, inds::GridIndices, lev::Int)
    (; nx, ny) = inds

    qx, qy = split_flux(state.q, inds, lev)
    q0x, q0y = split_flux(state.q0, inds, lev)
    Qx, Qy = split_flux(Q, inds)

    Q .= 0  # Zero out unset elements

    for j in 2:ny, i in 1:(nx + 1)
        Qx[i, j] = (qx[i, j] + qx[i, j - 1] + q0x[i, j] + q0x[i, j - 1]) / 2
    end

    for j in 1:(ny + 1), i in 2:nx
        Qy[i, j] = (qy[i, j] + qy[i - 1, j] + q0y[i, j] + q0y[i - 1, j]) / 2
    end

    Q
end

function direct_product!(fq, Q, Γ, Γbc, inds::GridIndices)
    # Gather the product used in computing advection term

    # fq is the output array: the product of flux and circulation such that the nonlinear term is
    # C'*fq (or ∇⋅fq)

    fq .= 0  # Zero out in case some locations aren't indexed
    direct_product_loops!(fq, Q, Γ, Γbc, inds)
end

function direct_product_loops!(
    fq::AbstractVector,
    Q::AbstractVector,
    Γ::AbstractMatrix,
    Γbc::AbstractVector,
    inds::GridIndices,
)
    # Helper function to compute the product of Q and Γ so that the advective term is ∇⋅fq

    (; nx, ny, T, B, L, R) = inds

    Qx, Qy = split_flux(Q, inds)
    fqx, fqy = split_flux(fq, inds)

    # x fluxes
    for j in 2:(ny - 1), i in 2:nx
        let Qy1 = Qy[i, j + 1], Γ1 = Γ[i - 1, j], Qy2 = Qy[i, j], Γ2 = Γ[i - 1, j - 1]
            fqx[i, j] = (Qy1 * Γ1 + Qy2 * Γ2) / 2
        end
    end

    # x fluxes bottom boundary
    for j in 1, i in 2:nx
        let Qy1 = Qy[i, j + 1], Γ1 = Γ[i - 1, j], Qy2 = Qy[i, j], Γ2 = Γbc[B + i]
            fqx[i, j] = (Qy1 * Γ1 + Qy2 * Γ2) / 2
        end
    end

    # x fluxes top boundary
    for j in ny, i in 2:nx
        let Qy1 = Qy[i, j], Γ1 = Γ[i - 1, j - 1], Qy2 = Qy[i, j + 1], Γ2 = Γbc[T + i]
            fqx[i, j] = (Qy1 * Γ1 + Qy2 * Γ2) / 2
        end
    end

    # y fluxes
    for j in 2:ny, i in 2:(nx - 1)
        let Qx1 = Qx[i + 1, j], Γ1 = Γ[i, j - 1], Qx2 = Qx[i, j], Γ2 = Γ[i - 1, j - 1]
            fqy[i, j] = -(Qx1 * Γ1 + Qx2 * Γ2) / 2
        end
    end

    # y fluxes left boundary
    for j in 2:ny, i in 1
        let Qx1 = Qx[i + 1, j], Γ1 = Γ[i, j - 1], Qx2 = Qx[i, j], Γ2 = Γbc[L + j]
            fqy[i, j] = -(Qx1 * Γ1 + Qx2 * Γ2) / 2
        end
    end

    # y fluxes right boundary
    for j in 2:ny, i in nx
        let Qx1 = Qx[i, j], Γ1 = Γ[i - 1, j - 1], Qx2 = Qx[i + 1, j], Γ2 = Γbc[R + j]
            fqy[i, j] = -(Qx1 * Γ1 + Qx2 * Γ2) / 2
        end
    end

    fq
end

function nonlinear(grid::MultiDomainGrid; rhs_force!, fq::AbstractVector)
    function (nonlin::AbstractVector, state::State, Γbc::AbstractVector, lev::Int)
        # Get flux-circulation product
        rhs_force!(fq, state, Γbc, lev)

        # Divergence of flux-circulation product
        rot!(nonlin, fq, grid.inds)

        # Scaling: 1/hc^2 to convert circulation to vorticity
        hc = gridstep(grid, lev)  # Coarse grid spacing
        nonlin .*= 1 / hc^2

        nonlin
    end
end

function get_trial_state(
    prob::Problem{CNAB}; nonlinear!, vort2flux!, A, Ainv, bc, rhs, rhsbc
)
    fluid = prob.fluid
    grid = fluid.grid
    nlevel = grid.nlevel
    scheme = prob.scheme
    dt = timestep(scheme)
    Re = fluid.Re

    function (qs::AbstractMatrix, Γs::AbstractMatrix, state::State)
        for lev in nlevel:-1:1
            bc .= 0
            rhsbc .= 0
            hc = gridstep(grid, lev)

            if lev < nlevel
                @views get_bc!(bc, state.Γ[:, lev + 1], grid.inds)
                apply_bc!(rhsbc, bc, grid.inds; fac=0.25 * dt / (Re * hc^2))
            end

            # Account for scaling between grids
            # Don't need bc's for anything after this, so we can rescale in place
            bc .*= 0.25

            #compute the nonlinear term for the current time step
            @views nonlinear!(state.nonlin[1][:, lev], state, bc, lev)

            @views A[lev](rhs, state.Γ[:, lev])

            for n in 1:length(scheme.β)
                βn = scheme.β[n]
                @views @. rhs += dt * βn * state.nonlin[n][:, lev]
            end

            # Include boundary conditions
            rhs .+= rhsbc

            # Trial circulation  Γs = Ainvs * rhs
            @views Ainv[lev](Γs[:, lev], rhs)
        end

        # Store nonlinear solution for use in next time step
        # Cycle nonlinear arrays
        _cycle!(state.nonlin)

        vort2flux!(state.ψ, qs, Γs)

        nothing
    end
end

# Rotate elements in `a` forward an index
_cycle!(a::Vector) = isempty(a) ? a : pushfirst!(a, pop!(a))

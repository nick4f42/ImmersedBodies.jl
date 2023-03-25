function base_flux!(
    qty::PsiOmegaGridQuantities, fluid::PsiOmegaFluidGrid{<:Any,GlobalFrame}, t::Float64
)
    # Set the base flux at time t for a fluid discretized in the global frame

    grids = discretized(fluid)
    u, v = qty.u
    qx, qy = split_flux(qty.q0, fluid.gridindex)

    for lev in 1:nlevels(grids)
        # Coarse grid spacing
        hc = gridstep(grids, lev)

        qx[:, :, lev] .= u * hc
        qy[:, :, lev] .= v * hc
    end
end

function base_flux!(
    qty::PsiOmegaGridQuantities,
    fluid::PsiOmegaFluidGrid{<:Any,OffsetFrame{GlobalFrame}},
    t::Float64,
)
    # Set the base flux at time t for a fluid discretized in a moving frame

    grids = discretized(fluid)
    qx, qy = split_flux(qty.q0, fluid.gridindex)

    f = fluid.frame(t) # evaluate frame at time t
    u0 = f.v # frame velocity
    Ω = f.Ω
    c = f.cθ # cos(θ)
    s = f.sθ # sin(θ)
    Rx = @SMatrix [c -s; s c] # basis of relative frame in global frame
    Rv = Ω * @SMatrix [-s -c; c -s] # affect of rotation on velocity

    for lev in 1:nlevels(grids)
        grid = sublevel(grids, lev)
        hc = gridstep(grid)

        # Transform from freestream velocity (ux, uy) in global frame to relative frame
        let (xs, ys) = xflux_ranges(grid)
            for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
                qx[i, j, lev] = hc * dot(Rx[:, 1], qty.u - u0 - Rv * SVector(x, y))
            end
        end
        let (xs, ys) = yflux_ranges(grid)
            for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
                qy[i, j, lev] = hc * dot(Rx[:, 2], qty.u - u0 - Rv * SVector(x, y))
            end
        end
    end
end

function C_linearmap(gridindex::PsiOmegaGridIndexing)
    C(y, x) = curl!(y, x, gridindex)
    CT(y, x) = rot!(y, x, gridindex)
    return LinearMap(C, CT, gridindex.nq, gridindex.nΓ)
end

function curl!(q_flat, ψ_flat, gridindex::PsiOmegaGridIndexing)
    (; nx, ny) = gridindex

    ψ = reshape(ψ_flat, nx - 1, ny - 1)
    qx, qy = split_flux(q_flat, gridindex)

    # X fluxes

    let i = 2:nx, j = 2:(ny - 1)
        @views @. qx[i, j] = ψ[i - 1, j] - ψ[i - 1, j - 1] # Interior
    end
    let i = 2:nx, j = 1
        @views @. qx[i, j] = ψ[i .- 1, j] # Top boundary
    end
    let i = 2:nx, j = ny
        @views @. qx[i, j] = -ψ[i .- 1, j .- 1] # Bottom boundary
    end

    # Y fluxes

    let i = 2:(nx - 1), j = 2:ny
        @views @. qy[i, j] = ψ[i - 1, j - 1] - ψ[i, j - 1] # Interior
    end
    let i = 1, j = 2:ny
        @views @. qy[i, j] = -ψ[i, j - 1] # Left boundary
    end
    let i = nx, j = 2:ny
        @views @. qy[i, j] = ψ[i - 1, j - 1] # Right boundary
    end

    return nothing
end

function rot!(Γ_flat, q_flat, gridindex::PsiOmegaGridIndexing)
    (; nx, ny) = gridindex

    Γ = reshape(Γ_flat, nx - 1, ny - 1)
    qx, qy = split_flux(q_flat, gridindex)

    i = 2:nx
    j = 2:ny

    @views @. Γ[i - 1, j - 1] = (qx[i, j - 1] - qx[i, j]) + (qy[i, j] - qy[i - 1, j])

    return nothing
end

function curl!(q_flat, ψ_flat, ψ_bc, gridindex::PsiOmegaGridIndexing)
    (; nx, ny, T, B, L, R) = gridindex

    ψ = unflatten_circ(ψ_flat, gridindex)
    qx, qy = split_flux(q_flat, gridindex)

    # X fluxes

    let i = 2:nx, j = 2:(ny - 1)
        @views @. qx[i, j] = ψ[i - 1, j] - ψ[i - 1, j - 1] # Interior
    end
    let i = 2:nx, j = 1
        @views @. qx[i, j] = ψ[i - 1, j] - ψ_bc[B + i] # Bottom boundary
    end
    let i = 2:nx, j = ny
        @views @. qx[i, j] = ψ_bc[i + T] - ψ[i - 1, j - 1] # Top boundary
    end

    let i = 1, j = 1:ny
        @views @. qx[i, j] = ψ_bc[(L + 1) + j] - ψ_bc[L + j] # Left boundary
    end
    let i = nx + 1, j = 1:ny
        @views @. qx[i, j] = ψ_bc[(R + 1) + j] - ψ_bc[R + j] # Right boundary
    end

    # Y fluxes

    let i = 2:(nx - 1), j = 2:ny
        @views @. qy[i, j] = ψ[i - 1, j - 1] - ψ[i, j - 1] # Interior
    end
    let i = 1, j = 2:ny
        @views @. qy[i, j] = ψ_bc[L + j] - ψ[i, j - 1] # Left boundary
    end
    let i = nx, j = 2:ny
        @views @. qy[i, j] = ψ[i - 1, j - 1] - ψ_bc[R + j] # Right boundary
    end
    let i = 1:nx, j = 1
        @views @. qy[i, j] = ψ_bc[B + i] - ψ_bc[(B + 1) + i] # Bottom boundary
    end
    let i = 1:nx, j = ny + 1
        @views @. qy[i, j] = ψ_bc[T + i] - ψ_bc[(T + 1) + i] # Top boundary
    end

    return nothing
end

function coarsify!(
    Γc_flat::AbstractVector, Γ_flat::AbstractVector, gridindex::PsiOmegaGridIndexing
)
    (; nx, ny) = gridindex
    Γc = unflatten_circ(Γc_flat, gridindex)
    Γ = unflatten_circ(Γ_flat, gridindex)

    # Indices
    is = @. nx ÷ 2 .+ ((-nx ÷ 2 + 2):2:(nx ÷ 2 - 2))
    js = @. ny ÷ 2 .+ ((-ny ÷ 2 + 2):2:(ny ÷ 2 - 2))
    # Coarse indices
    ics = @. nx ÷ 2 .+ ((-nx ÷ 4 + 1):(nx ÷ 4 - 1))
    jcs = @. ny ÷ 2 .+ ((-ny ÷ 4 + 1):(ny ÷ 4 - 1))

    for (j, jc) in zip(js, jcs), (i, ic) in zip(is, ics)
        Γc[ic, jc] =
            Γ[i, j] +
            0.5 * (Γ[i + 1, j] + Γ[i, j + 1] + Γ[i - 1, j] + Γ[i, j - 1]) +
            0.25 * (Γ[i + 1, j + 1] + Γ[i + 1, j - 1] + Γ[i - 1, j - 1] + Γ[i - 1, j + 1])
    end

    return nothing
end

function get_bc!(
    rbc::AbstractVector, r_flat::AbstractVector, gridindex::PsiOmegaGridIndexing
)
    # Given vorticity on a larger, coarser mesh, interpolate it's values to the edge of a
    # smaller, finer mesh.

    (; nx, ny, T, B, L, R) = gridindex
    r = unflatten_circ(r_flat, gridindex)

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

    return nothing
end

function apply_bc!(
    r_flat::AbstractVector,
    rbc::AbstractVector,
    fac::Float64,
    gridindex::PsiOmegaGridIndexing,
)
    # Given vorticity at edges of domain, rbc, (from larger, coarser mesh), add values to correct
    # laplacian of vorticity  on the (smaller, finer) domain, r.
    # r is a vorticity-like array of size (nx-1)×(ny-1)

    (; nx, ny, T, B, L, R) = gridindex
    r = unflatten_circ(r_flat, gridindex)

    # add bc's from coarser grid
    @views let i = 1:(nx - 1)
        let j = 1
            @. r[i, j] += fac * rbc[(B + 1) + i]
        end
        let j = ny - 1
            @. r[i, j] += fac * rbc[(T + 1) + i]
        end
    end

    @views let j = 1:(ny - 1)
        let i = 1
            @. r[i, j] += fac * rbc[(L + 1) + j]
        end
        let i = nx - 1
            @. r[i, j] += fac * rbc[(R + 1) + j]
        end
    end

    return nothing
end

function avg_flux!(
    Q::AbstractVector,
    qty::PsiOmegaGridQuantities,
    gridindex::PsiOmegaGridIndexing,
    lev::Int,
)
    (; nx, ny) = gridindex

    qx, qy = split_flux(qty.q, gridindex, lev)
    q0x, q0y = split_flux(qty.q0, gridindex, lev)
    Qx, Qy = split_flux(Q, gridindex)

    Q .= 0 # Zero out unset elements

    # Index into Qx from (1:nx+1)×(2:ny)
    let i = 1:(nx + 1), j = 2:ny
        @views @. Qx[i, j] = (qx[i, j] + qx[i, j .- 1] + q0x[i, j] + q0x[i, j .- 1]) / 2
    end

    # Index into Qy from (2:nx)×(1:ny+1)
    let i = 2:nx, j = 1:(ny + 1)
        @views @. Qy[i, j] = (qy[i, j] + qy[i .- 1, j] + q0y[i, j] + q0y[i .- 1, j]) / 2
    end

    return Q
end

function direct_product!(fq, Q, Γ, Γbc, gridindex::PsiOmegaGridIndexing)
    # Gather the product used in computing advection term

    # fq is the output array: the product of flux and circulation such that the nonlinear term is
    # C'*fq (or ∇⋅fq)

    fq .= 0 # Zero out in case some locations aren't indexed

    direct_product_loops!(fq, Q, Γ, Γbc, gridindex)

    return nothing
end

function direct_product_loops!(fq, Q, Γ, Γbc, gridindex::PsiOmegaGridIndexing)
    # Helper function to compute the product of Q and Γ so that the advective term is ∇⋅fq
    (; nx, ny, T, B, L, R) = gridindex

    Qx, Qy = split_flux(Q, gridindex)
    fqx, fqy = split_flux(fq, gridindex)

    # x fluxes
    @views let i = 2:nx,
        j = 2:(ny - 1),
        fqx = fqx[i, j],
        Qy1 = Qy[i, j .+ 1],
        Γ1 = Γ[i .- 1, j],
        Qy2 = Qy[i, j],
        Γ2 = Γ[i .- 1, j .- 1]

        @. fqx = (Qy1 * Γ1 + Qy2 * Γ2) / 2
    end

    # x fluxes bottom boundary
    @views let i = 2:nx,
        j = 1,
        fqx = fqx[i, j],
        Qy1 = Qy[i, j .+ 1],
        Γ1 = Γ[i .- 1, j],
        Qy2 = Qy[i, j],
        Γ2 = Γbc[B .+ i]

        @. fqx = (Qy1 * Γ1 + Qy2 * Γ2) / 2
    end

    # x fluxes top boundary
    @views let i = 2:nx,
        j = ny,
        fqx = fqx[i, j],
        Qy1 = Qy[i, j],
        Γ1 = Γ[i .- 1, j .- 1],
        Qy2 = Qy[i, j .+ 1],
        Γ2 = Γbc[T .+ i]

        @. fqx = (Qy1 * Γ1 + Qy2 * Γ2) / 2
    end

    # y fluxes
    @views let i = 2:(nx - 1),
        j = 2:ny,
        fqy = fqy[i, j],
        Qx1 = Qx[i .+ 1, j],
        Γ1 = Γ[i, j .- 1],
        Qx2 = Qx[i, j],
        Γ2 = Γ[i .- 1, j .- 1]

        @. fqy = -(Qx1 * Γ1 + Qx2 * Γ2) / 2
    end

    # y fluxes left boundary
    @views let i = 1,
        j = 2:ny,
        fqy = fqy[i, j],
        Qx1 = Qx[i .+ 1, j],
        Γ1 = Γ[i, j .- 1],
        Qx2 = Qx[i, j],
        Γ2 = Γbc[L .+ j]

        @. fqy = -(Qx1 * Γ1 + Qx2 * Γ2) / 2
    end

    # y fluxes right boundary
    @views let i = nx,
        j = 2:ny,
        fqy = fqy[i, j],
        Qx1 = Qx[i, j],
        Γ1 = Γ[i .- 1, j .- 1],
        Qx2 = Qx[i .+ 1, j],
        Γ2 = Γbc[R .+ j]

        @. fqy = -(Qx1 * Γ1 + Qx2 * Γ2) / 2
    end

    return nothing
end

Base.@kwdef struct Vort2Flux{F<:PsiOmegaFluidGrid,M}
    fluid::F
    ψbc::Vector{Float64}
    Γtmp::Vector{Float64}
    Δinv::M
end

function (v2f::Vort2Flux)(ψ::AbstractMatrix, q::AbstractMatrix, Γ::AbstractMatrix)
    # Multiscale method to solve C^T C s = omega and return the velocity, C s.  Results are
    # returned in vel on each of the first nlev grids.

    # Warning: the vorticity field on all but the finest mesh is modified by the routine in the
    # following way: the value in the center of the domain is interpolated from the next finer
    # mesh (the value near the edge is not changed.

    fluid = v2f.fluid
    grid = discretized(fluid)
    nlevel = nlevels(grid)

    (; ψbc, Γtmp, Δinv) = v2f
    (; nx, ny) = fluid.gridindex

    # Interpolate values from finer grid to center region of coarse grids
    for lev in 2:nlevel
        @views coarsify!(Γ[:, lev], Γ[:, lev - 1], fluid.gridindex)
    end

    # Invert Laplacian on largest grid with zero boundary conditions
    ψ .= 0
    ψbc .= 0
    @views mul!(ψ[:, nlevel], Δinv, Γ[:, nlevel]) # Δψ = Γ
    @views curl!(q[:, nlevel], ψ[:, nlevel], ψbc, fluid.gridindex) # q = ∇×ψ

    # Telescope in to finer grids, using boundary conditions from coarser
    for lev in (nlevel - 1):-1:1
        @views Γtmp .= Γ[:, lev]
        @views get_bc!(ψbc, ψ[:, lev + 1], fluid.gridindex)
        apply_bc!(Γtmp, ψbc, 1.0, fluid.gridindex)

        @views mul!(ψ[:, lev], Δinv, Γtmp) # Δψ = Γ
        if lev < nlevel
            @views curl!(q[:, lev], ψ[:, lev], ψbc, fluid.gridindex) # q = ∇×ψ
        end
    end

    return nothing
end

(v2f::Vort2Flux)(qty::PsiOmegaGridQuantities) = v2f(qty.ψ, qty.q, qty.Γ)

Base.@kwdef struct RhsForce
    gridindex::PsiOmegaGridIndexing
    Q::Vector{Float64}
end

function (rhsf::RhsForce)(
    fq::AbstractVector, qty::PsiOmegaGridQuantities, Γbc::AbstractVector, lev::Int
)
    gridindex = rhsf.gridindex
    (; nx, ny) = gridindex
    Q = rhsf.Q

    Γ = unflatten_circ(qty.Γ, gridindex, lev) # Circulation at this grid level
    avg_flux!(Q, qty, gridindex, lev) # Compute average fluxes across cells

    fq .= 0 # Zero the unset indices

    # Call helper function to loop over the arrays and store product in fq
    direct_product!(fq, Q, Γ, Γbc, gridindex)

    return nothing
end

struct LaplacianInv{P<:FFTW.r2rFFTWPlan}
    gridindex::PsiOmegaGridIndexing
    b_temp::Matrix{Float64}
    x_temp::Matrix{Float64}
    Λ::Matrix{Float64}
    dst_plan::P
    work::Matrix{Float64}
    scale::Float64
end

function lap_inv_linearmap(lap_inv::LaplacianInv)
    nΓ = length(lap_inv.Λ)
    return LinearMap(lap_inv, nΓ; issymmetric=true)
end

lap_inv_linearmap(args...) = lap_inv_linearmap(LaplacianInv(args...))

function LaplacianInv(gridindex::PsiOmegaGridIndexing)
    Λ = lap_eigs(gridindex)
    b = ones(gridindex.nx - 1, gridindex.ny - 1)
    dst_plan = FFTW.plan_r2r(b, FFTW.RODFT00, [1, 2]; flags=FFTW.EXHAUSTIVE)

    return LaplacianInv(gridindex, dst_plan, Λ)
end

function LaplacianInv(gridindex::PsiOmegaGridIndexing, dst_plan, Λ)
    (; nx, ny) = gridindex

    # TODO: Test without x_temp b_temp intermediaries
    b_temp = zeros(nx - 1, ny - 1) # Input
    x_temp = zeros(nx - 1, ny - 1) # Output
    work = zeros(nx - 1, ny - 1)
    scale = 1 / (4 * nx * ny)

    return LaplacianInv(gridindex, b_temp, x_temp, Λ, dst_plan, work, scale)
end

function (lapinv::LaplacianInv)(x::AbstractVector, b::AbstractVector)
    (; gridindex, b_temp, x_temp, Λ, dst_plan, work, scale) = lapinv
    (; nx, ny) = gridindex

    # TODO: Test if b_temp and x_temp are beneficial
    b_temp .= reshape(b, size(b_temp))

    mul!(work, dst_plan, b_temp)
    @. work *= scale / Λ
    mul!(x_temp, dst_plan, work)

    x .= reshape(x_temp, size(x))

    return nothing
end

function lap_eigs(gridindex::PsiOmegaGridIndexing)
    (; nx, ny) = gridindex
    return @. -2 * (cos(π * (1:(nx - 1)) / nx) + cos(π * (1:(ny - 1)) / ny)' - 2)
end

function A_Ainv_linearmaps(
    prob::Problem{<:PsiOmegaFluidGrid{CNAB}}, lap_inv::LaplacianInv, lev::Int
)
    # Compute the matrix (as a LinearMap) that represents the modified Poisson operator (I + dt/2
    # * Beta * RC) arising from the implicit treatment of the Laplacian. A system involving this
    # matrix is solved to compute a trial circulation that doesn't satisfy the BCs, and then again
    # to use the surface stresses to update the trial circulation so that it satisfies the BCs

    # Construct LinearMap to solve
    #     (I + dt/2 * Beta * RC) * x = b for x
    # where Beta = 1/(Re * h^2)

    # Solve by transforming to and from Fourier space and scaling by evals.

    grid = discretized(prob.fluid)
    gridindex = prob.fluid.gridindex

    hc = gridstep(grid, lev)
    dt = timestep(prob)
    Re = conditions(prob.fluid).Re

    Λ = lap_inv.Λ
    dst_plan = lap_inv.dst_plan

    Λexpl = @. inv(1 - Λ * dt / (2 * Re * hc^2)) # Explicit eigenvalues
    A = lap_inv_linearmap(gridindex, dst_plan, Λexpl)

    Λimpl = @. 1 + Λ * dt / (2 * Re * hc^2) # Implicit eigenvalues
    Ainv = lap_inv_linearmap(gridindex, dst_plan, Λimpl)

    return (A, Ainv)
end

function A_Ainv_linearmaps(prob::Problem{<:PsiOmegaFluidGrid{CNAB}}, lap_inv::LaplacianInv)
    nlevel = (nlevels ∘ discretized)(prob.fluid)

    x = map(lev -> A_Ainv_linearmaps(prob, lap_inv, lev), 1:nlevel)
    As = [A for (A, _) in x]
    Ainvs = [Ainv for (_, Ainv) in x]
    return (As, Ainvs)
end

Base.@kwdef mutable struct B_Times{V<:Vort2Flux,ME,MC,MAinv}
    vort2flux::V
    Ainv::MAinv
    C::MC
    E::ME
    Γtmp::Vector{Float64}
    qtmp::Vector{Float64}
    Γ::Matrix{Float64}
    ψ::Matrix{Float64}
    q::Matrix{Float64}
end

function B_Times(fluid::PsiOmegaFluidGrid, vort2flux::Vort2Flux, Γtmp, qtmp; Ainv, C, E)
    nlevel = (nlevels ∘ discretized)(fluid)
    (; nΓ, nq) = fluid.gridindex

    # TODO: Move allocations to solver
    Γ = zeros(nΓ, nlevel) # Working array for circulation
    ψ = zeros(nΓ, nlevel) # Working array for streamfunction
    q = zeros(nq, nlevel) # Working array for velocity flux

    return B_Times(vort2flux, Ainv, C, E, Γtmp, qtmp, Γ, ψ, q)
end

function B_linearmap(bodies::BodyGroup{<:RigidBody}, B_times::B_Times)
    nftot = 2 * npanels(bodies)
    return LinearMap(B_times, nftot; issymmetric=true)
end

function B_linearmap(bodies::BodyGroup, B_times::B_Times)
    nftot = 2 * npanels(bodies)
    return LinearMap(B_times, nftot; issymmetric=false)
end

function Binv_linearmap(prob::StaticBodyProblem{CNAB}, B)
    # Precompute 'Binv' matrix by evaluating mat-vec products for unit vectors. This is a big
    # speedup when the interpolation operator E isn't going to change (no FSI, for instance)
    nftot = 2 * npanels(prob.bodies)

    # Pre-allocate arrays
    Bmat = zeros(nftot, nftot)
    e = zeros(nftot) # Unit vector

    for j in 1:nftot
        # Construct unit vector
        if j > 1
            e[j - 1] = 0
        end
        e[j] = 1

        @views mul!(Bmat[:, j], B, e)
    end

    Binv = inv(Bmat)
    return LinearMap(Binv; issymmetric=true)

    # TODO: Diagnose why cholesky decomposition leads to non-hermitian error
    # B_decomp = cholesky!(Bmat)
    # return LinearMap((y, x) -> ldiv!(y, B_decomp, x), nftot; issymmetric=true)
end

function Binv_linearmap(prob::Problem{<:PsiOmegaFluidGrid,<:RigidBody}, B)
    nftot = 2 * npanels(prob.bodies)

    # Solves f = B*g for g... so g = Binv * f
    # TODO: Add external interface for cg! options
    Binv = LinearMap(nftot; issymmetric=true) do f, g
        cg!(f, B, g; maxiter=5000, reltol=1e-12)
    end

    return Binv
end

function Binv_linearmap(prob::Problem{<:PsiOmegaFluidGrid}, B, Q_Itilde_W)
    nftot = 2 * npanels(prob.bodies)

    # Solves f = B*g for g... so g = Binv * f
    # TODO: Add external interface for bicgstabl! options
    Binv = LinearMap(nftot; issymmetric=false) do f, g
        bicgstabl!(f, B + Q_Itilde_W, g)
    end

    return Binv
end

function (B::B_Times)(x::AbstractVector, z::AbstractVector)
    # Performs one matrix multiply of B*z, where B is the matrix used to solve for the
    # surface stresses that enforce the no-slip boundary condition.  (B arises from an LU
    # factorization of the full system)

    (; Ainv, C, E, Γtmp, qtmp, Γ, ψ, q) = B
    vort2flux! = B.vort2flux

    Γ .= 0 # TODO: This assignment might be unncessary. Check and potentially remove it

    # Get circulation from surface stress
    # Γ[:, 1] = Ainv * (E * C)' * z
    # Γ = ∇ x (E'*fb)
    mul!(qtmp, E', z)
    mul!(Γtmp, C', qtmp)
    mul!(view(Γ, :, 1), Ainv, Γtmp)

    # Get vel flux from circulation
    vort2flux!(ψ, q, Γ)

    # Interpolate onto the body
    @views mul!(x, E, q[:, 1])
end

Base.@kwdef struct Nonlinear{M}
    rhs_force::RhsForce
    grid::MultiLevelGrid
    C::M
    fq::Vector{Float64}
end

function (nonlinear::Nonlinear)(
    nonlin::AbstractVector, qty::PsiOmegaGridQuantities, Γbc::AbstractVector, lev::Int
)
    (; grid, C, fq) = nonlinear

    # Get flux-circulation product
    nonlinear.rhs_force(fq, qty, Γbc, lev)

    # Divergence of flux-circulation product
    mul!(nonlin, C', fq)

    # Scaling: 1/hc^2 to convert circulation to vorticity
    hc = gridstep(grid, lev) # Coarse grid spacing
    nonlin .*= 1 / hc^2

    return nothing
end

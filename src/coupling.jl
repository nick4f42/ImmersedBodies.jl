const reg_support = 3

struct Regularization
    grid::MultiDomainGrid
    panel_coords::Vector{NTuple{2,Int}}
    weights::Array{Float64,4}
end

function Regularization(prob::Problem)
    grid = prob.fluid.grid
    npanel = n_panels(prob.bodies)

    n_supp = 2 * reg_support + 1
    weights = zeros(npanel, 2, n_supp, n_supp)
    panel_coords = [(0, 0) for _ in 1:npanel]

    Regularization(grid, panel_coords, weights)
end

function _update!(reg::Regularization, indices, points::AbstractMatrix)
    for (index, point) in zip(indices, eachrow(points))
        _update!(reg, index, point)
    end
    reg
end

function _update!(reg::Regularization, index::Int, point::AbstractVector)
    grid = reg.grid.base

    h = grid.h
    nx, ny = grid.n
    x0, y0 = grid.x0
    xb, yb = point

    ix, iy = map((nx, ny), (x0, y0), (xb, yb)) do n, p0, pb
        # Coordinate, where i=1 is at the midpoint between the fluxes at the lower-left
        # corner, u[1,1] and v[1,1]
        i = round(Int, (pb - p0) / h + 1.0 - 0.25)
        if !(reg_support < i <= n - reg_support)
            error("Panel index=$index at $((xb, yb)) outside allowed region")
        end
        i
    end

    reg.panel_coords[index] = (ix, iy)

    # Get regularized weight at u flux and v flux midpoints
    x = @. x0 + h * (ix + ((-reg_support):reg_support) - 1)
    y = permutedims(@. y0 + h * (iy + ((-reg_support):reg_support) - 1))
    @. reg.weights[index, 1, :, :] = δh(x, xb, h) * δh(y + h / 2, yb, h)
    @. reg.weights[index, 2, :, :] = δh(x + h / 2, xb, h) * δh(y, yb, h)

    reg
end

# Interpolation onto the panels
function regT!(fb_flat::AbstractVector, q_flat::AbstractVector, reg::Regularization)
    npanel = length(reg.panel_coords)
    fb = reshape(fb_flat, npanel, 2)
    qx, qy = split_flux(q_flat, reg.grid.inds)

    for panel in 1:npanel
        i, j = _panel_fluid_inds(reg, panel)
        fb[panel, 1] = @views dot(qx[i, j], reg.weights[panel, 1, :, :])
        fb[panel, 2] = @views dot(qy[i, j], reg.weights[panel, 2, :, :])
    end

    fb_flat
end

# Regularization onto the fluid
function reg!(q_flat::AbstractVector, fb_flat::AbstractVector, reg::Regularization)
    npanel = length(reg.panel_coords)
    fb = reshape(fb_flat, npanel, 2)
    qx, qy = split_flux(q_flat, reg.grid.inds)

    q_flat .= 0
    for panel in 1:npanel
        i, j = _panel_fluid_inds(reg, panel)
        @views @. qx[i, j] += reg.weights[panel, 1, :, :] * fb[panel, 1]
        @views @. qy[i, j] += reg.weights[panel, 2, :, :] * fb[panel, 2]
    end

    q_flat
end

function _panel_fluid_inds(reg::Regularization, index::Int)
    map(i -> i .+ ((-reg_support):reg_support), reg.panel_coords[index])
end

function δh(rf::Real, rb::Real, dr::Real)
    # Discrete delta function used to relate flow to structure quantities

    # Take points on the flow domain (r) that are within the support (supp) of the IB points
    # (rb), and evaluate delta( abs(r - rb) )

    # Currently uses the Yang3 smooth delta function (see Yang et al, JCP, 2009), which has
    # a support of 6*h (3*h on each side)

    # Note that this gives slightly different answers than Fortran at around 1e-4,
    # apparently due to slight differences in the floating point arithmetic.  As far as I
    # can tell, this is what sets the bound on agreement between the two implementations.
    # It's possible this might be improved with arbitrary precision arithmetic (i.e.
    # BigFloats), but at least it doesn't seem to be a bug.

    # Note: the result is delta * h

    r = abs(rf - rb)
    r1 = r / dr
    r2 = r1 * r1
    r3 = r2 * r1
    r4 = r3 * r1

    return if (r1 <= 1.0)
        a5 = asin((1.0 / 2.0) * sqrt(3.0) * (2.0 * r1 - 1.0))
        a8 = sqrt(1.0 - 12.0 * r2 + 12.0 * r1)

        4.166666667e-2 * r4 +
        (-0.1388888889 + 3.472222222e-2 * a8) * r3 +
        (-7.121664902e-2 - 5.208333333e-2 * a8 + 0.2405626122 * a5) * r2 +
        (-0.2405626122 * a5 - 0.3792313933 + 0.1012731481 * a8) * r1 +
        8.0187537413e-2 * a5 - 4.195601852e-2 * a8 + 0.6485698427

    elseif (r1 <= 2.0)
        a6 = asin((1.0 / 2.0) * sqrt(3.0) * (-3.0 + 2.0 * r1))
        a9 = sqrt(-23.0 + 36.0 * r1 - 12.0 * r2)

        -6.250000000e-2 * r4 +
        (0.4861111111 - 1.736111111e-2 * a9) .* r3 +
        (-1.143175026 + 7.812500000e-2 * a9 - 0.1202813061 * a6) * r2 +
        (0.8751991178 + 0.3608439183 * a6 - 0.1548032407 * a9) * r1 - 0.2806563809 * a6 +
        8.22848104e-3 +
        0.1150173611 * a9

    elseif (r1 <= 3.0)
        a1 = asin((1.0 / 2.0 * (2.0 * r1 - 5.0)) * sqrt(3.0))
        a7 = sqrt(-71.0 - 12.0 * r2 + 60.0 * r1)

        2.083333333e-2 * r4 +
        (3.472222222e-3 * a7 - 0.2638888889) * r3 +
        (1.214391675 - 2.604166667e-2 * a7 + 2.405626122e-2 * a1) * r2 +
        (-0.1202813061 * a1 - 2.449273192 + 7.262731481e-2 * a7) * r1 +
        0.1523563211 * a1 +
        1.843201677 - 7.306134259e-2 * a7
    else
        0.0
    end
end

function B_operator(
    prob::Problem;
    reg::Regularization,
    vort2flux!,
    Ainv1!,
    Γ1_tmp::AbstractVector,
    q1_tmp::AbstractVector,
    Γ_tmp::AbstractMatrix,
    ψ_tmp::AbstractMatrix,
    q_tmp::AbstractMatrix,
)
    grid = prob.fluid.grid

    function (x, z)
        Γ_tmp .= 0

        # Get circulation from surface stress
        reg!(q1_tmp, z, reg)
        rot!(Γ1_tmp, q1_tmp, grid.inds)
        @views Ainv1!(Γ_tmp[:, 1], Γ1_tmp)

        # Get vel flux from circulation
        vort2flux!(ψ_tmp, q_tmp, Γ_tmp)

        # Interpolate onto the body
        @views regT!(x, q_tmp[:, 1], reg)
    end
end

function Binv_operator(prob::Problem; B!, cg_kw)
    if prob.bodies.static
        _Binv_precomputed(prob; B!)
    else
        _Binv_iterative(prob; B!, cg_kw)
    end
end

function _Binv_precomputed(prob::Problem; B!)
    # TODO: Use a matrix decomposition instead of inverting
    n = 2 * n_panels(prob.bodies)
    Binv = inv(_linear_op_matrix(B!, n))
    (x, z) -> mul!(x, Binv, z)
end

function _Binv_iterative(prob::Problem; B!, cg_kw)
    # NOTE: x is taken as the initial guess in cg!
    n = 2 * n_panels(prob.bodies)
    B = LinearMap(B!, n; ismutating=true, issymmetric=true)
    function (x, z)
        cg!(x, B, z; cg_kw...)
        x
    end
end

function _linear_op_matrix(f!, n::Int)
    x = zeros(n)
    M = zeros(n, n)
    for i in 1:n
        x[i] = 1
        if i > 1
            x[i - 1] = 0
        end
        @views f!(M[:, i], x)
    end
    M
end

function project_circ(grid::MultiDomainGrid; Ainv1!, reg, Γtmp, qtmp)
    function (state::State, Γs::AbstractMatrix)
        Γs1 = @view Γs[:, 1]

        state.Γ .= Γs

        reg!(qtmp, state.F̃b, reg)
        rot!(Γtmp, qtmp, grid.inds)
        Ainv1!(Γs1, Γtmp)  # use Γs as temporary buffer

        @views state.Γ[:, 1] .-= Γs1

        nothing
    end
end

function couple_surface_rigid(::Problem, grid::MultiDomainGrid; Binv!, reg, Ftmp, Q)
    h = gridstep(grid)

    function (state::State, qs::AbstractMatrix)
        # Solve the Poisson problem with nonzero boundary velocity ub

        @views @. Q = qs[:, 1] + state.q0[:, 1]

        regT!(Ftmp, Q, reg)

        ub = vec(state.panels.ub)  # Flattened velocities
        @. Ftmp -= ub * h  # Enforce no-slip conditions

        # If using the cg! iterative solver, this uses state.F̃b as the initial guess
        Binv!(state.F̃b, Ftmp)

        nothing
    end
end

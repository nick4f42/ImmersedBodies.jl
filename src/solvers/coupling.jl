struct Reg
    basegrid::UniformGrid
    gridindex::PsiOmegaGridIndexing
    body_idx::Matrix{Int}
    supp_idx::UnitRange{Int}
    weight::Array{Float64,4}
    idx_offsets::Vector{Int} # Cumulative index of each body
end

struct RegT
    reg::Reg
end

function E_linearmap(reg::Reg)
    regT = RegT(reg)

    nf = 2 * size(reg.body_idx, 1)
    nq = reg.gridindex.nq

    return LinearMap(regT, reg, nf, nq)
end

function Reg(prob::Problem{<:PsiOmegaFluidGrid}, panels::Panels)
    nf = 2 * npanels(panels)
    body_idx = zeros(Int, size(panels.pos))

    # TODO: Add external option for supp
    supp = 6
    supp_idx = (-supp):supp
    weight = zeros(nf, 2, 2 * supp + 1, 2 * supp + 1)

    basegrid = baselevel(discretized(prob.fluid))
    gridindex = prob.fluid.gridindex

    nbodies = length(panels.perbody)
    idx_offsets = zeros(Int, nbodies)
    for i in 1:(nbodies - 1)
        idx_offsets[i + 1] = idx_offsets[i] + npanels(panels.perbody[i])
    end

    reg = Reg(basegrid, gridindex, body_idx, supp_idx, weight, idx_offsets)
    for (i, point) in enumerate(eachrow(panels.pos))
        update!(reg, point, i)
    end

    return reg
end

function update!(reg::Reg, panels::PanelView, bodyindex::Int)
    offset = reg.idx_offsets[bodyindex]
    indices = offset .+ (1:npanels(panels))
    for (i, point) in zip(indices, eachrow(panels.pos))
        update!(reg, point, i)
    end
end

function update!(reg::Reg, bodypoint::AbstractVector, index::Int)
    h = gridstep(reg.basegrid)
    nx = reg.gridindex.nx
    ny = reg.gridindex.ny

    x0, y0 = minimum.(xycoords(reg.basegrid))
    px, py = bodypoint

    # Nearest indices of body relative to grid
    ibody = floor(Int, (px - x0) / h)
    jbody = floor(Int, (py - y0) / h)

    if (
        !all(in(1:nx), ibody .+ extrema(reg.supp_idx)) ||
        !all(in(1:ny), jbody .+ extrema(reg.supp_idx))
    )
        error("Body outside innermost fluid grid")
    end

    reg.body_idx[index, :] .= (ibody, jbody)

    # Get regularized weight near IB points (u-vel points)
    x = @. x0 + h * (ibody - 1 + reg.supp_idx)
    y = permutedims(@. y0 + h * (jbody - 1 + reg.supp_idx))
    @. reg.weight[index, 1, :, :] = δh(x, px, h) * δh(y + h / 2, py, h)
    @. reg.weight[index, 2, :, :] = δh(x + h / 2, px, h) * δh(y, py, h)

    return reg
end

function (reg::Reg)(q_flat, fb_flat)
    # Matrix E'

    nb = size(reg.body_idx, 1)

    q_flat .= 0
    fb = reshape(fb_flat, nb, 2)
    qx, qy = split_flux(q_flat, reg.gridindex)

    for k in 1:nb
        i = reg.body_idx[k, 1] .+ reg.supp_idx
        j = reg.body_idx[k, 2] .+ reg.supp_idx
        @views @. qx[i, j] += reg.weight[k, 1, :, :] * fb[k, 1]
        @views @. qy[i, j] += reg.weight[k, 2, :, :] * fb[k, 2]

        # TODO: Throw proper exception or remove
        if !isfinite(sum(x -> x^2, qx[i, j]))
            error("infinite flux")
        end
    end

    return q_flat
end

function (regT::RegT)(fb_flat, q_flat)
    # Matrix E
    reg = regT.reg

    nb = size(reg.body_idx, 1)

    fb_flat .= 0
    fb = reshape(fb_flat, nb, 2)
    qx, qy = split_flux(q_flat, reg.gridindex)

    for k in 1:nb
        i = reg.body_idx[k, 1] .+ reg.supp_idx
        j = reg.body_idx[k, 2] .+ reg.supp_idx
        fb[k, 1] += @views dot(qx[i, j], reg.weight[k, 1, :, :])
        fb[k, 2] += @views dot(qy[i, j], reg.weight[k, 2, :, :])
    end

    return fb_flat
end

function δh(rf, rb, dr)
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

abstract type SurfaceCoupler end

# Solve the Poisson equation (25) in Colonius & Taira (2008).
struct RigidSurfaceCoupler{MBinv,ME} <: SurfaceCoupler
    Binv::MBinv
    E::ME
    Ftmp::Vector{Float64}
    Q::Vector{Float64}
    h::Float64
end

function RigidSurfaceCoupler(; basegrid::UniformGrid, Binv, E, Ftmp, Q)
    h = gridstep(basegrid)
    return RigidSurfaceCoupler(Binv, E, Ftmp, Q, h)
end

function (coupler::RigidSurfaceCoupler)(state::StatePsiOmegaGridCNAB, qs)
    # Bodies moving in the grid frame
    # Solve the Poisson problem for bc2 = 0 (???) with nonzero boundary velocity ub
    # Bf̃ = Eq - ub
    #    = ECψ - ub

    (; F̃b, q0, panels) = quantities(state)
    (; Binv, E, Ftmp, Q, h) = coupler

    @views @. Q = qs[:, 1] + q0[:, 1]

    mul!(Ftmp, E, Q) # E*(qs .+ state.q0)

    ub = vec(panels.vel) # Flattened velocities

    # TODO: Is it worth dispatching to avoid this calculation for static bodies (ub = 0)
    @. Ftmp -= ub * h # Enforce no-slip conditions

    mul!(F̃b, Binv, Ftmp)

    return nothing
end

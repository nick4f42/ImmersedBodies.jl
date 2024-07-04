abstract type StructureOps end

Base.@kwdef mutable struct StackedStructureOps{O<:StructureOps,T1,T2,T3,T4}
    perbody::Vector{O}
    M::T1
    K::T2
    stru_to_fluid_offset::T3
    fluid_to_stru_force::T4
    Khat_inv::Matrix{Float64} # TODO: Use a factorization instead
    first::Bool # true if Khat_inv has not been initialized
    dt::Float64 # timestep of the problem
end

function StackedStructureOps(prob::Problem, panels::Panels)
    ops = [
        structure_ops(prob.bodies[i_body], panels.perbody[i_body]) for
        (; i_body) in prob.bodies.deforming
    ]

    Ms = Iterators.map(linearmap ∘ mass_matrix, ops)
    Ks = Iterators.map(linearmap ∘ stiff_matrix, ops)
    s2f = Iterators.map(linearmap ∘ structure_to_fluid_offset, ops)
    f2s = Iterators.map(linearmap ∘ fluid_to_structure_force, ops)

    # Stack the matrices for each body along the diagonal
    M = cat(Ms...; dims=(1, 2))
    K = cat(Ks...; dims=(1, 2))
    stru_to_fluid_offset = cat(s2f...; dims=(1, 2))
    fluid_to_stru_force = cat(f2s...; dims=(1, 2))

    return StackedStructureOps(;
        perbody=ops,
        M=M,
        K=K,
        stru_to_fluid_offset=stru_to_fluid_offset,
        fluid_to_stru_force=fluid_to_stru_force,
        Khat_inv=Matrix{Float64}(undef, size(K)),
        first=true,
        dt=timestep(prob),
    )
end

linearmap(x::LinearMap) = x
linearmap(x) = LinearMap(x)

function init!(
    ops::StackedStructureOps, bodies::BodyGroup, panels::Panels, states::DeformationState
)
    for (i_deform, (; i_body)) in enumerate(bodies.deforming)
        op = ops.perbody[i_deform]
        body = bodies[i_body]
        panel = panels.perbody[i_body]
        state = states.perbody[i_deform]
        init!(op, body, panel, state)
    end
end

function update!(
    ops::StackedStructureOps, bodies::BodyGroup, panels::Panels, states::DeformationState
)
    update_inv = ops.first
    ops.first = false

    for (i_deform, (; i_body)) in enumerate(bodies.deforming)
        op = ops.perbody[i_deform]
        body = bodies[i_body]
        panel = panels.perbody[i_body]
        state = states.perbody[i_deform]
        update_inv |= update!(op, body, panel, state)
    end

    if update_inv
        Khat = ops.K + (4 / ops.dt^2) * ops.M

        # TODO: use a factorization and avoid allocation
        ops.Khat_inv .= inv(Matrix(Khat))
    end

    return update_inv
end

struct EulerBernoulliOps{L1,L2} <: StructureOps
    M::Matrix{Float64}
    K::Matrix{Float64}
    Q::Matrix{Float64}
    struct_to_fluid::L1 # structure to fluid indices
    fluid_to_struct::L2 # fluid to structure indices
end

mass_matrix(ops::EulerBernoulliOps) = ops.M
stiff_matrix(ops::EulerBernoulliOps) = ops.K

function structure_to_fluid_offset(ops::EulerBernoulliOps)
    return ops.struct_to_fluid
end

function fluid_to_structure_force(ops::EulerBernoulliOps)
    return ops.Q * ops.fluid_to_struct
end

function structure_ops(body::EulerBernoulliBeam{LinearModel}, ::PanelView)
    nb = npanels(body)

    n = 2 * nb

    M = zeros(n, n)
    K = zeros(n, n)
    Q = zeros(n, n)

    struct_to_fluid = LinearMap(n) do x_fluid, x_struct
        x_fluid .= vec(transpose(reshape(x_struct, 2, nb)))
    end

    fluid_to_struct = LinearMap(n) do x_struct, x_fluid
        x_struct .= vec(transpose(reshape(x_fluid, nb, 2)))
    end

    return EulerBernoulliOps(M, K, Q, struct_to_fluid, fluid_to_struct)
end

function init!(
    ops::EulerBernoulliOps,
    body::EulerBernoulliBeam{LinearModel},
    panels::PanelView,
    ::DeformationStateView,
)
    nel = npanels(body) - 1 # Number of finite elements
    (; M, K, Q) = ops
    M .= 0
    K .= 0
    Q .= 0

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δs = panels.len[i_el]
        m = body.model.m[i_el]
        kb = body.model.kb[i_el]

        # Indices corresponding with the 4 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 2 + (1:4)

        # M_edup1 in the previous version
        Q_e =
            Δs / 420 * @SMatrix [
                156 22*Δs 54 -13*Δs
                22*Δs 4*Δs^2 -13*Δs -3*Δs^2
                54 13*Δs 156 -22*Δs
                13*Δs -3*Δs^2 -22*Δs 4*Δs^2
            ]

        M_e =
            m * Δs / 420 * @SMatrix [
                156 22*Δs 54 -13*Δs
                22*Δs 4*Δs^2 13*Δs -3*Δs^2
                54 13*Δs 156 -22*Δs
                -13*Δs -3*Δs^2 -22*Δs 4*Δs^2
            ]

        K_e =
            1 / (Δs^3) * @SMatrix [
                kb*12 kb*6*Δs -kb*12 kb*6*Δs
                kb*6*Δs kb*4*Δs^2 -kb*6*Δs kb*2*Δs^2
                -kb*12 -kb*6*Δs kb*12 -kb*6*Δs
                kb*6*Δs kb*2*Δs^2 -kb*6*Δs kb*4*Δs^2
            ]

        # Assemble into global matrices
        # Add contributions for each DOF in the element
        for (i, i_ind) in enumerate(el_ind), (j, j_ind) in enumerate(el_ind)
            M[i_ind, j_ind] += M_e[i, j]
            K[i_ind, j_ind] += K_e[i, j]
            Q[i_ind, j_ind] += Q_e[i, j]
        end
    end

    # Account for BCs
    for bc in body.bcs
        i = bc_point(bc).i
        for j in bc_indices(LinearModel, bc)
            k = 2 * (i - 1) + j
            M[k, :] .= 0.0
            M[:, k] .= 0.0
            K[k, :] .= 0.0
            K[:, k] .= 0.0
            Q[k, :] .= 0.0
            # Q[:, k] .= 0.0
            K[k, k] = 1.0
        end
    end
end

bc_indices(::Type{LinearModel}, ::ClampBC) = (1, 2)

function update!(
    ::EulerBernoulliOps,
    ::EulerBernoulliBeam{LinearModel},
    ::PanelView,
    ::DeformationStateView,
)
    return false # indicate no changes
end

struct SpringedMembraneOps{L1,L2,L3} <: StructureOps
    M::Matrix{Float64}
    K::Matrix{Float64}
    spring_to_fluid_offset::L1
    fluid_to_spring_force::L2
    spring_to_fluid_pos::L3
end

mass_matrix(ops::SpringedMembraneOps) = ops.M
stiff_matrix(ops::SpringedMembraneOps) = ops.K

structure_to_fluid_offset(ops::SpringedMembraneOps) = ops.spring_to_fluid_offset
fluid_to_structure_force(ops::SpringedMembraneOps) = ops.fluid_to_spring_force
structure_to_fluid_position(ops::SpringedMembraneOps) = ops.spring_to_fluid_pos

function structure_ops(body::SpringedMembrane, panels::PanelView)
    n_spring = length(body.m)
    nb = npanels(body)
    nf = 2 * nb

    M = zeros(n_spring, n_spring)
    K = zeros(n_spring, n_spring)

    spring_to_fluid_offset = LinearMap(nf, n_spring) do dx_body, dx_spring
        dx_body .= vec(body.deform_weights) .* dx_spring[end]
        return dx_body
    end

    fluid_to_spring_force = LinearMap(n_spring, nf) do f_spring, f_body
        fx, fy = eachcol(reshape(f_body, nb, 2))
        ux, uy = body.spring_normal

        f_spring .= 0
        f_spring[end] = sum(1:nb) do i
            (fx[i] * ux + fy[i] * uy) * panels.len[i]
        end

        return f_spring
    end

    function spring_to_fluid_pos(dx_body, dx_spring)
        k = dx_spring[end]
        dx = reshape(dx_body, :, 2)
        _, pts = PointSpacing.distribute_points(body.deformed(k); n=size(dx, 1))
        for I in CartesianIndices(dx)
            i, j = Tuple(I)
            dx[I] = pts[i][j] - body.xref[i,j]
        end
        dx_body
    end

    return SpringedMembraneOps(M, K, spring_to_fluid_offset, fluid_to_spring_force, spring_to_fluid_pos)
end

function init!(
    ops::SpringedMembraneOps, body::SpringedMembrane, ::PanelView, ::DeformationStateView
)
    n = length(body.m)

    ops.M .= 0
    for i in 1:n
        ops.M[i, i] = body.m[i]
    end

    Kt = diagm(body.k)

    A = zeros(n, n)
    A[1, 1] = 1
    for i in 2:n
        A[i, i] = 1
        A[i, i - 1] = -1
    end

    ops.K .= transpose(A) * Kt * A
    ops.K[diagind(ops.K)] .+= body.kg

    return nothing
end

function update!(
    ops::SpringedMembraneOps, body::SpringedMembrane, ::PanelView, deform_k::DeformationStateView
)
    k = deform_k.χ[end]

    vel = material_pt_velocity(body.deformed, k; s_undef_end=body.s_undef_end)
    ts = range(0, 1, size(body.deform_weights, 1))
    for i in axes(body.deform_weights, 1)
        body.deform_weights[i, :] = vel(ts[i])
    end

    return false # indicate no changes
end

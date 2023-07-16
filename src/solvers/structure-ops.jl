abstract type StructureOps end

Base.@kwdef mutable struct StackedStructureOps{O<:StructureOps,T1,T2,T3,T4}
    perbody::Vector{O}
    M::T1
    K::T2
    stru_to_fluid_offset::T3
    fluid_to_stru_force::T4
    Fint::Vector{Float64} # Internal forces
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

    Fint = zeros(size(M, 1))

    return StackedStructureOps(;
        perbody=ops,
        M=M,
        K=K,
        stru_to_fluid_offset=stru_to_fluid_offset,
        fluid_to_stru_force=fluid_to_stru_force,
        Fint=Fint,
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


struct BeforeFsiLoop end
struct BetweenFsiLoops end

function update!(
    ops::StackedStructureOps, when, bodies::BodyGroup, panels::Panels, states::DeformationState
)
    update_inv = ops.first
    ops.first = false

    i_offset = 0

    for (i_deform, (; i_body)) in enumerate(bodies.deforming)
        op = ops.perbody[i_deform]
        body = bodies[i_body]
        panel = panels.perbody[i_body]
        state = states.perbody[i_deform]
        update_inv |= update!(op, when, body, panel, state)

        # Update internal forces
        Fint_body = _get_Fint(op)
        ops.Fint[i_offset .+ eachindex(Fint_body)] .= Fint_body
        i_offset += length(Fint_body)
    end

    if update_inv
        Khat = ops.K + (4 / ops.dt^2) * ops.M

        # TODO: use a factorization and avoid allocation
        ops.Khat_inv .= inv(Matrix(Khat))
    end

    return update_inv
end

struct LinearEulerBernoulliOps{L1,L2} <: StructureOps
    M::Matrix{Float64}
    K::Matrix{Float64}
    Q::Matrix{Float64}
    struct_to_fluid::L1 # structure to fluid indices
    fluid_to_struct::L2 # fluid to structure indices
end

mass_matrix(ops::LinearEulerBernoulliOps) = ops.M
stiff_matrix(ops::LinearEulerBernoulliOps) = ops.K

function structure_to_fluid_offset(ops::LinearEulerBernoulliOps)
    return ops.struct_to_fluid
end

function fluid_to_structure_force(ops::LinearEulerBernoulliOps)
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

    return LinearEulerBernoulliOps(M, K, Q, struct_to_fluid, fluid_to_struct)
end

function init!(
    ops::LinearEulerBernoulliOps,
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
    ::LinearEulerBernoulliOps,
    ::EulerBernoulliBeam{LinearModel},
    ::PanelView,
    ::DeformationStateView,
)
    return false # indicate no changes
end

struct NonlinearEulerBernoulliOps{L0,L1,L2} <: StructureOps
    M::Matrix{Float64}
    K::Matrix{Float64}
    Q::L0
    β0::Vector{Float64}
    Fint::Vector{Float64} # Internal forces
    struct_to_fluid::L1 # structure to fluid indices
    fluid_to_struct::L2 # fluid to structure indices
end

mass_matrix(ops::NonlinearEulerBernoulliOps) = ops.M
stiff_matrix(ops::NonlinearEulerBernoulliOps) = ops.K

function structure_to_fluid_offset(ops::NonlinearEulerBernoulliOps)
    return ops.struct_to_fluid
end

function fluid_to_structure_force(ops::NonlinearEulerBernoulliOps)
    return ops.Q * ops.fluid_to_struct
end

function structure_ops(body::EulerBernoulliBeam{NonlinearModel}, ::PanelView)
    nb = npanels(body)
    nel = nb - 1 # Number of structural elements
    nf = 2 * nb # Number of x and y fluid panel coordinates
    n = 3 * nb # Number of structural variables

    M = zeros(n, n)
    K = zeros(n, n)

    # build_force in the FORTRAN code
    Q = LinearMap(n) do y, x
        y .= 0
        for i in 1:nel
            ds0 = body.ds0[i]
            el_ind = @. (i - 1) * 3 + (1:6)

            fx1, fy1, _, fx2, fy2, _ = view(x, el_ind)

            F_e = @SVector [
                ds0 / 3 * fx1 + ds0 / 6 * fx2,
                26 * ds0 / 70 * fy1 + 9 * ds0 / 70 * fy2,
                -11 * ds0^2 / 210 * fy1 - ds0^2 * 13 / 420 * fy2,
                ds0 / 6 * fx1 + ds0 / 3 * fx2,
                9 * ds0 / 70 * fy1 + 26 * ds0 / 70 * fy2,
                13 * ds0^2 / 420 * fy1 + ds0^2 * 11 / 210 * fy2,
            ]

            @views y[el_ind] .+= F_e
        end

        _apply_bcs!(y, NonlinearModel, body.bcs)

        return y
    end

    struct_to_fluid = LinearMap(nf, n) do x_fluid, x_struct
        a_struct = reshape(x_struct, 3, nb)
        x_fluid .= @views vec(transpose(a_struct[1:2, :]))
        return x_fluid
    end

    fluid_to_struct = LinearMap(n, nf) do x_struct, x_fluid
        a_struct = reshape(x_struct, 3, nb)
        a_fluid = reshape(x_fluid, nb, 2)
        a_struct[1:2, :] .= transpose(a_fluid)
        a_struct[3, :] .= 0
        return x_struct
    end

    β0 = Vector{Float64}(undef, nel)
    for i in 1:nel
        dx0 = body.xref[i + 1, 1] - body.xref[i, 1]
        dy0 = body.xref[i + 1, 2] - body.xref[i, 2]
        β0[i] = atan(dy0, dx0)
    end

    Fint = zeros(n)

    return NonlinearEulerBernoulliOps(M, K, Q, β0, Fint, struct_to_fluid, fluid_to_struct)
end

function init!(
    ::NonlinearEulerBernoulliOps,
    ::EulerBernoulliBeam{NonlinearModel},
    ::PanelView,
    ::DeformationStateView,
)
    return nothing
end

function _set_M!(
    ops::NonlinearEulerBernoulliOps,
    body::EulerBernoulliBeam{NonlinearModel},
    panels::PanelView,
    deform::DeformationStateView,
)
    nb = npanels(body) # Number of body points
    nel = nb - 1 # Number of finite elements
    M = ops.M

    M .= 0

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δs = panels.len[i_el]
        m = body.model.m[i_el]

        # Indices corresponding with the 6 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 3 + (1:6)

        M_e =
            m * Δs / 420 * @SMatrix [
                140 0 0 70 0 0
                0 156 22*Δs 0 54 -13*Δs
                0 22*Δs 4*Δs^2 0 13*Δs -3*Δs^2
                70 0 0 140 0 0
                0 54 13*Δs 0 156 -22*Δs
                0 -13*Δs -3*Δs^2 0 -22*Δs 4*Δs^2
            ]

        # Assemble into global matrices
        # Add contributions for each DOF in the element
        @views @. M[el_ind, el_ind] .+= M_e
    end

    # Account for BCs
    for bc in body.bcs
        i = bc_point(bc).i
        for j in bc_indices(NonlinearModel, bc)
            k = 3 * (i - 1) + j
            M[k, :] .= 0.0
            M[:, k] .= 0.0
        end
    end
end

function _set_K!(
    ops::NonlinearEulerBernoulliOps,
    body::EulerBernoulliBeam{NonlinearModel},
    panels::PanelView,
    deform::DeformationStateView,
)
    nb = npanels(body) # Number of body points
    nel = nb - 1 # Number of finite elements
    K = ops.K
    χ = deform.χ

    K .= 0

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δs = panels.len[i_el]
        Δs0 = body.ds0[i_el]
        kb = body.model.kb[i_el]
        ke = body.model.ke[i_el]

        # Indices corresponding with the 6 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 3 + (1:6)

        # ke is equivalent to K_s in the Fortran version
        r_c = kb / ke
        CL = ke / Δs0 * @SMatrix [
            1 0 0
            0 4*r_c 2*r_c
            0 2*r_c 4*r_c
        ]

        dx = panels.pos[i_el + 1, 1] - panels.pos[i_el, 1]
        dy = panels.pos[i_el + 1, 2] - panels.pos[i_el, 2]
        cβ = dx / Δs
        sβ = dy / Δs

        B = @SMatrix [
            -cβ -sβ 0 cβ sβ 0
            -sβ/Δs cβ/Δs 1 sβ/Δs -cβ/Δs 0
            -sβ/Δs cβ/Δs 0 sβ/Δs -cβ/Δs 1
        ]

        K1 = B' * CL * B

        z = SVector(sβ, -cβ, 0, -sβ, cβ, 0)
        r = -SVector(cβ, sβ, 0, -cβ, sβ, 0)

        # Better conditioned formula for Δs-Δs0 when the difference is small
        uL = (Δs^2 - Δs0^2) / (Δs + Δs0)

        Nf = ke * uL / Δs0

        θ1 = χ[el_ind[3]]
        θ2 = χ[el_ind[6]]

        β0 = ops.β0[i_el]
        β1 = θ1 + β0
        β2 = θ2 + β0

        θ1L = atan(cβ * sin(β1) - sβ * cos(β1), cβ * cos(β1) + sβ * sin(β1))
        θ2L = atan(cβ * sin(β2) - sβ * cos(β2), cβ * cos(β2) + sβ * sin(β2))

        Mf1 = 2 * kb / Δs0 * (2 * θ1L + θ2L)
        Mf2 = 2 * kb / Δs0 * (θ1L + 2 * θ2L)

        Kσ = Nf / Δs * z * z' + (Mf1 + Mf2) / Δs^2 * (r * z' + z * r')

        K_e = K1 + Kσ

        # Assemble into global matrices
        # Add contributions for each DOF in the element
        @views @. K[el_ind, el_ind] .+= K_e
    end

    # Account for BCs
    for bc in body.bcs
        i = bc_point(bc).i
        for j in bc_indices(NonlinearModel, bc)
            k = 3 * (i - 1) + j
            K[k, :] .= 0.0
            K[:, k] .= 0.0
            K[k, k] = 1.0
        end
    end

    return nothing
end

function _set_Fint!(
    ops::NonlinearEulerBernoulliOps,
    body::EulerBernoulliBeam{NonlinearModel},
    panels::PanelView,
    deform::DeformationStateView,
)
    nb = npanels(body) # Number of body points
    nel = nb - 1 # Number of finite elements
    Fint = ops.Fint
    χ = deform.χ

    Fint .= 0

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δs = panels.len[i_el]
        Δs0 = body.ds0[i_el]
        kb = body.model.kb[i_el]
        ke = body.model.ke[i_el]

        # Indices corresponding with the 6 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 3 + (1:6)

        dx = panels.pos[i_el + 1, 1] - panels.pos[i_el, 1]
        dy = panels.pos[i_el + 1, 2] - panels.pos[i_el, 2]
        cβ = dx / Δs
        sβ = dy / Δs

        B = @SMatrix [
            -cβ -sβ 0 cβ sβ 0
            -sβ/Δs cβ/Δs 1 sβ/Δs -cβ/Δs 0
            -sβ/Δs cβ/Δs 0 sβ/Δs -cβ/Δs 1
        ]

        # Better conditioned formula for Δs-Δs0 when the difference is small
        uL = (Δs^2 - Δs0^2) / (Δs + Δs0)

        Nf = ke * uL / Δs0

        θ1 = χ[el_ind[3]]
        θ2 = χ[el_ind[6]]

        β0 = ops.β0[i_el]
        β1 = θ1 + β0
        β2 = θ2 + β0

        θ1L = atan(cβ * sin(β1) - sβ * cos(β1), cβ * cos(β1) + sβ * sin(β1))
        θ2L = atan(cβ * sin(β2) - sβ * cos(β2), cβ * cos(β2) + sβ * sin(β2))

        Mf1 = 2 * kb / Δs0 * (2 * θ1L + θ2L)
        Mf2 = 2 * kb / Δs0 * (θ1L + 2 * θ2L)

        qL = @SVector [Nf, Mf1, Mf2]

        # Internal forces in global frame
        qint = B' * qL

        @views @. Fint[el_ind] .+= qint
    end

    _apply_bcs!(Fint, NonlinearModel, body.bcs)

    return nothing
end

function _apply_bcs!(x, model, bcs)
    for bc in bcs
        _apply_bc!(x, model, bc)
    end
end

function _apply_bc!(v::AbstractVector, model::Type{NonlinearModel}, bc::DeformingBodyBC)
    i = bc_point(bc).i
    for j in bc_indices(model, bc)
        k = 3 * (i - 1) + j
        v[k] = 0.0
    end
    return nothing
end

function update!(
    ops::NonlinearEulerBernoulliOps,
    ::BeforeFsiLoop,
    body::EulerBernoulliBeam{NonlinearModel},
    panels::PanelView,
    deform::DeformationStateView,
)
    _set_M!(ops, body, panels, deform)
    _set_K!(ops, body, panels, deform)
    _set_Fint!(ops, body, panels, deform)

    return true # indicate that K has changed
end

function update!(
    ops::NonlinearEulerBernoulliOps,
    ::BetweenFsiLoops,
    body::EulerBernoulliBeam{NonlinearModel},
    panels::PanelView,
    deform::DeformationStateView,
)
    _set_K!(ops, body, panels, deform)
    _set_Fint!(ops, body, panels, deform)

    return true # indicate that K has changed
end

function _get_Fint(ops::NonlinearEulerBernoulliOps)
    return ops.Fint
end

# Restricts x, y, θ
bc_indices(::Type{NonlinearModel}, ::ClampBC) = (1, 2, 3)

# Restricts x, y
bc_indices(::Type{NonlinearModel}, ::PinBC) = (1, 2)

struct SpringedMembraneOps{L1,L2} <: StructureOps
    M::Matrix{Float64}
    K::Matrix{Float64}
    spring_to_fluid_offset::L1
    fluid_to_spring_force::L2
end

mass_matrix(ops::SpringedMembraneOps) = ops.M
stiff_matrix(ops::SpringedMembraneOps) = ops.K

structure_to_fluid_offset(ops::SpringedMembraneOps) = ops.spring_to_fluid_offset
fluid_to_structure_force(ops::SpringedMembraneOps) = ops.fluid_to_spring_force

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

    return SpringedMembraneOps(M, K, spring_to_fluid_offset, fluid_to_spring_force)
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
    ::SpringedMembraneOps, ::SpringedMembrane, ::PanelView, ::DeformationStateView
)
    return false # indicate no changes
end

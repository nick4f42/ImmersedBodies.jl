@kwdef mutable struct CNAB{N,T,B,U,P,R<:Reg,M,Au,Aω,Ab,Aib,IB<:ImmersedBody{N,T,Aib}}
    const prob::IBProblem{N,T,B,U}
    const t0::T
    i::Int
    t::T
    const dt::T
    const β::Vector{T}
    const plan::P
    const reg::R
    const Binv::M
    ω::Vector{Aω}
    ψ::Vector{Aω}
    const u::Vector{Au}
    const f_tilde::Aib
    const f_work::Aib
    const ib::IB
    const nonlin::Vector{Vector{Aω}}
    nonlin_count::Int
    const u_work::Au
    const ω_work::Aω
    const ψ_b_work::Ab
end

function CNAB(
    prob::IBProblem{N,T}; dt, t0=zero(T), n_step=2, delta=DeltaYang3S(), backend=CPU()
) where {N,T}
    (; grid, body) = prob
    ω = grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels)

    plan = let ωe = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())
        laplacian_plans(ωe, grid.n)
    end

    n_ib = maxpoints(body)

    args = (;
        prob,
        t0,
        i=0,
        t=zero(T),
        dt,
        β=ab_coeffs(T, n_step),
        plan,
        reg=Reg(backend, T, delta, n_ib, Val(N)),
        ω,
        ψ=grid_zeros(backend, grid, Loc_ω; levels=1:grid.levels),
        u=grid_zeros(backend, grid, Loc_u; levels=1:grid.levels),
        f_tilde=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        f_work=KernelAbstractions.zeros(backend, SVector{N,T}, n_ib),
        ib=ImmersedBody{N,T}(backend, n_ib),
        nonlin=map(1:n_step-1) do _
            grid_zeros(backend, grid, Loc_ω, ExcludeBoundary(); levels=1:grid.levels)
        end,
        nonlin_count=0,
        u_work=grid_zeros(backend, grid, Loc_u),
        ω_work=grid_zeros(backend, grid, Loc_ω),
        ψ_b_work=boundary_zeros(backend, grid, Loc_ω),
    )

    sol = let sol = CNAB(; args..., Binv=nothing)
        init_body!(sol.ib, sol.prob.body)
        init_reg!(sol)
        CNAB(; args..., Binv=coupling_step_inverse(sol))
    end

    set_time!(sol, 0)

    for level in 1:grid.levels
        for i in eachindex(sol.ω[level])
            sol.ω[level][i] .= 0
            sol.ψ[level][i] .= 0
        end
        for i in eachindex(sol.u[level])
            sol.u[level][i] .= 0
        end
    end

    for level in eachindex(sol.u)
        add_flow!(sol.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
    end

    sol
end

function set_time!(sol::CNAB, i::Integer)
    sol.i = i
    sol.t = sol.t0 + sol.dt * (i - 1)
    sol
end

function step!(sol::CNAB)
    set_time!(sol, sol.i + 1)

    prediction_step!(sol)
    coupling_step!(sol)
    projection_step!(sol)
    apply_vorticity!(sol)

    sol
end

function init_reg!(sol::CNAB{N,T,<:AbstractStaticBody}) where {N,T}
    update_weights!(sol.reg, sol.prob.grid, sol.ib.x)
end

update_reg!(sol::CNAB{N,T,<:AbstractStaticBody}) where {N,T} = nothing

_A_factor(sol::CNAB) = sol.dt / (2sol.prob.Re)

function Ainv(sol::CNAB, level)
    h = gridstep(sol.prob.grid, level)
    a = _A_factor(sol)
    EigenbasisTransform(λ -> 1 / (1 - a * λ / h^2), sol.plan)
end

function prediction_step!(sol::CNAB)
    _cycle!(sol.nonlin)

    for level in sol.prob.grid.levels:-1:1
        prediction_step!(sol, level)
    end

    sol.nonlin_count = min(sol.nonlin_count + 1, length(sol.nonlin))
end

function prediction_step!(sol::CNAB, level)
    grid = sol.prob.grid
    h = gridstep(grid, level)
    ωˢ = grid_view(sol.ψ[level], grid, Loc_ω, ExcludeBoundary())
    u_work = grid_view(sol.u_work, grid, Loc_u, ExcludeBoundary())
    a = _A_factor(sol)

    curl!(u_work, sol.ω[level]; h)
    rot!(ωˢ, u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i], ω = sol.ω[level][i]
            @loop ωˢ (I in ωˢ) ωˢ[I] = ω[I] - a * ωˢ[I]
        end
    end

    if level < grid.levels
        multidomain_interpolate!(sol.ψ_b_work, sol.ψ[level+1]; n=grid.n)
        add_laplacian_bc!(ωˢ, a / h^2, sol.ψ_b_work)
    end

    nonlin_full = sol.nonlin_count == length(sol.nonlin)

    if nonlin_full
        for i_step in eachindex(sol.nonlin), i in eachindex(ωˢ)
            let ωˢ = ωˢ[i], N = sol.nonlin[i_step][level][i], k = sol.dt * sol.β[end-i_step]
                @loop ωˢ (I in ωˢ) ωˢ[I] = ωˢ[I] + k * N[I]
            end
        end
    end

    nonlinear!(u_work, sol.u[level], sol.ω[level])
    rot!(sol.nonlin[end][level], u_work; h)

    for i in eachindex(ωˢ)
        let ωˢ = ωˢ[i],
            N = sol.nonlin[end][level][i],
            k = nonlin_full ? sol.dt * sol.β[end] : sol.dt

            @loop ωˢ (I in ωˢ) ωˢ[I] = ωˢ[I] + k * N[I]
        end
    end

    Ainv(sol, level)(ωˢ, ωˢ)
end

function coupling_step!(sol::CNAB)
    grid = sol.prob.grid
    ωˢ = sol.ψ
    ψ = (sol.ω[1],)
    u¹ = sol.u[1]
    rhs = sol.f_work

    multidomain_poisson!(ωˢ, ψ, (u¹,), sol.ψ_b_work, grid, sol.plan)
    add_flow!(u¹, sol.prob.u0, grid, 1, sol.i, sol.t)
    interpolate_body!(rhs, sol.reg, u¹)

    update_body!(sol.ib, sol.prob.body, sol.i, sol.t)
    update_reg!(sol)

    rhs .-= sol.ib.u

    sol.Binv(sol.f_tilde, rhs, sol)
end

struct CNAB_Binv_Precomputed{M}
    B::M
end

function (x::CNAB_Binv_Precomputed)(f, u_ib, sol::CNAB{N,T}) where {N,T}
    flatten(a) = vec(reinterpret(reshape, T, a))

    let f = flatten(f), u_ib = flatten(u_ib)
        ldiv!(f, x.B, u_ib)
    end
end

function coupling_step_inverse(sol::CNAB{N,T,<:AbstractStaticBody}) where {N,T}
    backend = get_backend(sol.f_tilde)
    n_ib = maxpoints(sol.prob.body)

    n = N * n_ib
    B_map = LinearMap(n; ismutating=true) do u_ib, f
        coupling_step_matmul!(u_ib, f, sol)
    end
    f = vec(reinterpret(reshape, T, sol.f_work))
    B_mat = KernelAbstractions.zeros(backend, T, n, n)
    for i in 1:n
        @. f = ifelse((1:n) == i, 1, 0)
        mul!(@view(B_mat[:, i]), B_map, f)
    end
    CNAB_Binv_Precomputed(lu!(B_mat))
end

function coupling_step_matmul!(
    u_ib::AbstractVector{<:Number}, f, sol::CNAB{N,T}
) where {N,T}
    unflatten(a) = reinterpret(reshape, SVector{N,T}, reshape(a, N, :))

    let u_ib = unflatten(u_ib), f = unflatten(f)
        coupling_step_matmul!(u_ib, f, sol)
    end

    u_ib
end

function coupling_step_matmul!(u_ib, f, sol::CNAB)
    grid = sol.prob.grid
    h = grid.h
    u¹ = sol.u_work
    ψ¹ = sol.ω_work
    ω = sol.ω
    ω¹ = grid_view(ω[1], grid, Loc_ω, ExcludeBoundary())

    regularize!(u¹, sol.reg, f)
    rot!(ω¹, u¹; h)
    Ainv(sol, 1)(ω¹, ω¹)

    for level in 2:grid.levels, i in eachindex(ω[level])
        fill!(ω[level][i], 0)
    end

    multidomain_poisson!(ω, (ψ¹,), (u¹,), sol.ψ_b_work, grid, sol.plan)
    interpolate_body!(u_ib, sol.reg, u¹)
end

function projection_step!(sol::CNAB)
    grid = sol.prob.grid

    regularize!(sol.u_work, sol.reg, sol.f_tilde)

    (sol.ω, sol.ψ) = (sol.ψ, sol.ω)

    ω_work = grid_view(sol.ω_work, grid, Loc_ω, ExcludeBoundary())

    rot!(ω_work, sol.u_work; h=grid.h)
    Ainv(sol, 1)(ω_work, ω_work)
    for i in eachindex(ω_work)
        let ω = sol.ω[1][i], ω_work = ω_work[i]
            @loop ω (I in ω_work) ω[I] -= ω_work[I]
        end
    end
end

function apply_vorticity!(sol::CNAB)
    grid = sol.prob.grid
    multidomain_poisson!(sol.ω, sol.ψ, sol.u, sol.ψ_b_work, grid, sol.plan)

    for level in 1:grid.levels
        if level == grid.levels
            for i in eachindex(sol.ψ_b_work)
                foreach(b -> fill!(b, 0), sol.ψ_b_work[i])
            end
        else
            multidomain_interpolate!(sol.ψ_b_work, sol.ω[level+1]; n=grid.n)
        end

        set_boundary!(sol.ω[level], sol.ψ_b_work)

        add_flow!(sol.u[level], sol.prob.u0, grid, level, sol.i, sol.t)
    end
end

function ab_coeffs(T, n)
    if n == 1
        T[1]
    elseif n == 2
        T[-1//3, 3//2]
    else
        throw(DomainError(n, "only n=1 and n=2 are supported"))
    end
end

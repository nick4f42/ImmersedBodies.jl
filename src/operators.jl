function nonlinear!(nonlin, u, ω)
    for (i, nonlinᵢ) in pairs(nonlin)
        @loop nonlinᵢ (I in nonlinᵢ) nonlinᵢ[I] = nonlinear(i, u, ω, I)
    end
    nonlin
end

function nonlinear(i, u, ω, I)
    δ = unit(length(I))
    permute(i, vec_kind(u), vec_kind(ω)) do j, k
        let ω = ensure_3d(ω)
            uI = (u[j][I] + u[j][I-δ(i)] + u[j][I+δ(j)] + u[j][I-δ(i)+δ(j)]) / 4
            ωI = (ω[k][I] + ω[k][I+δ(j)]) / 2
            uI * ωI
        end
    end
end

function rot!(ω, u; h)
    for (i, ωᵢ) in axispairs(ω)
        @loop ωᵢ (I in ωᵢ) ωᵢ[I] = rot(i, u, I; h)
    end
    ω
end

function rot(i, u, I; h)
    δ = unit(length(I))
    permute(i) do j, k
        (u[k][I] - u[k][I-δ(j)]) / h
    end
end

function nonlinear!(nonlin::Tuple, u, ω)
    for (i, nonlinᵢ) in pairs(nonlin)
        @loop nonlinᵢ (I in nonlinᵢ) nonlinᵢ[I] = nonlinear(i, u, ω, I)
    end
    nonlin
end

function nonlinear(i::Int, u, ω, I)
    δ = unit(length(I))
    permute(i, vec_kind(u), vec_kind(ω)) do j, k
        let ω = ensure_3d(ω)
            uI = (u[j][I] + u[j][I-δ(i)] + u[j][I+δ(j)] + u[j][I-δ(i)+δ(j)]) / 4
            ωI = (ω[k][I] + ω[k][I+δ(j)]) / 2
            uI * ωI
        end
    end
end

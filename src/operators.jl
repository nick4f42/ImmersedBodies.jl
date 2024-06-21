function cross!(nonlin::GridArray{LocVec_u}, u::GridArray{LocVec_u}, ω::GridArray{LocVec_ω})
    for (i, nonlinᵢ) in axispairs(nonlin)
        @loop (I in nonlinᵢ) nonlinᵢ[I] = _cross(setaxis(nonlin, i), u, ω, I)
    end
end

function _cross(
    loc::Loc_u, u::GridArray{LocVec_u,T,N}, ω::GridArray{LocVec_ω,T,N}, I
) where {T,N}
    let i = loc.i, (u, ω) = unwrap.((u, ω))
        permute(i) do (j, k)
            if j ≤ N && k ≤ N
                uI = _interp(loc, u[j], (i, j, k), I)
                ωI = _interp(loc, ω[k], (i, j, k), I)
                uI * ωI
            else
                zero(T)
            end
        end
    end
end

function _interp(loc::Loc_u, u::GridArray{Loc_u}, I)
    let i = loc.i, (uⱼ, j) = array_axis(u), δ = δ_for(I)
        (uⱼ[I] + uⱼ[I-δ(i)] + uⱼ[I+δ(j)] + uⱼ[I-δ(i)+δ(j)]) / 4
    end
end

function _interp(loc::Loc_ω, ω::GridArray{Loc_ω}, j, I)
    let ωₖ = array_axis(ω), δ = δ_for(I)
        (ωₖ[I] + ωₖ[I+δ(j)]) / 2
    end
end

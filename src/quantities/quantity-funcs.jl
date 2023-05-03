function freestream_velocity end
function flow_velocity end
function streamfunction end
function vorticity end
function body_point_pos end
function body_point_vel end
function body_traction end
function body_lengths end

for (qty, field) in [
    (:body_point_pos, :pos),
    (:body_point_vel, :vel),
    (:body_lengths, :len),
    (:body_traction, :traction),
]
    @eval $qty(prob::Problem; bodyindex=1:length(prob.bodies)) = $qty(prob, bodyindex)

    @eval function $qty(::Problem, bodyindex::Integer)
        return state::AbstractState -> bodypanels(state).perbody[bodyindex].$field
    end

    @eval function $qty(prob::Problem, bodyindex::AbstractVector)
        return BodyArrayQuantity(1, bodyindex) do state::AbstractState
            panels = bodypanels(state)
            map(bodyindex) do i
                panels.perbody[i].$field
            end
        end
    end
end

function body_deformation(prob::Problem; bodyindex, kw...)
    return body_deformation(prob, bodyindex; kw...)
end
function body_deformation(prob::Problem, bodyindex::Integer; deriv::Integer=0)
    _check_deform_deriv(deriv)
    i_deform = _valid_deform_index(prob.bodies, bodyindex)
    return function (state::AbstractState)
        deform = deformation(state).perbody[i_deform]
        return (deform.χ, deform.ζ, deform.ζdot)[1 + deriv]
    end
end
function body_deformation(prob::Problem, bodyindex::AbstractVector; deriv::Integer=0)
    _check_deform_deriv(deriv)
    i_deform = map(i -> _valid_deform_index(prob.bodies, i), bodyindex)
    return BodyArrayQuantity(1, bodyindex) do state::AbstractState
        deform = deformation(state)
        return map(i_deform) do i
            d = deform.perbody[i]
            (d.χ, d.ζ, d.ζdot)[1 + deriv]
        end
    end
end
function _valid_deform_index(bodies::BodyGroup, i_body::Integer)
    if !haskey(bodies.index_to_deform, i_body)
        throw(ArgumentError("Body at index $i_body is not a deforming body"))
    end
    return bodies.index_to_deform[i_body]
end
function _check_deform_deriv(deriv::Int)
    if !(deriv in 0:2)
        throw(DomainError(deriv, "Derivative must be 0, 1, or 2"))
    end
    return nothing
end

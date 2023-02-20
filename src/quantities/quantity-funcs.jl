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
    @eval function $qty(prob::Problem; bodyindex=1:length(prob.bodies))
        return state -> map(bodyindex) do i
            bodypanels(state).perbody[i].$field
        end
    end
end

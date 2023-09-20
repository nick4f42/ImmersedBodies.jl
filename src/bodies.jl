function _init!(panels::PanelState, body::StaticBody)
    panels.xb .= body.panels.xb
    panels.ds .= body.panels.ds
    panels.ub .= 0
    true
end

_update!(::PanelState, ::StaticBody, t) = false

_init!(panels::PanelState, ::MovingRigidBody) = false

function _update!(panels::PanelState, body::MovingRigidBody, t)
    motion = body.motion(t)

    r0 = motion.pos
    v0 = motion.vel
    Ω = motion.angular_vel
    c = cos(motion.angle)
    s = sin(motion.angle)

    Rx = @SMatrix [c -s; s c]
    Rv = Ω * @SMatrix [-s -c; c -s]

    for i in axes(panels.xb, 1)
        rb = body.panels.xb[i, :]
        panels.xb[i, :] = r0 + Rx * rb
        panels.ub[i, :] = v0 + Rv * rb
    end

    true
end

function _init_body_motion!(state::State, reg::Regularization)
    bodies = state.prob.bodies
    for part in bodies.preset
        panels = view(state.panels, part.panel_range)
        if _init!(panels, part.body)
            _update!(reg, part.panel_range, panels.xb)
        end
    end
end

function _update_body_motion!(state::State, reg::Regularization)
    bodies = state.prob.bodies
    for part in bodies.preset
        panels = view(state.panels, part.panel_range)
        if _update!(panels, part.body, state.t)
            _update!(reg, part.panel_range, panels.xb)
        end
    end
end

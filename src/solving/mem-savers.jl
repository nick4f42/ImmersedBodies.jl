struct QuantityMemSaver{Q<:Quantity}
    qty::Q
    times::Vector{Float64}
    values::Vector{Any}
    QuantityMemSaver(qty::Q, times::Vector{Float64}) where {Q} = new{Q}(qty, times, Any[])
end

function update_saver(saver::QuantityMemSaver{<:ArrayQuantity}, state::AbstractState)
    push!(saver.values, saver.qty(state))
    return nothing
end

function mem_saver_values(saver::QuantityMemSaver{<:ArrayQuantity})
    arrays = [a for a in saver.values] # convert vector to proper type
    return ArrayValues(saver.times, arrays)
end

function update_saver(saver::QuantityMemSaver{<:GridQuantity}, state::AbstractState)
    array = saver.qty(state).array
    push!(saver.values, array)
    return nothing
end

function mem_saver_values(saver::QuantityMemSaver{<:GridQuantity})
    arrays = [a for a in saver.values] # convert vector to proper type
    return GridValues(saver.times, arrays, saver.qty.coords)
end

function update_saver(
    saver::QuantityMemSaver{<:MultiLevelGridQuantity}, state::AbstractState
)
    array = saver.qty(state).array
    push!(saver.values, array)
    return nothing
end

function mem_saver_values(saver::QuantityMemSaver{<:MultiLevelGridQuantity})
    arrays = [a for a in saver.values] # convert vector to proper type
    return MultiLevelGridValues(saver.times, arrays, saver.qty.coords)
end

function update_saver(saver::QuantityMemSaver{<:ConcatArrayQuantity}, state::AbstractState)
    arrays = saver.qty(state).arrays
    push!(saver.values, arrays)
    return nothing
end

function mem_saver_values(saver::QuantityMemSaver{<:ConcatArrayQuantity})
    arrays = [a for a in saver.values]
    return ConcatArrayValues(saver.times, arrays, saver.qty.dim)
end

function update_saver(saver::QuantityMemSaver{<:BodyArrayQuantity}, state::AbstractState)
    arrays = saver.qty(state).arrays
    push!(saver.values, arrays)
    return nothing
end

function mem_saver_values(saver::QuantityMemSaver{<:BodyArrayQuantity})
    arrays = [a for a in saver.values]
    return BodyArrayValues(saver.times, arrays, saver.qty.dim, saver.qty.bodies)
end

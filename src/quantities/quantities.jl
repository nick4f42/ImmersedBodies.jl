module Quantities

using ..ImmersedBodies
using ..ImmersedBodies.Bodies
import ..ImmersedBodies: timevalue

using HDF5
using EllipsisNotation

export Quantity, quantity
export GridQuantity, GridValue, GridValues, coordinates
export ArrayQuantity, ArrayValues
export MultiLevelGridQuantity, MultiLevelGridValue, MultiLevelGridValues
export ConcatArrayQuantity, ConcatArrayValue, ConcatArrayValues
export BodyArrayQuantity, BodyArrayValue, BodyArrayValues, bodyindices

export freestream_velocity, flow_velocity, streamfunction, vorticity
export body_point_pos, body_point_vel, body_traction, body_lengths
export body_deformation

"""
    Quantity

A function of [`AbstractState`](@ref).
"""
abstract type Quantity end

quantity(f::Quantity) = f
quantity(f) = ArrayQuantity(f)

include("quantity-types.jl")
include("quantity-funcs.jl")

end # module Quantities

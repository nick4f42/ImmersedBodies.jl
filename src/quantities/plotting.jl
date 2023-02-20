module Recipes

using ..Quantities
using Plots

@recipe function f(val::GridValue{<:Number,2})
    tick_direction --> :out
    aspect_ratio --> :equal
    grid --> false

    seriestype --> :heatmap

    xs, ys = val.coords
    xlims --> extrema(xs)
    ylims --> extrema(ys)

    return (val.coords..., transpose(val.array))
end

@recipe function f(val::MultiLevelGridValue{2})
    tick_direction --> :out
    aspect_ratio --> :equal
    grid --> false

    seriestype --> :heatmap

    xs, ys = last(val.coords)
    xlims --> extrema(xs)
    ylims --> extrema(ys)

    # TODO: Only include sublevels that are inside the given xlims, ylims
    lastaxis = axes(val, ndims(val))
    for i in reverse(lastaxis)
        @series (val.coords[i]..., transpose(val.array[:, :, i]))
    end
end

end # module Recipes

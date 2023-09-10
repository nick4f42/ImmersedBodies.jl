# Getting Started

## Installation

```julia
import Pkg; Pkg.add(url="https://github.com/NUFgroup/ImmersedBodies.jl")

```

## Example Problem

First, load the `ImmersedBodies` package.

```julia
using ImmersedBodies
```

Next, we define the fluid for the problem. The fluid domain is `nlevel`
Cartesian grids that progressively double in scale. The smallest grid is at
$-1<x<1, -1.5<y<1.5$. The fluid is specified with a Reynolds number of 200 and a
freestream velocity of 1 along $x$. The grid step can be manually set by adding
an `h` keyword argument to `fluid_grid`.

```julia
fluid = Fluid(
    grid = fluid_grid(xlims = (-1.0, 2.0), ylims = (-1.5, 1.5), nlevel = 5),
    Re = 200.0,
    freestream_vel = t -> (1.0, 0.0),
)
```

Next, we define the bodies. In this case, 2 cylinders with the same radius at
different heights. Each body is a [`StaticBody`](@ref), meaning the body is
stationary in the fluid grid. Each body is given a [`Panels`](@ref) argument
that specifies the points on the body. You can manually specify points to use,
or use the `ImmersedBodies.Curves` module to automatically discretize a curve
that will work with the fluid grid.

```julia
bodies = Bodies(
    cyl1=StaticBody(
        Panels(fluid.grid, Curves.Circle(r=0.5, center=(0.0, -0.75)))
    ),
    cyl2=StaticBody(
        Panels(fluid.grid, Curves.Circle(r=0.5, center=(0.0, +0.75)))
    ),
)
```

Next, the time-stepping scheme. Here we automatically compute the time step to
hit a target `cfl` number based on a given maximum fluid velocity `Umax`.

```julia
scheme = default_scheme(fluid.grid; Umax=2.0, cfl=0.2)
```

Finally, we combine the `fluid`, `bodies`, and `scheme` into the problem.

```julia
prob = Problem(; fluid, bodies, scheme)
```

We now solve the problem from time `0` to `t` and specify values to save to an
HDF5 file. There are interfaces to the HDF5 file format for many programming
languages, so you can analyze the data in whichever language you prefer.

```julia
solve(
    prob,
    t=15.0,
    save=SaveHDF5(
        "soln.h5",
        "problem" => prob,
        "fluid" => fluid_group(
            prob,
            TimestepRange(; step=200),
            (x_velocity!, y_velocity!, vorticity!)
        ),
        "bodies" => body_group(
            prob,
            TimestepRange(; step=10),
            (boundary_force!,)
        ),
        "totals" => body_group(
            prob,
            TimestepRange(; step=1),
            (boundary_total_force!,)
        ),
    ),
)
```

Here is an example script for plotting the results in Julia with
[Makie](https://docs.makie.org/stable/).

```julia
using HDF5
using GLMakie

file = h5open("soln.h5")

# Load the positions of the two cylinders
bodypos = NamedTuple(
    name => read(file["problem/bodies/$name/positions"]) for name in (:cyl1, :cyl2)
)

# Reference the vorticity dataset in the file. This doesn't load the data into memory until
# we index it.
vorticity_dset = file["fluid/vorticity"]

# The coords attribute stores the coordinates that correspond to the vorticity array. The
# fields are `start`, `length`, and `step` to work directly with the `range` function.
# `vort_coords[:, level]` are the coordinates on the `level`th subdomain.
vort_coords = map(r -> range(; r...), read_attribute(vorticity_dset, "coords"))

# Set up the Makie plot
fig = Figure(resolution = (800, 400))
ax = Axis(fig[1, 1], aspect = DataAspect(), limits = ((-1.5, 8.5), (-2.5, 2.5)))

# Plot contours of the vorticity in reverse order so that the smaller domains are plotted on
# top
levels = reverse(axes(vorticity_dset, 3))
contours = map(levels) do lev
    x, y = vort_coords[:, lev]
    @views contourf!(
        ax, x, y,
        vorticity_dset[:, :, lev, end],
        colormap = :RdBu,
        levels = range(-5, 5, 12),
        extendlow = :auto,
        extendhigh = :auto,
    )
end

Colorbar(fig[1, 2], contours[1])

# Plot the cylinder bodies
for pos in bodypos
    poly!(ax, Point2.(eachrow(pos)), color = :gray)
end

# Create a gif by updating the vorticity plot
times = axes(vorticity_dset, 4)
record(fig, "vorticity.gif", times; framerate = 20) do i
    ω = vorticity_dset[:, :, :, i]
    for (level, contour) in zip(levels, contours)
        contour[3] = @view ω[:, :, level]
    end
end
```

# NACA

```julia
using ImmersedBodies
using ImmersedBodies.Curves

fluid = Fluid(
    grid = fluid_grid(xlims=(-1.0, 2.0), ylims=(-0.5, 0.5), nlevel=3),
    Re=400.0,
    freestream_vel=t -> (1.0, 0.0),
)

α = deg2rad(20)
airfoil = naca"0012" |> translate((-0.5, 0.0)) |> rotate(-α)
bodies = Bodies(
    airfoil=StaticBody(Panels(fluid.grid, airfoil)),
)

scheme = default_scheme(fluid.grid; Umax=2.0, cfl=0.2)

prob = Problem(; fluid, bodies, scheme)

solve(
    prob,
    t=10.0,
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

```julia
using HDF5
using GLMakie

file = h5open("soln.h5")

bodypos = read(file["problem/bodies/airfoil/positions"])
vorticity_dset = file["fluid/vorticity"]
vort_coords = map(r -> range(; r...), read_attribute(vorticity_dset, "coords"))

fig = Figure(resolution = (800, 400))

ax = Axis(fig[1, 1], aspect = DataAspect(), limits = ((-1.5, 6.5), (-2, 2)))

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

poly!(ax, Point2.(eachrow(bodypos)), color = :gray)

times = axes(vorticity_dset, 4)
record(fig, "vorticity.gif", times; framerate = 20) do i
    ω = vorticity_dset[:, :, :, i]
    for (level, contour) in zip(levels, contours)
        contour[3] = @view ω[:, :, level]
    end
end
```

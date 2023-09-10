```@meta
CurrentModule = ImmersedBodies
```

# Problems

## Fluid

```@docs
gridstep
gridsize
extents
coords
fluid_grid
CartesianGrid
MultiDomainGrid
MultiDomainExtents
subdomain
GridMotion
StaticGrid
MovingGrid
GridVelocity
Fluid
```

## Bodies

```@docs
n_panels
Panels
PanelState
AbstractBody
PresetBody
Bodies
PanelSection
section_body
panel_range
bodypanels
StaticBody
RigidBodyTransform
MovingRigidBody
```

## Time-stepping Scheme

```@docs
timestep
AbstractScheme
default_scheme
CNAB
```

## Problem

```@docs
Problem
```

## State

```@docs
State
x_velocity
y_velocity
vorticity
boundary_pos
boundary_len
boundary_vel
boundary_force
boundary_total_force
```

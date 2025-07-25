# API

The API of Agents.jl is defined on top of the fundamental structures [`AgentBasedModel`](@ref), [Space](@ref available_spaces), [`AbstractAgent`](@ref) which are described in the [Tutorial](@ref) page.
In this page we list the remaining API functions, which constitute the bulk of Agents.jl functionality.

## [AgentBasedModel](@id ABM_Implementations)

- [`AgentBasedModel`](@ref)
- [`StandardABM`](@ref)
- [`EventQueueABM`](@ref)

```@docs
AgentBasedModel
step!(::AgentBasedModel, args...)
```

### Discrete time models

```@docs
StandardABM
```

### Continuous time models

```@docs
EventQueueABM
AgentEvent
add_event!
```

## Agent types

```@docs
@agent
@multiagent
AbstractAgent
SoAType
```

### Minimal agent types

The [`@agent`](@ref) macro can be used to define new agent types from the minimal agent types that are listed below:

```@docs
NoSpaceAgent
GraphAgent
GridAgent
ContinuousAgent
OSMAgent
```

## Agent/model retrieval and access

```@docs
getindex(::ABM, ::Integer)
getproperty(::ABM, ::Symbol)
random_id
random_agent
nagents
allagents
allids
hasid
abmproperties
abmrng
abmscheduler
abmspace
abmtime
abmevents
```

## [Available spaces](@id available_spaces)

Here we list the spaces that are available "out of the box" from Agents.jl. To create your own, see the developer documentation on [creating a new space type](@ref make_new_space).

### Discrete spaces

```@docs
GraphSpace
GridSpace
GridSpaceSingle
```

Here is a specification of how the metrics look like:

```@example MAIN
include("distances_example_plot.jl") # hide
```

### Continuous spaces

```@docs
ContinuousSpace
OpenStreetMapSpace
```

## Adding agents
```@docs
add_agent!
add_agent_own_pos!
replicate!
random_position
```

## Moving agents
```@docs
move_agent!
walk!
randomwalk!
get_direction
```

### Movement with paths
For [`OpenStreetMapSpace`](@ref), and [`GridSpace`](@ref)/[`ContinuousSpace`](@ref) using [`Pathfinding`](@ref), a special
movement method is available.

```@docs
plan_route!
plan_best_route!
move_along_route!
is_stationary
```

## Removing agents
```@docs
remove_agent!
remove_all!
sample!
```

## Space utility functions
```@docs
normalize_position
spacesize
```

## [`DiscreteSpace` exclusives](@id DiscreteSpace_exclusives)
```@docs
positions
npositions
ids_in_position
id_in_position
agents_in_position
random_id_in_position
random_agent_in_position
fill_space!
has_empty_positions
empty_positions
empty_nearby_positions
random_empty
add_agent_single!
move_agent_single!
swap_agents!
isempty(::Int, ::ABM)
```

## `GraphSpace` exclusives
```@docs
add_edge!
rem_edge!
add_vertex!
rem_vertex!
```

## [`ContinuousSpace` exclusives](@id ContinuosSpace_exclusives)
```@docs
nearest_neighbor
get_spatial_property
get_spatial_index
interacting_pairs
elastic_collision!
euclidean_distance
manhattan_distance
```

## `OpenStreetMapSpace` exclusives
```@docs
OSM
OSM.lonlat
OSM.nearest_node
OSM.nearest_road
OSM.random_road_position
OSM.plan_random_route!
OSM.road_length
OSM.route_length
OSM.same_position
OSM.same_road
OSM.test_map
OSM.download_osm_network
```

## Nearby Agents
```@docs
nearby_ids
nearby_agents
nearby_positions
random_nearby_id
random_nearby_agent
random_nearby_position
```

## A note on iteration

Most iteration in Agents.jl is **dynamic** and **lazy**, when possible, for performance reasons.

**Dynamic** means that when iterating over the result of e.g. the [`ids_in_position`](@ref) function, the iterator will be affected by actions that would alter its contents.
Specifically, imagine the scenario
```@example docs
using Agents
# We don't need to make a new agent type here,
# we use the minimal agent for 4-dimensional grid spaces
model = StandardABM(GridAgent{4}, GridSpace((5, 5, 5, 5)))
add_agent!((1, 1, 1, 1), model)
add_agent!((1, 1, 1, 1), model)
add_agent!((2, 1, 1, 1), model)
for id in ids_in_position((1, 1, 1, 1), model)
    remove_agent!(id, model)
end
collect(allids(model))
```
You will notice that only 1 agent was removed. This is simply because the final state of the iteration of `ids_in_position` was reached unnaturally, because the length of its output was reduced by 1 _during_ iteration.
To avoid problems like these, you need to `collect` the iterator to have a non dynamic version.

**Lazy** means that when possible the outputs of the iteration are not collected and instead are generated on the fly.
A good example to illustrate this is [`nearby_ids`](@ref), where doing something like
```julia
a = random_agent(model)
sort!(nearby_ids(random_agent(model), model))
```
leads to error, since you cannot `sort!` the returned iterator. This can be easily solved by adding a `collect` in between:
```@example docs
a = random_agent(model)
sort!(collect(nearby_agents(a, model)))
```

## Higher-order interactions

There may be times when pair-wise, triplet-wise or higher interactions need to be
accounted for across most or all of the model's agent population. The following methods
provide an interface for such calculation.

These methods follow the conventions outlined above in [A note on iteration](@ref).

```@docs
iter_agent_groups
map_agent_groups
index_mapped_groups
```

## Data collection and analysis
```@docs
run!
ensemblerun!
paramscan
```


### Manual data collection

The central simulation function is [`run!`](@ref).
Here are some functions that aid in making custom data collection loops, instead of using the `run!` function:

```@docs
init_agent_dataframe
collect_agent_data!
init_model_dataframe
collect_model_data!
dataname
```

For example, the core loop of `run!` is just
```julia
df_agent = init_agent_dataframe(model, adata)
df_model = init_model_dataframe(model, mdata)

t0 = abmtime(model)
t = t0
while until(t, t0, n, model)
  if should_we_collect(t, model, when)
      collect_agent_data!(df_agent, model, adata)
  end
  if should_we_collect(t, model, when_model)
      collect_model_data!(df_model, model, mdata)
  end
  step!(model, 1)
  t = abmtime(model)
end
return df_agent, df_model
```
(here `until` and `should_we_collect` are internal functions)

## [Schedulers](@id Schedulers)

```@docs
Schedulers
schedule(::ABM)
```

### Predefined schedulers

Some useful schedulers are available below as part of the Agents.jl API:

```@docs
Schedulers.fastest
Schedulers.ByID
Schedulers.Randomly
Schedulers.Partially
Schedulers.ByProperty
Schedulers.ByType
Schedulers.ByKind
```

### [Advanced scheduling](@id advanced_scheduling)
You can use [Function-like objects](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects) to make your scheduling possible of arbitrary events.
For example, imagine that after the `n`-th step of your simulation you want to fundamentally change the order of agents. To achieve this you can define
```julia
mutable struct MyScheduler
    n::Int # step number
    w::Float64
end
```
and then define a calling method for it like so
```julia
function (ms::MyScheduler)(model::ABM)
    ms.n += 1 # increment internal counter by 1 each time its called
              # be careful to use a *new* instance of this scheduler when plotting!
    if ms.n < 10
        return allids(model) # order doesn't matter in this case
    else
        ids = collect(allids(model))
        # filter all ids whose agents have `w` less than some amount
        filter!(id -> model[id].w < ms.w, ids)
        return ids
    end
end
```
and pass it to e.g. `step!` by initializing it
```julia
ms = MyScheduler(100, 0.5)
step!(model, agentstep, modelstep, 100; scheduler = ms)
```


### How to use `Distributed`
To use the `parallel=true` option of [`ensemblerun!`](@ref) you need to load `Agents` and define your fundamental types at all processors. See the [Performance Tips](@ref) page for parallelization.

## Path-finding
```@docs
Pathfinding
Pathfinding.AStar
Pathfinding.penaltymap
Pathfinding.nearby_walkable
Pathfinding.random_walkable
```

### Pathfinding Metrics
```@docs
Pathfinding.DirectDistance
Pathfinding.MaxDistance
Pathfinding.PenaltyMap
```

Building a custom metric is straightforward, if the provided ones do not suit your purpose.
See the [Developer Docs](@ref) for details.

## Save, Load, Checkpoints
There may be scenarios where interacting with data in the form of files is necessary. The following
functions provide an interface to save/load data to/from files.
```@docs
AgentsIO.save_checkpoint
AgentsIO.load_checkpoint
AgentsIO.populate_from_csv!
AgentsIO.dump_to_csv
```

It is also possible to write data to file at predefined intervals while running your model, instead of storing it in memory:
```@docs
offline_run!
```

In case you require custom serialization for model properties, refer to the [Developer Docs](@ref)
for details.

## Visualizations

```@docs
abmplot
abmplot!
abmexploration
abmvideo
ABMObservable
add_interaction!
translate_polygon
scale_polygon
rotate_polygon
```

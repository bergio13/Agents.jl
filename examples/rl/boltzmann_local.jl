using Agents, Random, CairoMakie
using POMDPs, Crux, Flux, Distributions

## 1. Agent Definition 
@agent struct BoltzmannAgent(GridAgent{2})
    wealth::Int
end

## 2. Gini Coefficient Calculation Function
function gini(wealths::Vector{Int})
    n = length(wealths)
    if n <= 1
        return 0.0
    end
    sorted_wealths = sort(wealths)
    sum_wi = sum(sorted_wealths)
    if sum_wi == 0
        return 0.0
    end
    numerator = sum((2i - n - 1) * w for (i, w) in enumerate(sorted_wealths))
    denominator = n * sum_wi
    return numerator / denominator
end

## 3. Agent Step Function
function boltz_step!(agent::BoltzmannAgent, model::ABM, action::Int)
    # Action definitions:
    # 1: Stay
    # 2: North (+y)
    # 3: South (-y)
    # 4: East  (+x)
    # 5: West  (-x)

    current_x, current_y = agent.pos
    width, height = getfield(model, :space).extent

    dx, dy = 0, 0
    if action == 2
        dy = 1
    elseif action == 3
        dy = -1
    elseif action == 4
        dx = 1
    elseif action == 5
        dx = -1
    end

    # Apply periodic boundary wrapping
    new_x = mod1(current_x + dx, width)
    new_y = mod1(current_y + dy, height)
    target_pos = (new_x, new_y)

    #println("Agent $(agent.id) action: $action, target position: $target_pos")
    move_agent!(agent, target_pos, model)

    # Wealth exchange
    others = [a for a in agents_in_position(agent.pos, model) if a.id != agent.id]
    if !isempty(others)
        other = rand(others)
        if other.wealth > agent.wealth && other.wealth > 0
            # Transfer wealth from other to agent
            agent.wealth += 1
            other.wealth -= 1
        end
    end
end

## 4. Represent the state of the environment
struct BoltzmannState
    agent_wealths::Vector{Int}
    agent_positions::Vector{NTuple{2,Int}}
    step_count::Int
    current_gini::Float64
end

# Helper to convert model to BoltzmannState
function model_to_state(model::ABM, step_count::Int)
    wealths = [a.wealth for a in allagents(model)]
    positions = [a.pos for a in allagents(model)]
    current_gini = gini(wealths)
    return BoltzmannState(wealths, positions, step_count, current_gini)
end

# Convert state to vector
function state_to_vector(s::BoltzmannState)::Vector{Float32}
    n_agents = length(s.agent_wealths)
    state_vec = Vector{Float32}(undef, n_agents * 3 + 2)

    # Interleave agent data: wealth1, x1, y1, wealth2, x2, y2, ...
    for i in 1:n_agents
        idx = 3 * (i - 1) + 1
        state_vec[idx] = Float32(s.agent_wealths[i])
        state_vec[idx+1] = Float32(s.agent_positions[i][1])  # x
        state_vec[idx+2] = Float32(s.agent_positions[i][2])  # y
    end

    # Add global info at the end
    state_vec[3*n_agents+1] = Float32(s.step_count)
    state_vec[3*n_agents+2] = Float32(s.current_gini)

    return state_vec
end

## 5. Custom MDP
mutable struct BoltzmannEnv <: POMDP{BoltzmannState,Int,Vector{Int}}
    abm_model::ABM
    num_agents::Int
    dims::Tuple{Int,Int}
    initial_wealth::Int
    max_steps::Int
    gini_threshold::Float64
    observation_radius::Int
    current_agent_id::Int
    rng::AbstractRNG
end

# Constructor for BoltzmannEnv 

function BoltzmannEnv(; num_agents=50, dims=(10, 10), seed=123, initial_wealth=1, max_steps=200, gini_threshold=0.2, observation_radius=1, current_agent_id=1)
    rng = MersenneTwister(seed)
    env = BoltzmannEnv(
        boltzmann_money_model_rl_init(num_agents=num_agents, dims=dims, seed=seed, initial_wealth=initial_wealth),
        num_agents,
        dims,
        initial_wealth,
        max_steps,
        gini_threshold,
        observation_radius,
        current_agent_id,
        rng
    )
    return env
end

# A function to initialize the ABM specifically for the RL environment
function boltzmann_money_model_rl_init(; num_agents=100, dims=(10, 10), seed=1234, initial_wealth=1)
    space = GridSpace(dims; periodic=true)
    rng = MersenneTwister(seed)
    properties = Dict{Symbol,Any}(
        :gini_coefficient => 0.0,
        :step_count => 0
    )
    model = StandardABM(BoltzmannAgent, space;
        (model_step!)=boltz_model_step!,
        rng,
        properties=properties
    )

    for _ in 1:num_agents
        add_agent_single!(BoltzmannAgent, model, rand(1:initial_wealth))
    end
    wealths = [a.wealth for a in allagents(model)]
    model.gini_coefficient = gini(wealths)
    return model
end

# Model Step Function 
function boltz_model_step!(model::ABM)
    wealths = [agent.wealth for agent in allagents(model)]
    model.gini_coefficient = gini(wealths)
    model.step_count += 1
end

## 4. Local Observation Structure
struct LocalObservation
    agent_id::Int
    own_wealth::Int
    own_position::NTuple{2,Int}
    occupancy_grid::Matrix{Bool}  # Binary grid showing occupied cells
end

# Helper to get neighborhood observation for a specific agent
function get_local_observation(model::ABM, agent_id::Int, observation_radius::Int=2)
    target_agent = model[agent_id]
    agent_pos = target_agent.pos

    # Create occupancy grid
    grid_size = 2 * observation_radius + 1
    occupancy_grid = zeros(Bool, grid_size, grid_size)

    # Get world dimensions for periodic boundary handling
    width, height = getfield(model, :space).extent

    # Fill occupancy grid
    for other_agent in allagents(model)
        if other_agent.id == agent_id
            continue  # Don't include self in occupancy grid
        end

        # Calculate relative position with periodic boundaries
        dx_raw = other_agent.pos[1] - agent_pos[1]
        dy_raw = other_agent.pos[2] - agent_pos[2]

        # Handle periodic boundaries
        dx = dx_raw
        if abs(dx_raw) > width / 2
            dx = dx_raw - sign(dx_raw) * width
        end

        dy = dy_raw
        if abs(dy_raw) > height / 2
            dy = dy_raw - sign(dy_raw) * height
        end

        # Check if within observation radius
        if abs(dx) <= observation_radius && abs(dy) <= observation_radius
            # Convert to grid coordinates (center is at observation_radius + 1)
            grid_x = dx + observation_radius + 1
            grid_y = dy + observation_radius + 1
            occupancy_grid[grid_x, grid_y] = true
        end
    end

    return LocalObservation(
        agent_id,
        target_agent.wealth,
        agent_pos,
        occupancy_grid
    )
end

# Convert local observation to vector
function observation_to_vector(obs::LocalObservation, observation_radius::Int)::Vector{Float32}
    # Fixed size observation vector
    grid_size = 2 * observation_radius + 1
    total_grid_cells = grid_size * grid_size

    # Format: [own_wealth, flattened_occupancy_grid...]
    obs_size = 1 + total_grid_cells
    obs_vec = zeros(Float32, obs_size)

    # Own wealth
    obs_vec[1] = Float32(obs.own_wealth)

    # Flatten occupancy grid and add to observation
    flattened_grid = reshape(obs.occupancy_grid, total_grid_cells)
    obs_vec[2:1+total_grid_cells] = Float32.(flattened_grid)

    return obs_vec
end

## 6. Implement POMDPs.jl Interface for BoltzmannEnv
const NUM_INDIVIDUAL_ACTIONS = 5 # Stay, N, S, E, W

function POMDPs.actions(env::BoltzmannEnv)
    return Crux.DiscreteSpace(NUM_INDIVIDUAL_ACTIONS)  # Actions: 1=Stay, 2=North, 3=South, 4=East, 5=West
end


function POMDPs.observations(env::BoltzmannEnv)
    # Return a continuous space for observations
    observation_radius = env.observation_radius
    grid_size = 2 * observation_radius + 1
    total_grid_cells = grid_size * grid_size
    return Crux.ContinuousSpace((1 + total_grid_cells,), Float32)  # Own wealth + occupancy grid
end

function POMDPs.observation(env::BoltzmannEnv, s::Vector{Float32})
    # Get local observation for the current agent
    local_obs = get_local_observation(env.abm_model, env.current_agent_id, env.observation_radius)

    # Convert local observation to vector
    obs_vec = observation_to_vector(local_obs, env.observation_radius)

    # Return as Dirac distribution
    return obs_vec
end

# Define the initial state of the MDP
function POMDPs.initialstate(env::BoltzmannEnv)
    env.abm_model = boltzmann_money_model_rl_init(
        num_agents=env.num_agents,
        dims=env.dims,
        seed=1234,
        initial_wealth=env.initial_wealth
    )
    return Dirac(state_to_vector(model_to_state(env.abm_model, 0)))
end

function POMDPs.initialobs(env::BoltzmannEnv, initial_state::Vector{Float32})
    # Convert to observation
    obs = POMDPs.observation(env, initial_state)
    return Dirac(obs)
end

# Define the transition function
function POMDPs.transition(env::BoltzmannEnv, s, a::Int)
    agent = env.abm_model[env.current_agent_id]
    boltz_step!(agent, env.abm_model, a)

    # Update current agent (round-robin)
    env.current_agent_id = (env.current_agent_id % env.num_agents) + 1

    # If we've cycled through all agents, run model step
    if env.current_agent_id == 1
        boltz_model_step!(env.abm_model)
    end

    # Return the new state
    next_state = state_to_vector(model_to_state(env.abm_model, env.abm_model.step_count))
    return next_state
end

function POMDPs.gen(env::BoltzmannEnv, state, action::Int, rng::AbstractRNG)
    next_state = POMDPs.transition(env, state, action)
    observation = POMDPs.observation(env, next_state)
    r = POMDPs.reward(env, state, action, next_state)
    return (sp=next_state, o=observation, r=r)
end

# Define the reward function 
function POMDPs.reward(env::BoltzmannEnv, s, a::Int, sp)
    reward = 0.0
    new_gini = sp[end] # Last element is the Gini coefficient
    prev_gini = s[end] # Last element of the previous state

    if new_gini < prev_gini
        reward = (prev_gini - new_gini) * 20.0
    else
        reward = -0.05
    end

    if POMDPs.isterminal(env, sp)
        if sp[end] < env.gini_threshold
            reward += 50.0 / (sp[end-1] + 1)
        elseif sp[end-1] >= env.max_steps
            reward -= 1
        end
    end
    return reward
end

# Define terminal states
function POMDPs.isterminal(env::BoltzmannEnv, s)
    s[end] < env.gini_threshold || s[end-1] >= env.max_steps
end


Crux.state_space(env::BoltzmannEnv) = Crux.ContinuousSpace((env.num_agents * 3 + 2,)) # 3 values per agent: wealth, x, y + gini
POMDPs.discount(env::BoltzmannEnv) = 0.99 # Discount factor


# Setup the environment
N_AGENTS = 3
env_mdp = BoltzmannEnv(num_agents=N_AGENTS, dims=(15, 15), initial_wealth=10, max_steps=50, gini_threshold=0.1)

POMDPs.actions(env_mdp).vals
length(POMDPs.actions(env_mdp).vals)

S = Crux.state_space(env_mdp)
Crux.dim(S)[1]
output_size = length(POMDPs.actions(env_mdp).vals)
as = POMDPs.actions(env_mdp).vals
O = observations(env_mdp)


QS() = DiscreteNetwork(
    Chain(
        Dense(Crux.dim(S)[1], 64, relu),
        Dense(64, 64, relu),
        Dense(64, output_size)
    ), as)

solver = DQN(π=QS(), S=Crux.state_space(env_mdp), N=200_000, buffer_size=10000, buffer_init=1000)
policy = solve(solver, env_mdp)
plot_learning(solver)


V() = ContinuousNetwork(Chain(Dense(Crux.dim(O)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
B() = DiscreteNetwork(Chain(Dense(Crux.dim(O)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)

𝒮_ppo = PPO(π=ActorCritic(B(), V()), S=O, N=100_000, ΔN=200, log=(period=1000,))
@time π_ppo = solve(𝒮_ppo, env_mdp)
plot_learning(𝒮_ppo)

function run_policy_in_abm(π, env::BoltzmannEnv; max_steps=50)
    # Initialize the ABM from the env
    abm = boltzmann_money_model_rl_init(
        num_agents=env.num_agents,
        dims=env.dims,
        seed=1234,
        initial_wealth=env.initial_wealth
    )

    for step in 1:max_steps
        state_vec = state_to_vector(model_to_state(abm, step))
        joint_action_idx = action(π, state_vec)

        # Convert action index to joint action
        joint_action = int_to_joint_action(joint_action_idx[1], env.num_agents)

        agents_by_id = sort(collect(allagents(abm)), by=a -> a.id)
        for (i, agent) in enumerate(agents_by_id)
            boltz_step!(agent, abm, joint_action[i])
        end

        boltz_model_step!(abm)

        println("Step $step | Gini: ", abm.gini_coefficient)

        if abm.gini_coefficient < env.gini_threshold
            println("Reached Gini threshold.")
            break
        end
    end

    return abm
end

# Run trained policy in a fresh ABM instance
final_abm = run_policy_in_abm(π_ppo, env_mdp; max_steps=50)

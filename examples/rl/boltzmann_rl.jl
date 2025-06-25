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
    width, height = 5, 5

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
    if agent.wealth > 0
        others = [a for a in agents_in_position(agent.pos, model) if a.id != agent.id]
        if !isempty(others)
            other = rand(others)
            agent.wealth -= 1
            other.wealth += 1
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
mutable struct BoltzmannEnv <: MDP{BoltzmannState,Vector{Int}}
    abm_model::ABM
    num_agents::Int
    dims::Tuple{Int,Int}
    initial_wealth::Int
    max_steps::Int
    gini_threshold::Float64
    rng::AbstractRNG
end

# Constructor for BoltzmannEnv 
function BoltzmannEnv(; num_agents=50, dims=(10, 10), seed=123, initial_wealth=1, max_steps=200, gini_threshold=0.1)
    rng = MersenneTwister(seed)
    env = BoltzmannEnv(
        boltzmann_money_model_rl_init(num_agents=num_agents, dims=dims, seed=seed, initial_wealth=initial_wealth),
        num_agents,
        dims,
        initial_wealth,
        max_steps,
        gini_threshold,
        rng
    )
    return env
end

# A function to initialize the ABM specifically for the RL environment
function boltzmann_money_model_rl_init(; num_agents=100, dims=(10, 10), seed=123, initial_wealth=1)
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
        add_agent_single!(BoltzmannAgent, model, initial_wealth)
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

## 6. Implement POMDPs.jl Interface for BoltzmannEnv

# Define the observation space size for Crux (n_agents * 3: wealth, x, y)
#function POMDPs.observations(env::BoltzmannEnv, s::BoltzmannState)
#    obs_vec = vcat(s.agent_wealths, collect(Iterators.flatten(s.agent_positions))...)
#    return obs_vec
#end

const NUM_INDIVIDUAL_ACTIONS = 5 # Stay, N, S, E, W

function POMDPs.actions(env::BoltzmannEnv)
    #individual_actions = 1:NUM_INDIVIDUAL_ACTIONS
    #return collect(Iterators.product(fill(individual_actions, env.num_agents)...))
    return 1:(NUM_INDIVIDUAL_ACTIONS^env.num_agents)
end

# Define the initial state of the MDP
function POMDPs.initialstate(env::BoltzmannEnv)
    env.abm_model = boltzmann_money_model_rl_init(
        num_agents=env.num_agents,
        dims=env.dims,
        seed=rand(env.rng, Int),
        initial_wealth=env.initial_wealth
    )
    return Dirac(state_to_vector(model_to_state(env.abm_model, 0)))
end

# Define the transition function
function POMDPs.transition(env::BoltzmannEnv, s, a::Vector{Int})
    agents_by_id = sort(collect(allagents(env.abm_model)), by=x -> x.id) # Ensure consistent order

    for (i, agent) in enumerate(agents_by_id)
        # Ensure the action index `a[i]` is valid.
        boltz_step!(agent, env.abm_model, a[i])
    end

    # Run the model step (updates Gini coefficient, increments step_count)
    boltz_model_step!(env.abm_model)

    # Return the new state
    next_state = state_to_vector(model_to_state(env.abm_model, env.abm_model.step_count))
    return next_state
end

# Define the reward function 
function POMDPs.reward(env::BoltzmannEnv, s, a::Vector{Int}, sp)
    reward = 0.0
    new_gini = sp[end] # Last element is the Gini coefficient
    prev_gini = s[end] # Last element of the previous state

    if new_gini < prev_gini
        reward = (prev_gini - new_gini) * 20.0
    else
        reward = -1.0
    end

    if POMDPs.isterminal(env, sp)
        if sp[end] < env.gini_threshold
            reward += 50.0 / (sp[end-1] + 1e-6)
        elseif sp[end-1] >= env.max_steps
            reward -= 1.0
        end
    end
    return reward
end

# Define terminal states
function POMDPs.isterminal(env::BoltzmannEnv, s)
    s[end] < env.gini_threshold || s[end-1] >= env.max_steps
end

# Helper functions to convert between single integer (action) and joint action vector.
function int_to_joint_action(action_idx::Int, num_agents::Int)
    action_idx -= 1 # Convert to 0-indexed
    joint_action = Vector{Int}(undef, num_agents)
    for i in 1:num_agents
        joint_action[i] = (action_idx % NUM_INDIVIDUAL_ACTIONS) + 1 # Convert back to 1-indexed
        action_idx = floor(Int, action_idx / NUM_INDIVIDUAL_ACTIONS)
    end
    return reverse(joint_action)
end

function joint_action_to_int(joint_action::Vector{Int})
    action_idx = 0
    num_agents = length(joint_action)
    for i in 1:num_agents
        action_idx += (joint_action[i] - 1) * (NUM_INDIVIDUAL_ACTIONS^(num_agents - i))
    end
    return action_idx + 1 # Convert to 1-indexed
end


function POMDPs.gen(env::BoltzmannEnv, state, action_idx::Int, rng::AbstractRNG)
    #println(action_idx)
    joint_action = int_to_joint_action(action_idx, env.num_agents)
    #println("Joint action: ", joint_action)
    next_state = POMDPs.transition(env, state, joint_action)
    r = POMDPs.reward(env, state, joint_action, next_state)
    return (sp=next_state, r=r)
end

Crux.state_space(env::BoltzmannEnv) = Crux.ContinuousSpace((env.num_agents * 3 + 2,)) # 3 values per agent: wealth, x, y + gini
POMDPs.discount(env::BoltzmannEnv) = 0.99 # Discount factor


# Setup the environment
N_AGENTS = 6
env_mdp = BoltzmannEnv(num_agents=N_AGENTS, dims=(5, 5), initial_wealth=1, max_steps=100)

POMDPs.actions(env_mdp)
length(POMDPs.actions(env_mdp))
rand(POMDPs.actions(env_mdp))

S = Crux.state_space(env_mdp)
Crux.dim(S)[1]
output_size = length(POMDPs.actions(env_mdp))
as = [POMDPs.actions(env_mdp)...]



QS() = DiscreteNetwork(
    Chain(
        Dense(Crux.dim(S)[1], 64, relu),
        Dense(64, 64, relu),
        Dense(64, output_size)
    ), as)

solver = DQN(π=QS(), S=Crux.state_space(env_mdp), N=100_000, buffer_size=1000, buffer_init=1000)
policy = solve(solver, env_mdp)
plot_learning(solver)


V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
B() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)

𝒮_ppo = PPO(π=ActorCritic(B(), V()), S=Crux.state_space(env_mdp), N=100_000, ΔN=200, log=(period=1000,))
@time π_ppo = solve(𝒮_ppo, env_mdp)
plot_learning(𝒮_ppo)
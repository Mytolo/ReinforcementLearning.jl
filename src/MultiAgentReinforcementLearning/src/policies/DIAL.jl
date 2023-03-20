using Distributions
using Flux
import Base.+

function +(a::Dict{T, V}, b::Dict{T, V}) where {T, V <: Real}
    Dict{T, V}(k => a[k] + b[k] for (k, _) in a)
end

function +(a::Dict{T, V}, b::Dict{T, W}) where {T, V <: Vector{<: Real}, W <: Vector{<: Real}}
    Dict{T, V}(k => a[k] .+ b[k] for (k, _) in a)
end

export DIAL, MultiActionMessage

Base.@kwdef mutable struct DIAL <: AbstractPolicy
    c_nets::Dict{String, <:Approximator}
    emb::Dict{String, <:Approximator}
    rnn_idx::Vector{Int}
    agents::Array{String}
    num_episode_steps::Integer
    explorer::AbstractExplorer
    τ::Dict{String, Trajectory}
    σ::Float32 = 0.2
    n_message_bits::Int64
    training::Bool = true
end

"""
  Defining Struct for encapsulating actual actions and messages taken by algorihms incooperating communication of agents
"""
struct MultiActionMessage{Ta, Tm}
    action::Ta
    message::Tm
end


function dru(π::DIAL, message)
    m = Flux.sigmoid_fast(rand(Distributions.MultivariateNormal(message, π.σ .* ones(size(message)))))
    if !π.training
        m[isless.(-m, 0)] .= 1
        m[isless.(m, 0)] .= 0
    end
    m
end

function build_message_dict_vector(π::DIAL, env::AbstractEnv)
    n = (r) -> length(π.τ[r][:message])
    m = Dict(p => getindex(π.τ[p][:message], n(p) - π.n_message_bits + 1:n(p)) for p ∈ players(env))
    messages = Dict(p =>
                    π.emb["message"](
                      # convert(Array{Float32},
                        reduce(vcat,
                               [dru(π, m[r]) for r ∈ filter(e -> e != p, players(env))]
                        )
                      # )
                    )
            for p ∈ players(env))
    messages
end

function (π::DIAL)(::PreEpisodeStage, env::AbstractEnv)
    for p in players(env)
        h = [Flux.reset!(π.c_nets[p][iᵣ]) for iᵣ in π.rnn_idx]
        push!(π.τ[p][:hiddenState], h)
    end
end

# action selection
function (π::DIAL)(env::AbstractEnv)
  zₐ = Dict(p => π.emb["agent"](convert(Array{Float32}, Flux.onehot(p, players(env)))) for p ∈ players(env))
  zₒ = Dict(p => π.emb["observation"](state(env, p)) for p in players(env))
  z = zₐ + zₒ
  if length(π.τ[players(env)[end]][:action]) > 0
    # TODO: Implement message embedding as embedding of the messages of the other agents
    zₘ = build_message_dict_vector(π, env)
    zᵤ = Dict(p =>
        π.emb["action"](convert(Array{Float32}, Flux.onehot(getindex(π.τ[p][:action], length(π.τ[p][:action])), 1:length(action_space(env)))))
         for p ∈ players(env))
    z = z + zₘ + zᵤ
  end
  c_net = [π.c_nets[p](z[p]) for p ∈ players(env)]
  a = [π.explorer(getindex(c_net[iₚ], collect(1:length(action_space(env))))) for iₚ ∈ 1:length(players(env))]
  a = Dict(zip(players(env), a))
  m = [getindex(c_net[iₚ], collect(length(action_space(env)) + 1:length(c_net[iₚ]))) for iₚ ∈ 1:length(players(env))]
  m = Dict(zip(players(env), m))
  A = MultiActionMessage{Dict, Dict}(a, m)
  A
end

# trajectory addition before action is taken. Add current action, transmitted message and observation
function (π::DIAL)(::PreActStage, env::AbstractEnv, action::MultiActionMessage)
    # add messages, hidden states of cnet
    for p in players(env)
        oₜ = state(env, p)
        uₜ, mₜ = (action.action[p], action.message[p])
        push!(π.τ[p][:action], uₜ)
        push!(π.τ[p][:message], mₜ)
        push!(π.τ[p][:observation], oₜ)
        h = [π.c_nets[p][iᵣ].state for iᵣ in π.rnn_idx]
        push!(π.τ[p][:hiddenState], h)
    end
end

function (π::DIAL)(::PostActStage, env::AbstractEnv)
    rₜ = reward(env)
    dₜ = is_terminated(env)
    for p in players(env)
        oₙ = state(env, p)
        push!(π.τ[p][:reward], rₜ)
        push!(π.τ[p][:terminal], dₜ)
        push!(π.τ[p][:next_observation], oₙ)
    end
end

function (π::DIAL)(::PostEpisodeStage, env::AbstractEnv)
    if length(π.τ[players(env)[1]]) > 32*25
    end
end

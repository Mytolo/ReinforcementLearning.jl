export MultiAgentManager

Base.@kwdef mutable struct MultiAgentManager{T <: AbstractPolicy} <: AbstractPolicy
    agents::Dict{S,T}
end

Base.getindex(A::MultiAgentManager, x) = getindex(A.agents, x)

"""
    MultiAgentManager(player => policy...)

This is the simplest form of multiagent system. At each step they observe the
environment from their own perspective and get updated independently.
"""

RLBase.prob(A::MultiAgentManager, env::AbstractEnv, args...) = prob(A[current_player(env)].policy, env, args...)

(A::MultiAgentManager)(env::AbstractEnv) = A(env, DynamicStyle(env))

function (A::MultiAgentManager)(env::AbstractEnv, ::Stochastic)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    return A[current_player(env)](env)
end

function (A::MultiAgentManager)(env::AbstractEnv, ::Sequential)
    while current_player(env) == chance_player(env)
        env |> legal_action_space |> rand |> env
    end
    return A[current_player(env)](env)
end

function (A::MultiAgentManager)(env::AbstractEnv, ::Simultaneous)
    @error "MultiAgentManager doesn't support simultaneous environments. Please consider applying `SequentialEnv` wrapper to environment first."
end

function (A::MultiAgentManager)(stage::PreActStage, env::AbstractEnv)
    A[current_player(env)](stage, env)
end

function (A::MultiAgentManager)(stage::AbstractStage, env::AbstractEnv)
    A[current_player(env)](stage, env)
end

function (A::MultiAgentManager{<:Agent})(::PostActStage, env::AbstractEnv)
    # in the multi agent case, the immediate rewards are updated when last player took its action
    if current_player(env) == last(players(env))
        for (p, agent) in A.agents
            update!(agent.cache, reward(env, p), is_terminated(env))
        end
    end
end

function RLBase.optimise!(A::MultiAgentManager)
    for (_, agent) in A.agents
        RLBase.optimise!(agent)
    end
end

using MultiAgentReinforcementLearning
using StableRNGs
using PyCall
using ReinforcementLearningEnvironments
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningZoo
using Distributions
using Flux
using Flux: glorot_uniform

using Test


@testset "MultiAgentReinforcementLearning" begin
    include("independent_learner.jl")
    include("maddpg_learner.jl")
    @testset "False" begin
        @test 2 != 1
    end
end
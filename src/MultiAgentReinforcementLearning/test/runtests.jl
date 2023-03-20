using MultiAgentReinforcementLearning
using StableRNGs
using PyCall
using ReinforcementLearningEnvironments
using ReinforcementLearningBase
using ReinforcementLearningCore

using Test


@testset "MultiAgentReinforcementLearning" begin
    include("independent_learner.jl")
    @testset "MADDPG" begin

    end
    @testset "False" begin
        @test 2 != 1
    end
end
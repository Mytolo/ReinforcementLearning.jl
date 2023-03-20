using MultiAgentReinforcementLearning
using StableRNGs
using PyCall
using ReinforcementLearningEnvironments
using ReinforcementLearningBase
using ReinforcementLearningCore

using Test


@testset "MultiAgentReinforcementLearning" begin
    @testset "IndependentLearningAgents" begin
        e = PettingzooEnv("mpe.simple_spread_v2");
        m = MultiAgentManager(Dict(player => 
                        Agent(RandomPolicy(action_space(e, player)), 
                            Trajectory(
                                container=CircularArraySARTTraces(
                                  capacity=100,
                                  state=Float32 => (length(state_space(e, player).domains),),
                                ),
                                sampler=NStepBatchSampler{SS′ART}(
                                    n=1,
                                    γ=1,
                                    batch_size=1,
                                    rng=StableRNG(1)
                                ),
                                controller=InsertSampleRatioController(
                                    threshold=1,
                                    n_inserted=0                       
                                ))
                        )
                        for player in players(e)), 
                        current_player(e))
        s = [state(e, player) for player in players(e)]
        r = [reward(e, player) for player in players(e)]
        n = 1
        while !is_terminated(e)
            run(m, e, StopAfterStep(3), EmptyHook())
            n += 1
        end
        
    end
    @testset "False" begin
        @test 2 != 1    
    end
end
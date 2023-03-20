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
                        current_player(e));
        s = [state(e, player) for player in players(e)]
        r = [reward(e, player) for player in players(e)]
        n = 1
        while n <= 25*length(players(e))
            println()
            m(PreActStage(), e)

            e |> m |> e

            if n % length(players(e)) == 0
              s = hcat(s, [state(e, player) for player in players(e)])
              r = isempty(r) ? [reward(e, player) for player in players(e)] : hcat(r, [reward(e, player) for player in players(e)])
            end

            if n % length(players(e)) == 0 && n > length(players(e))
                for a in players(e)
                    @test n ÷ length(players(e)) - 1 == length(m.agents[a].trajectory.container)
                end
                for (i, a) in enumerate(players(e))
                    # states are stored previous to taking action => step - 1 where step is n ÷ length(players(e)) because of sequential execution
                    @test s[i, n ÷ length(players(e)) - 1] == m.agents[a].trajectory.container[:state][n ÷ length(players(e)) - 1]
                end
                for (i, a) in enumerate(players(e))
                    # rewards are stored in trajectory after players done their action
                    @test r[i, n ÷ length(players(e)) - 1] == m.agents[a].trajectory.container[:reward][n ÷ length(players(e)) - 1]
                end
            end
            
            optimise!(m)

            m(PostActStage(), e)
            n += 1
        end
        
    end
    @testset "False" begin
        @test 2 != 1    
    end
end
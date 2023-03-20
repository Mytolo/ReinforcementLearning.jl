function test_env(name::String, cap::Integer, nᵢ::Integer)
    e = PettingzooEnv(name)
    m = MultiAgentManager(Dict(player =>
                    Agent(RandomPolicy(action_space(e, player)),
                        Trajectory(
                            container=CircularArraySARTTraces(
                              capacity=cap,
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
    s = [state(e, player) for player ∈ players(e)]
    r = [reward(e, player) for player ∈ players(e)]
    t = []
    a = []

    n = 1
    loc_actions = []
    loc_terminal = []
    while n <= nᵢ*length(players(e))
        if is_terminated(e)
            reset!(e)
        end

        m(PreActStage(), e)

        action = m(e)

        e(action)
        push!(loc_actions, action)
        push!(loc_terminal, is_terminated(e))

        if n % length(players(e)) == 1 && n > 1 && n < (nᵢ-1)*length(players(e))
          r = hcat(r, [reward(e, player) for player ∈ players(e)])
          s = hcat(s, [state(e, player) for player in players(e)])
        end

        if n % length(players(e)) == 0 && n < nᵢ*length(players(e))
            a = isempty(a) ? loc_actions : hcat(a, loc_actions)
            t = isempty(t) ? loc_terminal : hcat(t, loc_terminal)
            loc_actions = []
            loc_terminal = []
        end


        if n % length(players(e)) == 0 && n > length(players(e))
            for (i, p) in enumerate(players(e))

                @test min(cap, n ÷ length(players(e)) - 1) == length(m.agents[p].trajectory.container)
                @test s[i, max(1, n ÷ length(players(e)) - cap):(n == nᵢ * length(players(e)) ? end : end-1)] == m.agents[p].trajectory.container[:state]
                @test r[i, max(1, n ÷ length(players(e)) - cap):(n == nᵢ * length(players(e)) ? end : end-1)] ≈ m.agents[p].trajectory.container[:reward]
                @test t[i, max(1, n ÷ length(players(e)) - cap):(n == nᵢ * length(players(e)) ? end : end-1)] ≈ m.agents[p].trajectory.container[:terminal]
                @test a[i, max(1, n ÷ length(players(e)) - cap):(n == nᵢ * length(players(e)) ? end : end-1)] ≈ m.agents[p].trajectory.container[:action]

            end
        end


        # no need to call optimise!(m)

        m(PostActStage(), e)
        n += 1
    end
    for (i, p) in enumerate(players(e)) @test r[i, max(1, n ÷ length(players(e)) - cap):end] ≈ m.agents[p].trajectory.container[:reward] end
    for (i, p) in enumerate(players(e)) @test s[i, max(1, n ÷ length(players(e)) - cap):end] == m.agents[p].trajectory.container[:state] end
    for (i, p) in enumerate(players(e)) @test t[i, max(1, n ÷ length(players(e)) - cap):end] ≈ m.agents[p].trajectory.container[:terminal] end
    for (i, p) in enumerate(players(e)) @test a[i, max(1, n ÷ length(players(e)) - cap):end] ≈ m.agents[p].trajectory.container[:action] end
end

@testset "IndependentLearningAgents" begin
    @testset "Pettingzoo_MPE_independent" begin
        test_env("mpe.simple_spread_v2", 2, 100)
        test_env("mpe.simple_spread_v2", 40, 100)
        test_env("mpe.simple_spread_v2", 40, 25)
    end
end
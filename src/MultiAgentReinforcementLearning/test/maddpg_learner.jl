function test_maddpg_env(name::String, cap::Integer, nᵢ::Integer)
    rng = StableRNG(123)
    init = glorot_uniform(rng)
    e = PettingZooEnv(name; continuous_actions=true)
    na = (player) -> length(action_space(e, player).domains)
    critic_dim = sum(length(state(e, p)) + na(p) for p in players(e))
    create_actor(player) = Chain(
        Dense(length(state(e, player)), 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, na(player); init = init)
    )

    create_critic(critic_dim) = Chain(
        Dense(critic_dim, 1, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
    ) 
    create_policy(player) = DDPGPolicy(
            actor = Approximator(
                model = TwinNetwork(
                    create_actor(player),
                    ρ = 0.995f0
                ),
                optimiser = Flux.Optimise.Optimiser(ClipNorm(0.5), Adam(1e-2)),
              ),
            critic = Approximator(
                model = TwinNetwork(
                    create_critic(critic_dim),
                    ρ = 0.995f0
                ),
                optimiser = Flux.Optimise.Optimiser(ClipNorm(0.5), Adam(1e-2)),
            ),
            γ = 0.95f0,
            na = na(player),
            start_steps = 0,
            start_policy = e -> rand(Distributions.Uniform(0, 1), na(player)),
            update_after = 512 * 25, # batch_size * e.max_steps
            act_upper_limit = 1.0,
            act_lower_limit = 0.0,
            act_noise = 9e-2,
        )
        create_trajectory(player) = Trajectory(
            container=CircularArraySARTTraces(
                capacity=cap,
                state=Float32 => (length(state(e, player)),),
                action=Float64 => (length(action_space(e, player).domains),)
            ),
            sampler=NStepBatchSampler{SS′ART}(
                n=1,
                γ=1,
                batch_size=1,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=1
            )
        )
    m = MADDPGManager(
        Dict(
            player => Agent(
                policy = create_policy(player),
                trajectory = create_trajectory(player),
            ) for player in players(e)
        ),
        512, # batch_size
        1, # update_freq
        0, # initial update_step
        rng
    )
    s = [state(e, player) for player in players(e)]
    r = []
    t = []
    a = []

    n = 1
    while n <= nᵢ

        m(PreActStage(), e)

        action = m(e)        
        e(action)

        if is_terminated(e)
            reset!(e)
        end


        if n != nᵢ
            r = isempty(r) ? [reward(e, player) for player ∈ players(e)] : hcat(r, [reward(e, player) for player ∈ players(e)])
            a = isempty(a) ? [action[p] for p in players(e)] : hcat(a, [action[p] for p in players(e)])
            t = isempty(t) ? [is_terminated(e) for _ ∈ players(e)] : hcat(t, [is_terminated(e) for _ ∈ players(e)])
            
            # Test state here. Otherwise, the index would be confusingly be end - 2 and for the last iteration step end-1 since push
            # to s.
            for (i, p) in enumerate(players(e)) @test n < 2 || s[i, end-1] == m.agents[p].trajectory.container[:state][end] end
            s = isempty(s) ? [state(e, player) for player in players(e)] : hcat(s, [state(e, player) for player in players(e)])
        end
        



        
        for (i, p) in enumerate(players(e))
            @test min(cap, n - 1) == length(m.agents[p].trajectory.container)
            # if n is 1, there are no elements stored in the trajectory
            @test n < 2 || r[i, max(1, n - cap):(n == nᵢ ? end : end-1)] ≈ m.agents[p].trajectory.container[:reward]
            @test n < 2 || t[i, max(1, n - cap):(n == nᵢ ? end : end-1)] ≈ m.agents[p].trajectory.container[:terminal]
            @test n < 2 || a[i, max(1, n - cap):(n == nᵢ ? end : end-1)] ≈ m.agents[p].trajectory.container[:action]
        end
        


        # no need to call optimise!(m)

        m(PostActStage(), e)
        n += 1
    end
    for (i, p) in enumerate(players(e)) @test r[i, max(1, n - cap - 1):end] ≈ m.agents[p].trajectory.container[:reward] end
    for (i, p) in enumerate(players(e)) @test s[i, max(1, n - cap - 1):end-1] == m.agents[p].trajectory.container[:state] end
    for (i, p) in enumerate(players(e)) @test t[i, max(1, n - cap - 1):end] ≈ m.agents[p].trajectory.container[:terminal] end
    for (i, p) in enumerate(players(e)) @test a[i, max(1, n - cap - 1):end] ≈ m.agents[p].trajectory.container[:action] end
end

@testset "MADDPG_Learner" begin
    @testset "Pettingzoo_MPE_independent" begin
        test_maddpg_env("mpe.simple_spread_v2", 1000, 100)
        test_maddpg_env("mpe.simple_spread_v2", 2, 100)
        test_maddpg_env("mpe.simple_spread_v2", 2, 1_000)
    end
end
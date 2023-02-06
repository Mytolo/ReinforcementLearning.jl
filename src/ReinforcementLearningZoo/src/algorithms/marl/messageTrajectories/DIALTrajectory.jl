export CircularArrayOAHMRTOTrajectory

const OAHMRTO = (:observation, :action, :hiddenState, :message, :reward, :terminal, :next_observation)

const CircularArrayOAHMRTOTrajectory = Trajectory{
    <:NamedTuple{
        OAHMRTO,
        <:Tuple{
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer,
            <:CircularArrayBuffer
        },
    },
}



CircularArrayOAHMRTOTrajectory(;
    capacity::Int,
    observation = Int => (),
    action = Int => (),
    hiddenState = Int => (),
    message = Float32 => (),
    reward = Float32 => (),
    terminal = Bool => (),
    next_observation = Int => (),
) = merge(
CircularArrayTrajectory(; capacity = capacity + 1, observation = observation, action = action, hiddenState = hiddenState, message = message),
CircularArrayTrajectory(; capacity = capacity, reward = reward, terminal = terminal, next_observation = next_observation),
)

Base.length(t::CircularArrayOAHMRTOTrajectory) = length(t[:terminal])
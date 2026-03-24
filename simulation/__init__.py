from gymnasium.envs.registration import register

register(
    id="autonomous-driving-v1",
    entry_point="simulation.environment:DrivingEnv",
    max_episode_steps=800,
)
from gym.envs.registration import register

register(
    id='WindTurbine-analytical-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(60.0/0.05), # 30s -> 600 steps
)

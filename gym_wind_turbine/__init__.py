from gym.envs.registration import register

register(
    id='WindTurbine-analytical-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(180.0/0.05),
)

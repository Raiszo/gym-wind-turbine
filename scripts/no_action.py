import gym
import numpy as np
import gym_wind_turbine
from gym_wind_turbine.envs.wind_turbine_analytical import WindTurbineAnalytical
from rl_agents.ppo import get_env
from scripts.plots import make_plots


def run_no_action(env: gym.Wrapper, max_steps: int) -> float:
    reward_sum = 0.0
    state = env.reset()

    for i in range(1, max_steps + 1):
        state, reward, done, _ = env.step(np.array([0.0]))

        if done:
            break

    assert isinstance(env.env, WindTurbineAnalytical)

    rec, dt = env.env.get_recordings()
    make_plots(rec, dt)


    return reward_sum

if __name__ == '__main__':
    # env = get_env('WindTurbine-analytical-v0', record=True, omega_0=0.9, T_gen_0=500.0)
    env = get_env('WindTurbine-analytical-v0', record=True)
    run_no_action(env, 240*20)

import gym
import numpy as np
import tensorflow as tf
import gym_wind_turbine
from gym_wind_turbine.envs.wind_turbine_analytical import WindTurbineAnalytical
from rl_agents.ppo import GaussianSample, get_env
from scripts.plots import make_plots
from tqdm import tqdm


def run_episode(env: gym.Wrapper, actor: tf.keras.Model, max_steps: int) -> float:
    state = tf.constant(env.reset(), dtype=tf.float32)

    # screen = env.render(mode='rgb_array')
    reward_sum = 0.0

    for _ in tqdm(range(1, max_steps + 1)):
        state = tf.expand_dims(state, 0)
        action_na = actor(state).mean()

        # print(action_na[0].numpy())
        state, reward, done, _ = env.step(action_na[0].numpy())
        # print(action_na[0], reward)
        # print(state, reward, done)
        # print(reward)
        reward_sum += reward.astype(np.float32).item()
        state = tf.constant(state, dtype=tf.float32)

        # screen = env.render(mode='rgb_array')
        # print(reward)

        if done:
            break

    assert isinstance(env.env, WindTurbineAnalytical)

    rec, dt = env.env.get_recordings()
    make_plots(rec, dt)


    return reward_sum


if __name__ == '__main__':
    # env = get_env('WindTurbine-analytical-v0', record=True, omega_0=0.9, T_gen_0=500.0)
    env = get_env('WindTurbine-analytical-v0', record=True)
    custom_objects={'GaussianSample': GaussianSample}
    with tf.keras.utils.custom_object_scope(custom_objects):
        actor = tf.keras.models.load_model(actor_dir)

        run_episode(env, actor, 240*20)

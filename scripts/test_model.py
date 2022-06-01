from typing import Tuple
import gym
import numpy as np
import tensorflow as tf
import gym_wind_turbine
from gym_wind_turbine.envs.wind_turbine_analytical import RecordedVariables, WindTurbineAnalytical
from rl_agents.ppo import GaussianSample, get_env
from scripts.plots import make_plots
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
import string, random

def run_episode(env: gym.Wrapper, actor: tf.keras.Model, max_steps: int) -> Tuple[RecordedVariables, float]:
    state = tf.constant(env.reset(), dtype=tf.float32)

    reward_sum = 0.0

    for _ in tqdm(range(1, max_steps + 1)):
        state = tf.expand_dims(state, 0)
        action_na = actor(state).mean()

        state, reward, done, _ = env.step(action_na[0].numpy())
        reward_sum += reward.astype(np.float32).item()
        state = tf.constant(state, dtype=tf.float32)

        if done:
            break

    # just for type checking
    assert isinstance(env.env, WindTurbineAnalytical)

    return env.env.get_recordings()


if __name__ == '__main__':
    parser = ArgumentParser(description='Test a (trained) model against an environment')
    parser.add_argument('model', type=str, help='path to model')
    parser.add_argument('--env', type=str, help='gym environment')

    args = parser.parse_args()

    env = get_env(args.env, record=True)
    custom_objects={'GaussianSample': GaussianSample}
    with tf.keras.utils.custom_object_scope(custom_objects):
        actor = tf.keras.models.load_model(args.model)

        rec, dt = run_episode(env, actor, 240*20)
        make_plots(rec, dt)

    save_to_disk = input('OK to save recordings to disk [Y/N]? ').lower() == 'y'

    if save_to_disk:
        rewards = np.stack(rec['rewards'], axis=0)

        recordings_df = pd.DataFrame({
            't': rec['t'],
            'v_wind': rec['v_wind'],
            'w_r': rec['w_r'],
            'w_r_dot': rec['w_r_dot'],
            'T_r': rec['T_aero'],
            'T_em': rec['T_gen'],
            'action': [float('NaN')] + rec['action'],
            'clipped_action': [float('NaN')] + rec['clipped_action'],
            'reward': np.hstack(((np.nan), np.sum(rewards, axis=1))),
            'C_p': rec['C_p'],
            'tsr': rec['tsr'],
        })

        random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        file_name = f'{args.env}-{random_id}.csv'
        recordings_df.to_csv(file_name)

        print(f'data saved to {file_name}')

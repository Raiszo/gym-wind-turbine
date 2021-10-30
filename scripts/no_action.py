import gym
import gym_wind_turbine
import numpy as np

env = gym.make('WindTurbine-analytical-v0', record=True)

obs = env.reset()

while True:
# for i in range(20):
    obs, reward, done, info = env.step(np.array([0.0]))
    print(obs, reward, done)

    if done:
        break

import gym
import gym_wind_turbine
import numpy as np
from rl_agents import run_ppo_experiment


base_dir = 'experiments'

def main():
    ####
    # Experiment parameters
    ####
    # base dir is experiments/trials
    run_ppo_experiment(
        environment='WindTurbine-analytical-v0',
        n_iterations=600, iteration_size=8192,
        n_epochs=10, minibatch_size=128,
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=5e-3,
        actor_output_activation='linear',
        base_dir=base_dir,
        # early_stop_reward_threshold=-200
    )

if __name__ == '__main__':
    main()

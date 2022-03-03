from gym_wind_turbine.envs.wind_turbine_analytical import RecordedVariables, Rotor
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

rotor = Rotor(rho=1.25, R=38.5, beta=0)

# TODO: maybe computing a 3d grid of tsr, v_wind and Power could be used as a lookup table
# would be a lot faster if the experiment uses random wind speed
@lru_cache
def mpp(v_wind: float) -> float:
    tsr = np.arange(0.01, 14, 0.01)

    C_p = rotor.compute_cp(tsr)
    P = 0.5 * rotor.rho * np.pi * rotor.R**2 * C_p * v_wind**3

    return np.amax(P)


def make_plots(rec: RecordedVariables, dt: float):
    t = np.array(rec['t'])
    v_wind = np.array(rec['v_wind'])
    T_aero = np.array(rec['T_aero'])
    T_gen = np.array(rec['T_gen'])
    w_r = np.array(rec['w_r'])
    action = np.array(rec['action'])
    clipped_action = np.array(rec['clipped_action'])
    rewards = np.stack(rec['rewards'], axis=0)

    C_p = np.array(rec['C_p'])
    tsr = np.array(rec['tsr'])

    ###########
    # POWER
    ###########
    plt.figure(1)
    plt.plot(t, T_aero*1e-3, 'b', label='$T_a$')
    plt.plot(t, T_gen*1e-3*105.494, 'r--', label='$n_g T_g$')
    plt.ylabel('Torque [kN.m]')
    plt.xlabel('t [s]')
    plt.legend()

    ###########
    # Torque comparison T_a // n_g T_g
    ###########
    mpp_vec = np.vectorize(mpp)
    plt.figure(2)
    plt.plot(t, T_aero*w_r*1e-3, 'b', label='$P_a$')
    plt.plot(t, mpp_vec(v_wind)*1e-3, 'r--', label='$MPP$')
    plt.ylabel('Power [kW]')
    plt.xlabel('t [s]')
    plt.legend()


    ###########
    # multi plot: w_v, w_r, C_p, tsr
    ###########
    plot_vars = [
        (v_wind, 'wind velocity [m/s]'),
        (w_r, 'omega rad/s'),
        (C_p, 'C_p'),
        (tsr, 'tsr')
    ]

    _, ax = plt.subplots(4,1, sharex=True)
    for [a,v,label] in zip(ax, *zip(*plot_vars)):
        a.plot(t, v)
        a.set_ylabel(label)
        a.grid()
    ax[-1].set_xlabel('t [s]')

    ###########
    # Action
    ###########
    plt.figure(4)
    plt.plot(t[:-1], action/dt, label='raw')
    plt.plot(t[:-1], clipped_action/dt, label='clipped')
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('$\\dot T_g$ [kN.m/s]')
    plt.grid()

    ###########
    # REWARD
    ###########
    # remember that the tuple for training was state, action, reward
    # given state s1 the agent took action a1 that "moved" it to state s2
    # and gave it reward r1
    # so it is natural for the reward vector to have 1 element less than
    # state and action
    plt.figure(5)
    # plt.plot(t[:-1], rewards[0], label='power reward')
    # plt.plot(t[:-1], rewards[1], label='control reward')
    # plt.plot(t[:-1], rewards[2], label='alive bonus')
    plt.plot(t[:-1], np.sum(rewards, axis=1), 'b')
    plt.xlabel('t [s]')
    plt.ylabel('Reward')
    plt.grid()

    plt.show()

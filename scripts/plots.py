from gym_wind_turbine.envs.wind_turbine_analytical import RecordedVariables, WindTurbineAnalytical
import numpy as np
import matplotlib.pyplot as plt


def make_plots(rec: RecordedVariables, dt: float):
    t = np.array(rec['t'])
    T_aero = np.array(rec['T_aero'])
    T_gen = np.array(rec['T_gen'])
    omega = np.array(rec['omega'])
    action = np.array(rec['action'])
    clipped_action = np.array(rec['clipped_action'])
    rewards = np.stack(rec['rewards'], axis=1)

    C_p = np.array(rec['C_p'])
    tsr = np.array(rec['tsr'])

    plt.figure(1)
    plt.plot(t, T_aero*1e-3, 'b', label='T_aero')
    plt.plot(t, T_gen*1e-3*105.494, 'r', label='T_gen x N_gear')
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('Torque [kN.m]')
    plt.grid()

    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(t, T_aero*omega*1e-3)
    ax[0].set_ylabel('Power [kW]')
    ax[0].grid()

    ax[1].plot(t, C_p)
    ax[1].set_ylabel('C_p')
    ax[1].grid()

    ax[2].plot(t, tsr)
    ax[2].grid()
    ax[2].set_ylabel('tsr')
    ax[2].set_xlabel('t [s]')

    plt.figure(3)
    plt.plot(t[:-1], action/dt, label='raw')
    plt.plot(t[:-1], clipped_action/dt, label='clipped')
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('T_gen_dot [kN.m/s]')
    plt.grid()

    plt.figure(4)
    plt.plot(t, omega)
    plt.xlabel('t [s]')
    plt.ylabel('omega rad/s')
    plt.grid()

    # remember that the tuple for training was state, action, reward
    # given state s1 the agent took action a1 that "moved" it to state s2
    # and gave it reward r1
    # so it is natural for the reward vector to have 1 element less than
    # state and action
    plt.figure(5)
    plt.plot(t[:-1], rewards[0], label='power reward')
    plt.plot(t[:-1], rewards[1], label='control reward')
    plt.plot(t[:-1], rewards[2], label='alive bonus')
    plt.legend()
    plt.xlabel('t [s]')
    plt.grid()
    
    plt.show()

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
    # multi plot: v_w, w_r, P, T
    ###########
    fg, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, sharex=True, figsize=[8, 4.8])
    # wind
    ax0.plot(t, v_wind)
    ax0.set_ylabel('$v_w$ [m/s]')

    # POWER vs ideal power
    mpp_vec = np.vectorize(mpp)
    ax1.plot(t, T_aero*w_r*1e-3, 'b', label='$P_r$')
    ax1.plot(t, mpp_vec(v_wind)*1e-3, 'r--', label='$P_r^*$')
    ax1.set_ylabel('Power [kW]')
    ax1.set_xlabel('t [s]')
    ax1.legend()

    # w_r
    ax2.plot(t, w_r)
    ax2.set_ylabel('$w_r$ rad/s')

    # Torque comparison T_a // n_g T_g
    ax3.plot(t, T_aero*1e-3, 'b', label='$T_r$')
    ax3.plot(t, T_gen*1e-3*105.494, 'r--', label='$T_g$')
    ax3.set_ylabel('Torque [kN.m]')
    ax3.set_xlabel('t [s]')
    ax3.legend()


    fg.tight_layout()

    ###########
    # Action
    ###########
    plt.figure(2)
    plt.plot(t[:-1], (action/dt)*105.494, label='raw')
    plt.plot(t[:-1], (clipped_action/dt)*105.494, label='clipped')
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('$\dot{T_g}$ [kN.m/s]')
    plt.grid()
    plt.tight_layout()


    ###########
    # REWARD
    ###########
    # remember that the tuple for training was state, action, reward
    # given state s1 the agent took action a1 that "moved" it to state s2
    # and gave it reward r1
    # so it is natural for the reward vector to have 1 element less than
    # state and action
    # plt.figure(5)
    # plt.plot(t[:-1], rewards[0], label='power reward')
    # plt.plot(t[:-1], rewards[1], label='control reward')
    # plt.plot(t[:-1], rewards[2], label='alive bonus')
    # plt.plot(t[:-1], np.sum(rewards, axis=1), 'b')
    # plt.xlabel('t [s]')
    # plt.ylabel('Reward')
    # plt.grid()

    plt.show()

# potencia de rotor -> mecanica -> aerodynamic
# torque/par electromagnetico, no mencionar potencia
# - explicar el tema de iteracion
# - MPP -> P_a^ideal
# - graficas
# ----- par mecanico y electromagnetico tomando n_g
# ----- omega
# ----- v_wind
# ----- potencia de rotor y rotor ideal

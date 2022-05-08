import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gym_wind_turbine.envs.wind_turbine_analytical import Rotor


def plotIn2d():
    v_wind = 8.0
    rotor = Rotor(rho=1.25, R=38.5, beta=0)

    w_r = np.arange(0.01, 5, 0.01)
    tsr = rotor.R * w_r / v_wind
    # tsr = np.arange(0.01, 14, 0.01)
    m_inv = 1/tsr - 0.035
    c_p = 0.22 * (116*m_inv - 5) * np.exp(-12.5*m_inv)


    # c_p_max = c_p.max()
    tsr_max = tsr[np.argmax(c_p)]
    w_r_max = tsr_max * v_wind / rotor.R

    P_aero_max, _, _ = rotor.get_aerodynamics(v_wind, w_r_max)
    print('P_aero_max', P_aero_max)

    plt.plot(tsr, c_p)
    plt.grid()

    plt.show()


def plotIn3d():
    v_wind = np.arange(8.0, 12.0, 0.01)
    # v_wind = np.arange(1.0, 15.0, 0.01)
    # for a tsr between 0 and 14 use w_r 0 and 2.9
    wr = np.arange(0.01, 4, 0.01)

    rotor = Rotor(rho=1.25, R=38.5, beta=0)

    # make meshgrid, x=C_p & y=v_wind
    x_wr, y_vw = np.meshgrid(wr, v_wind)
    tsr = rotor.R*x_wr / y_vw
    C_p = rotor.compute_cp(tsr)
    P = 0.5 * rotor.rho * np.pi * rotor.R**2 * C_p * y_vw**3
    P = np.clip(P, 0, None)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_wr, y_vw, P, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('$w_r$')
    ax.set_ylabel('$v_{wind}$')
    ax.set_zlabel('$P_r$')

    plt.show()

    # # c_p_max = c_p.max()
    # idx = np.argmax(C_p)
    # tsr_max = tsr[idx]
    # P_max = P[idx]
    # # w_r_max = tsr_max * v_wind / rotor.R

    # print('P max:', P_max)

    # plt.plot(tsr, C_p)

    # plt.show()

if __name__ == '__main__':
    plotIn3d()
    # plotIn2d()

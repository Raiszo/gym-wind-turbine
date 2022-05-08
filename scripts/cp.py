import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gym_wind_turbine.envs.wind_turbine_analytical import Rotor


def main():
    v_wind = np.arange(8.0, 12.0, 0.01)
    tsr = np.arange(0.01, 14, 0.01)

    rotor = Rotor(rho=1.25, R=38.5, beta=0)

    # make meshgrid, x=C_p & y=v_wind
    x_tsr, y_vw = np.meshgrid(tsr, v_wind)
    C_p = rotor.compute_cp(x_tsr)
    P = 0.5 * rotor.rho * np.pi * rotor.R**2 * C_p * y_vw**3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_tsr, y_vw, P, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('tsr')
    ax.set_ylabel('$v_wind$')
    ax.set_zlabel('P')

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
    main()

import numpy as np
import matplotlib.pyplot as plt
from gym_wind_turbine.envs.wind_turbine_analytical import Rotor


def main():
    v_wind = 11.0
    tsr = np.arange(0.01, 14, 0.01)
    m_inv = 1/tsr - 0.035
    c_p = 0.22 * (116*m_inv - 5) * np.exp(-12.5*m_inv)

    rotor = Rotor(rho=1.25, R=38.5, beta=0)

    # c_p_max = c_p.max()
    tsr_max = tsr[np.argmax(c_p)]
    w_r_max = tsr_max * v_wind / rotor.R

    P_aero_max = rotor.get_aerodynamic_power(v_wind, w_r_max)
    print('P_aero_max', P_aero_max)

    plt.plot(tsr, c_p)
    plt.grid()

    plt.show()

if __name__ == '__main__':
    main()

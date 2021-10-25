from dataclasses import dataclass
import functools
import gym
from gym import spaces
import numpy as np

@dataclass
class DriveTrain:
    n_gear: float
    """gear box ration"""
    I_rotor: float
    """rotor inertia [kg*m^2]"""
    I_gen: float
    """generator inertia [kg*m^2]"""
    K_rotor: float
    """rotor friction coefficient [N.m/rad/s]"""
    K_gen: float
    """generator friction coefficient [N.m/rad/s]"""

    @functools.cached_property
    def I_t(self):
        """total inertia"""
        return self.I_rotor + self.n_gear ** 2 * self.I_gen

    @functools.cached_property
    def K_t(self):
        return self.K_rotor + self.n_gear ** 2 * self.K_gen

    def get_rotor_angular_acc(self, T_aero: float, T_gen: float):
        return (T_aero - self.n_gear*T_gen) / self.I_t

@dataclass
class Rotor:
    rho: float
    """air density [kg/m3]"""
    R: float
    """rotor radius [m]"""
    beta: float
    """blade pitch angle"""

    def get_aerodynamic_power(self, v_wind: float, w_rotor: float):
        """
        Make sure w_rotor is not < 0, else C_p may overflow

        returns power in Watts
        v_wind [m/s]
        w_rotor [rad/s]
        """
        tsr = self.R * w_rotor / v_wind
        m_inv = 1/(tsr + 0.08*self.beta) - 0.035/(self.beta**3 + 1)
        C_p = 0.22 * (116*m_inv - 0.4*self.beta - 5) * np.exp(-12.5*m_inv)

        return 0.5 * self.rho * np.pi * self.R**2 * C_p * v_wind**3
        

@dataclass
class EnvProps:
    pass


class WindTurbineAnalytical(gym.Env):
    dt = 0.05

    def __init__(self) -> None:
        super().__init__()

        self.rotor = Rotor(rho=1.25, R=38.5, beta=0)
        self.drive_train = DriveTrain(
            n_gear= 105.494,
            I_rotor= 4456761.0,
            I_gen= 123.0,
            K_rotor= 45.52,
            K_gen= 0.4,
        )
        self.v_wind = 11        # [m/s]

        obs_space = np.array([
            [0, 30.0],
            [0, np.finfo(np.float64).max],
            [np.finfo(np.float64).min, np.finfo(np.float64).max],
            [0, np.finfo(np.float64).max],
            [0, np.finfo(np.float64).max],
        ])
        self.observation_space = spaces.Box(
            low=obs_space[:, 0],
            high=obs_space[:, 1],
            dtype=np.float64,
        )

        self.action_space = spaces.Box(
            low=-15.0,
            high=15.0,
            shape=(1,)
        )

        self.omega = 1.0        # [rad/s]
        self.state: np.ndarray

    def reset(self):
        """
        state:
        - wind velocity: always constant ~ 11 m/s
        - rotor angular velocity ~ 2 rad/s
        - rotor angular acceleration ~ rad/s^2
        - aerodynamic torque ~ 250 kN.m
        - generator torque ~ 250 kN.m
        """
        self.omega = 1.0

        P_aero = self.rotor.get_aerodynamic_power(self.v_wind, self.omega)
        T_aero = P_aero / self.omega
        # print(P_aero, T_aero)
        self.state = np.array([
            self.v_wind,
            self.omega,
            0.0,                # omega_dot
            T_aero/1e3,         # T_aero
            50.0,              # T_gen
        ])

        return self.state

    def step(self, a: np.ndarray):
        _, last_omega, omega_dot, _last_T_aero, _T_gen = self.state
        T_gen = (_T_gen + a[0])*1e3

        # going to do the aerodynamic calculations first
        P_aero = self.rotor.get_aerodynamic_power(self.v_wind, self.omega)
        T_aero = P_aero / self.omega
        omega_dot = self.drive_train.get_rotor_angular_acc(T_aero, T_gen)

        # now integrate to have the new omega
        # for now just use simple euler integration
        self.omega += omega_dot * self.dt
        # self.omega = self.omega if self.omega > 5e-5 else 5e-5

        last_P_aero = _last_T_aero*1e3 / last_omega
        rewards = np.array([
            1.0/1e5 * (P_aero - last_P_aero),
            - 0.01 * np.square(a).sum(),
            0.05
        ])

        self.state = np.array([
            self.v_wind,
            self.omega,
            omega_dot,
            T_aero/1e3,
            T_gen/1e3,
        ])

        done = not self.observation_space.contains(self.state) and self.omega < 5e-5

        return self.state, rewards.sum(), done, {}

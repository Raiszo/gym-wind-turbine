from typing import Tuple, TypedDict, List, Union
from dataclasses import dataclass
import functools
import gym
from gym import spaces
import numpy as np

from gym_wind_turbine.envs.wind_generators import WindGenerator


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

    def compute_cp(self, tsr: Union[float, np.ndarray]):
        m_inv = 1/(tsr + 0.08*self.beta) - 0.035/(self.beta**3 + 1)
        C_p = 0.22 * (116*m_inv - 0.4*self.beta - 5) * np.exp(-12.5*m_inv)
        return C_p

    def get_aerodynamics(self, v_wind: float, w_rotor: float) -> Tuple[float, float, float]:
        """
        Make sure w_rotor is not < 0, else C_p may overflow

        returns power in Watts
        v_wind [m/s]
        w_rotor [rad/s]
        """
        tsr = self.R * np.where(w_rotor > 1e-3, w_rotor, 1e-3) / v_wind
        C_p = self.compute_cp(tsr)
        P = 0.5 * self.rho * np.pi * self.R**2 * C_p * v_wind**3

        return P, C_p, tsr

class RecordedVariables(TypedDict):
    """System variables """
    t: List[float]

    v_wind: List[float]
    w_r: List[float]
    w_r_dot: List[float]
    T_aero: List[float]
    T_gen: List[float]
    action: List[float]
    clipped_action: List[float]
    rewards: List[np.ndarray]

    C_p: List[float]
    tsr: List[float]


class WindTurbineAnalytical(gym.Env):
    dt = 0.05

    def __init__(self, rotor: Rotor, drive_train: DriveTrain,
                 wind_generator: WindGenerator, record=False,
                 omega_0: float=1.0, T_gen_0: float=1000.0) -> None:
        super().__init__()
        self.record: bool = record

        self.rotor = rotor
        self.drive_train = drive_train
        self.wind_generator = wind_generator

        obs_space = np.array([
            [0, 30.0],
            [1e-3, np.finfo(np.float64).max],
            [np.finfo(np.float64).min, np.finfo(np.float64).max],
            [0, np.finfo(np.float64).max],
            [0, np.finfo(np.float64).max],
        ])
        self.observation_space = spaces.Box(
            low=obs_space[:, 0],
            high=obs_space[:, 1],
            dtype=np.float64,
        )

        action_space = np.array([-5.0, 5.0]) * self.dt
        self.action_space = spaces.Box(
            low=action_space[0],
            high=action_space[1],
            shape=(1,)
        )

        self.omega = omega_0    # [rad/s]
        self.state: np.ndarray

        self._recordings: RecordedVariables = {
            't': [],
            'v_wind': [],
            'w_r': [],
            'w_r_dot': [],
            'T_aero': [],
            'T_gen': [],
            'action': [],
            'clipped_action': [],
            'rewards': [],
            'C_p': [],
            'tsr': []
        }

        self.T_gen_0 = T_gen_0
        self.omega_0 = omega_0

    def reset(self):
        """
        state:
        - wind velocity: always constant ~ 11 m/s
        - rotor angular velocity ~ 2 rad/s
        - rotor angular acceleration ~ rad/s^2
        - aerodynamic torque ~ 250 kN.m
        - generator torque ~ 250 kN.m
        """
        self.omega = self.omega_0
        self.t = 0.0
        v_wind = self.wind_generator.reset()

        P_aero, C_p, tsr = self.rotor.get_aerodynamics(v_wind, self.omega)
        T_aero = P_aero / self.omega
        T_gen = self.T_gen_0
        self.state = np.array([
            v_wind,
            self.omega,
            0.0,                # omega_dot
            T_aero/1e3,
            T_gen/1e3,
        ])

        if self.record:
            self._recordings['t'] = [self.t]
            self._recordings['v_wind'] = [v_wind]
            self._recordings['w_r'] = [self.omega]
            self._recordings['w_r_dot'] = [0.0]
            self._recordings['T_aero'] = [T_aero]
            self._recordings['T_gen'] = [T_gen]
            self._recordings['action'] = []
            self._recordings['clipped_action'] = []
            self._recordings['rewards'] = []
            self._recordings['C_p'] = [C_p]
            self._recordings['tsr'] = [tsr]

        return self.state

    def step(self, a: np.ndarray):
        _, last_omega, _, last_T_aero, last_T_gen = self.state
        v_wind = self.wind_generator.read()
        # add some saturation :3
        u = np.clip(a[0], self.action_space.low, self.action_space.high)
        # action is T_gen_dot in kN.m
        # print(a)
        T_gen = last_T_gen*1e3 + u[0]*1e3

        # get new omega
        last_P_aero = last_T_aero*1e3 * last_omega
        omega_dot = self.drive_train.get_rotor_angular_acc(last_T_aero*1e3, T_gen)
        # now integrate to update omega
        # for now just use simple euler integration
        self.omega += omega_dot * self.dt

        # since there is a new omega that could be less than 0, please implement a short circuit if so
        P_aero, C_p, tsr = self.rotor.get_aerodynamics(v_wind, self.omega)
        T_aero = P_aero / self.omega

        # print(P_aero, last_P_aero, P_aero-last_P_aero)
        rewards = np.array([
            1.0/1e3 * (P_aero - last_P_aero),
            - 0.1 * np.square(a).sum(),
            0.05,
        ])
        # print(rewards)

        self.state = np.array([
            v_wind,
            self.omega,
            omega_dot,
            T_aero/1e3,
            T_gen/1e3,
        ])
        self.t += self.dt

        done = not self.observation_space.contains(self.state)

        if self.record:
            # we could export more variables like C_p
            self._recordings['t'].append(self.t)
            self._recordings['v_wind'].append(v_wind)
            self._recordings['w_r'].append(self.omega)
            self._recordings['w_r_dot'].append(omega_dot)
            self._recordings['T_aero'].append(T_aero)
            self._recordings['T_gen'].append(T_gen)
            self._recordings['action'].append(a[0])
            self._recordings['clipped_action'].append(u[0])
            self._recordings['rewards'].append(rewards)
            self._recordings['C_p'].append(C_p)
            self._recordings['tsr'].append(tsr)

        return self.state, rewards.sum(), done, {}

    def get_recordings(self):
        return self._recordings, self.dt

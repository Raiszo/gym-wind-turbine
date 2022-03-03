from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Protocol
import numpy as np
import random
import pandas as pd

class WindGenerator(Protocol):
    def read(self) -> float:
        """returns wind in m/s"""
        ...
    def reset(self) -> float: ...

@dataclass
class ConstantWind:
    value: float
    def read(self):
        return self.value
    def reset(self):
        return self.value

class RandomConstantWind:
    """constant wind velocity that changes everytime it is reseted"""
    def __init__(self) -> None:
        self.value = self._reset()

    def _reset(self) -> float:
        return np.random.uniform(low=8.0, high=12.0, size=(1,))[0]

    def read(self):
        return self.value

    def reset(self):
        self.value = self._reset()
        return self.value


class DatasetConstantWind:
    value: float

    """
    constant wind velocity that changes everytime it is reseted
    Only values between 8 and 12 m/s are returned
    """
    def __init__(self) -> None:
        path = Path(__file__).parent / 'datasets/10min.csv'
        df = pd.read_csv(path)
        self.__values = df.query('windspeed >= 8 and windspeed <= 12').windspeed.to_numpy()

        # just to not get undefined values
        self.value = self.__reset()

    def __reset(self):
        """return a random value from dataset"""
        return np.random.choice(self.__values)

    def read(self):
        return self.value

    def reset(self):
        self.value = self.__reset()
        return self.value


def split_in_windows(a: np.ndarray, func: Callable[[int], bool]):
    """
    returns array a split in windows which satisfy func
    """
    mask = func(a)
    idx = np.where(mask[:-1] != mask[1:])[0]
    splits = np.split(a, idx+1)
    is_odd = mask[0]
    return splits[0::2] if is_odd else splits[1::2]


class DatasetWind:
    value: float
    windows: List[np.ndarray]
    window: np.ndarray

    def __init__(self, duration: float, dt: float) -> None:
        self.duration = duration
        self.dt = dt

        path = Path(__file__).parent / 'datasets/10min.csv'
        df = pd.read_csv(path)
        windspeed = df.windspeed.to_numpy()
        all_windows = split_in_windows(windspeed, lambda x: (x >= 8) & (x <= 12))
        self.windows = list(filter(lambda x: x.shape[0] >= 32, all_windows))

        self.i = 0              # time
        self.window = random.choice(self.windows)
        self.ptr = 0            # position in window

    def read(self):
        self.i += 1

        if self.ptr < self.window.shape[0]-1 and self.i*self.dt >= self.duration:
            self.i = 0
            self.ptr += 1

        return self.window[self.ptr]

    def reset(self):
        # self.value = self.__reset()
        self.i = 0
        self.window = random.choice(self.windows)
        self.ptr = 0            # position in window
        return self.window[self.ptr]


class RandomStepsWind:
    """random value step signals every 100 seconds"""
    def __init__(self, duration: float, dt: float) -> None:
        self.duration = duration
        self.dt = dt
        self._reset()

    def _reset(self):
        self.i = 0
        self.value: float = np.random.uniform(low=8.0, high=12.0, size=(1,))[0]

    def read(self):
        self.i += 1
        if self.i*self.dt >= self.duration:
            self._reset()

        return self.value

    def reset(self):
        self._reset()
        return self.value

if __name__ == "__main__":
    # a = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    # a = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1,1,1,1,1,1])
    # res = split_in_windows(a, lambda x: x > 0)
    # print(res)

    path = Path(__file__).parent / 'datasets/10min.csv'
    df = pd.read_csv(path)
    windspeed_data = df.windspeed.to_numpy()
    print(windspeed_data.shape)
    windows = split_in_windows(windspeed_data, lambda x: (x >= 8) & (x <= 12))
    # print(len(list(filter(lambda x: x.shape[0] >= 40, windows))))
    print(max([w.shape[0] for w in windows]))

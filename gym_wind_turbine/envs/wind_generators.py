from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import numpy as np
import csv
import random


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

    """constant wind velocity that changes everytime it is reseted"""
    def __init__(self) -> None:
        path = Path(__file__).parent / 'datasets/10min.csv'
        with path.open() as f:
            reader = csv.reader(f)
            reader.__next__()   # skip headers
            values = [float(row[1]) for row in reader]

        self.__values = values
        # just to not get undefined values
        self.value = self.__reset()

    def __reset(self):
        return random.choice(self.__values)

    def read(self):
        return self.value

    def reset(self):
        self.value = self.__reset()
        return self.value


class DatasetWind:
    value: float

    def __init__(self, duration: float, dt: float) -> None:
        self.duration = duration
        self.dt = dt

        path = Path(__file__).parent / 'datasets/10min.csv'
        with path.open() as f:
            reader = csv.reader(f)
            reader.__next__()   # skip headers
            values = [float(row[1]) for row in reader]

        self.i = 0
        self.__values = values
        self.ptr = random.randrange(0, len(self.__values), 1)
        # just to not get undefined values
        # self.value = self.__reset()

    def read(self):
        self.i += 1

        if self.ptr < len(self.__values) and self.i*self.dt >= self.duration:
            self.i = 0
            self.ptr += 1

        return self.__values[self.ptr]

    def reset(self):
        # self.value = self.__reset()
        self.i = 0
        self.ptr = random.randrange(0, len(self.__values), 1)
        return self.__values[self.ptr]


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

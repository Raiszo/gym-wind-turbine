from gym_wind_turbine.envs.wind_generators import ConstantWind, DatasetConstantWind, DatasetWind, RandomConstantWind, RandomStepsWind
from gym_wind_turbine.envs.wind_turbine_analytical import DriveTrain, Rotor
from gym.envs.registration import register

windey_rotor = Rotor(rho=1.25, R=38.5, beta=0)
windey_drivetrain = DriveTrain(
    n_gear= 105.494,
    I_rotor= 4456761.0,
    I_gen= 123.0,
    K_rotor= 45.52,
    K_gen= 0.4,
)

const_wind = ConstantWind(11.0)
random_const_wind = RandomConstantWind()
dataset_const_wind = DatasetConstantWind()

# initial conditions are set
# constant wind constant wind speed
register(
    id='WindTurbine-analytical-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(180.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
        'wind_generator': const_wind,
    }
)

# just 60 seconds
# cw stands for constant wind
register(
    id='WindTurbine-cw-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(90.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
        'wind_generator': const_wind,
    }
)

# initial conditions are set
# wind speed is initialized randomly
# rcw stands for random constant wind
register(
    id='WindTurbine-rcw-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(90.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
        'wind_generator': random_const_wind,
    }
)

# Dataset Constant Wind, a value of wind speed is chosen from a dataset
register(
    id='WindTurbine-dcw-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(90.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
        'wind_generator': dataset_const_wind,
    }
)

# Dataset wind that changes every 30 seconds
register(
    id='WindTurbine-dsw-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(30 * 30.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
        'wind_generator': DatasetWind(duration=30, dt=0.05),
    }
)

# wind velocity changes every 50 seconds as step signals
register(
    id='WindTurbine-rsw-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(240.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
        'wind_generator': RandomStepsWind(duration=50, dt=0.05),
    }
)

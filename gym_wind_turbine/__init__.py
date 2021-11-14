from gym_wind_turbine.envs.wind_turbine_analytical import ConstantWind, DriveTrain, RandomConstantWind, Rotor
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

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

# initial conditions are set
# constant wind constant wind speed
register(
    id='WindTurbine-analytical-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(180.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
    }
)

# initial conditions are set
# wind speed is initialized randomly
register(
    id='WindTurbine-random-v0',
    entry_point='gym_wind_turbine.envs:WindTurbineAnalytical',
    max_episode_steps=int(60.0/0.05),
    kwargs={
        'rotor': windey_rotor,
        'drive_train': windey_drivetrain,
    }
)

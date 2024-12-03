from stable_baselines3.common.utils import get_linear_fn

# Define a Global Dictionary to consider during the development of this Project
CONFIG = {
    'MIN_FUEL': 50,            # Minimum Fuel Level of the SpaceCraft
    'MAX_FUEL': 100,           # Maximum Fuel Level of the SpaceCraft
    'N_ENVS': 4,               # Number of Environments to consider when training
    'N_ITERATIONS': 150_000,   # Number of Steps / Iterations to consider during Training
    'N_EPISODES': 10           # Number of Episodes to Consider
}

PATHS_CONFIG = {
    'PPO':{
        'P1': './ExperimentalResults/PPO/P1/'
    },
    'DQN':{
        'P1': './ExperimentalResults/DQN/P1/'
    },
    # A2C, ACER - MAYBE
}

# Define the Settings for the PPO Algorithm
PPO_SETTINGS_1 = {
    'n_steps': 2048,
    'batch_size': 64,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 10,
    'ent_coef': 0.001,
    'learning_rate': get_linear_fn(start=2.5e-4, end=0, end_fraction=1),
    'clip_range': get_linear_fn(start=0.2, end=0, end_fraction=1),
}

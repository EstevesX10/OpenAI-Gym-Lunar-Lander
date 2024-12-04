from stable_baselines3.common.utils import get_linear_fn

# Define a Global Dictionary to consider during the development of this Project
CONFIG = {
    'MIN_FUEL': 5,                # Minimum Fuel Level of the SpaceCraft
    'MAX_FUEL': 10,               # Maximum Fuel Level of the SpaceCraft
    'N_ENVS': 4,                  # Number of Environments to consider when training
    'N_ITERATIONS': 5_000_000,      # Number of Steps / Iterations to consider during Training
    'N_EPISODES': 10              # Number of Episodes to Consider
}

PATHS_CONFIG = {
    'OriginalEnvironment': {
        'PPO':{
            'Settings-1':'./ExperimentalResults/OriginalEnvironment/PPO/Settings-1/',
            'Settings-2':'./ExperimentalResults/OriginalEnvironment/PPO/Settings-2/'
        },
        'DQN':{
            'Settings-1':'./ExperimentalResults/OriginalEnvironment/DQN/Settings-1/',
        },
    },
    'CustomEnvironment': {
        'PPO':{
            'Settings-1':'./ExperimentalResults/CustomEnvironment/PPO/Settings-1/',
            'Settings-2':'./ExperimentalResults/CustomEnvironment/PPO/Settings-2/'
        },
        'DQN':{
            'Settings-1':'./ExperimentalResults/CustomEnvironment/DQN/Settings-1/',
        },
    },
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

# Define the Settings for the DQN Algorithm
# DQN_SETTING_1 = {
#     # 'n_timesteps': 1e5,
#     'learning_rate':  6.3e-4,
#     'batch_size': 128,
#     'buffer_size': 50000,
#     'learning_starts': 0,
#     'gamma': 0.99,
#     'target_update_interval': 250,
#     'train_freq': 4,
#     'gradient_steps': -1,
#     'exploration_fraction': 0.12,
#     'exploration_final_eps': 0.1,
#     'policy_kwargs': dict(net_arch=[256, 256]),
# }

DQN_SETTING_1 = {
    'learning_rate':1e-3,
    'buffer_size':100_000,
    'learning_starts':10_000,  # Minimum steps before training begins
    'batch_size':64,
    'tau':1.0,  # Soft update coefficient (Polyak averaging)
    'gamma':0.99,
    'train_freq':4,  # Train every 4 steps
    'target_update_interval':10_000,  # Target network update frequency
    'exploration_initial_eps':1.0,
    'exploration_final_eps':0.01,
    'exploration_fraction':0.1,  # Fraction of timesteps to decay epsilon
    'max_grad_norm':10.0,
    'policy_kwargs':dict(net_arch=[128, 128]),
}
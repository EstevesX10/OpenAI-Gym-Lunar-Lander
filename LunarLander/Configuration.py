from stable_baselines3.common.utils import get_linear_fn

# Define a Global Dictionary to consider during the development of this Project
CONFIG = {
    "MIN_FUEL": 9,  # Minimum Fuel Level of the SpaceCraft
    "MAX_FUEL": 13,  # Maximum Fuel Level of the SpaceCraft
    "N_ENVS": 4,  # Number of Environments to consider when training
    "N_ITERATIONS": 30_000_000,  # Number of Steps / Iterations to consider during Training
    "N_EPISODES": 10,  # Number of Episodes to Consider
}

PATHS_CONFIG = {
    "OriginalEnvironment": {
        "PPO": {
            "Settings-1": "./ExperimentalResults/OriginalEnvironment/PPO/Settings-1/",
            "Settings-2": "./ExperimentalResults/OriginalEnvironment/PPO/Settings-2/",
        },
        "DQN": {
            "Settings-1": "./ExperimentalResults/OriginalEnvironment/DQN/Settings-1/",
        },
    },
    "CustomEnvironment": {
        "PPO": {
            "Settings-1": "./ExperimentalResults/CustomEnvironment/PPO/Settings-1/",
            "Settings-2": "./ExperimentalResults/CustomEnvironment/PPO/Settings-2/",
        },
        "DQN": {
            "Settings-1": "./ExperimentalResults/CustomEnvironment/DQN/Settings-1/",
        },
    },
}

# Define the Settings for the PPO Algorithm
PPO_SETTINGS_1 = {
    "n_steps": 2048,
    "batch_size": 64,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "n_epochs": 10,
    "ent_coef": 0.001,
    "learning_rate": get_linear_fn(start=2.5e-4, end=0, end_fraction=1),
    "clip_range": get_linear_fn(start=0.2, end=0, end_fraction=1),
}

PPO_SETTINGS_2 = {
    "n_steps": 4096,            # Increase the number of steps per rollout
    "batch_size": 128,          # Use a larger batch size
    "gae_lambda": 0.98,         # Slightly increase GAE lambda for smoother advantage estimation
    "gamma": 0.999,             # Higher discount factor for long-term rewards
    "n_epochs": 15,             # More training epochs per update
    "ent_coef": 0.01,           # Increase entropy coefficient to encourage exploration
    "learning_rate": get_linear_fn(start=3e-4, end=1e-5, end_fraction=1),   # Higher initial learning rate with a smaller end
    "clip_range": get_linear_fn(start=0.3, end=0.1, end_fraction=1),        # Larger clip range for stability
}

# Define the Settings for the DQN Algorithm
DQN_SETTING_1 = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "learning_starts": 10_000,          # Minimum steps before training begins
    "batch_size": 64,
    "tau": 1.0,                         # Soft update coefficient (Polyak averaging)
    "gamma": 0.99,
    "train_freq": 4,                    # Train every 4 steps
    "target_update_interval": 10_000,   # Target network update frequency
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,
    "exploration_fraction": 0.1,        # Fraction of timesteps to decay epsilon
    "max_grad_norm": 10.0,
    "policy_kwargs": dict(net_arch=[128, 128]),
}

DQN_SETTING_2 = {
    "learning_rate": 5e-4,              # Lower learning rate for stability
    "buffer_size": 500_000,             # Larger replay buffer
    "learning_starts": 5_000,           # Start training earlier
    "batch_size": 64,                   # Keep batch size unchanged
    "tau": 0.01,                        # Use soft target updates
    "gamma": 0.95,                      # Focus more on immediate rewards
    "train_freq": 2,                    # Train every 2 steps
    "target_update_interval": 20_000,   # Less frequent target updates
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,      # Decay to a slightly higher final epsilon
    "exploration_fraction": 0.2,        # Extend exploration phase
    "max_grad_norm": 5.0,               # Clip gradients more aggressively
    "policy_kwargs": dict(net_arch=[256, 256]),  # More complex neural network
}

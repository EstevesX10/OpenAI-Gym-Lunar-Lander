from typing import Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EveryNTimesteps,
    CallbackList,
    EvalCallback,
)

from Environment import MyLunarLander
from Configuration import CONFIG, PATHS_CONFIG


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.last_saved_step = 0
        self.checkFolder(self.save_path)

    def checkFolder(self, path: str) -> None:
        folderPath = Path(path)
        folderPath.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self) -> None:
        # Initialize last_saved_step to the current num_timesteps
        self.last_saved_step = self.num_timesteps

    def _on_step(self) -> bool:
        # Check if the model has crossed the next saving threshold
        next_save_step = (
            (self.last_saved_step // self.check_freq) + 1
        ) * self.check_freq

        if self.num_timesteps >= next_save_step:
            model_path = os.path.join(
                self.save_path, f"model_step_{next_save_step}.zip"
            )
            self.model.save(model_path)
            self.last_saved_step = next_save_step
            if self.verbose > 0:
                print(f"Saved model at {model_path}")
        return True


class LunarLanderManager:
    def __init__(
        self,
        environmentName: str,
        algorithm: str,
        settingsNumber: int,
        algorithmSettings: dict,
    ) -> None:
        """
        # Description
            -> Constructor of the class LunarLanderManager. It helps instanciate any
            object of the class LunarLanderManager. This class helps train and test
            a given Reinforcemente Learning Algorithm upon a selected Environment.
        ----------------------------------------------------------------------------
        := param: environmentName - Name of the Environment to Use (either 'LunarLander' or 'MyLunarLander').
        := param: algorithm - RL Algorithm to use (either 'PPO' or 'DQN').
        := param: settingsNumber - Identification Number of the Settings to use.
        := param: algorithmSettings - Dictionary with the settings to use for the RL Algorithm.
        := return: None, since we are simply creating a instance of the Class.
        """

        # Store the given parameters
        self.envName = environmentName
        self.algorithm = algorithm
        self.settingsNumber = settingsNumber
        self.algorithmSettings = algorithmSettings

        # Define the Policy
        self.policy = "MlpPolicy"

        # Define the interval in which to save the model
        self.savingInterval = 100_000

        # Check if the environment selected is valid
        if self.envName in ["LunarLander", "MyLunarLander"]:
            # Store the name of the Environment (Original or Custom)
            self._envVersion = (
                "OriginalEnvironment"
                if self.envName == "LunarLander"
                else "CustomEnvironment"
            )
        else:
            # Invalid Environment Selected
            raise ValueError("Invalid Lunar Lander Environment selected!")

        # Check if the given algorithm is valid
        if self.algorithm not in ["PPO", "DQN"]:
            raise ValueError(
                "Invalid Algorithm selected! (Please choose between PPO or DQN)"
            )

        # Define the Folder in which to store the results
        self.resultsFolder = PATHS_CONFIG[self._envVersion][self.algorithm][
            f"Settings-{self.settingsNumber}"
        ]

    def isResultsFolderEmpty(self) -> bool:
        """
        # Description
            -> This method allows to check if a folder with the Model's experimental is empty or not.
        ---------------------------------------------------------------------------------------------
        := return: Boolean value that dictates whether or not the results folder is empty or not.
        """

        # Convert folder path to Path object and check if empty
        folder = Path(self.resultsFolder)

        # Check if it is empty
        return folder.exists() and folder.is_dir() and not any(folder.iterdir())

    def hasComputedStep(self, step: int) -> bool:
        """
        # Description
            -> This method helps determine if the model has computed a certain step.
        ----------------------------------------------------------------------------
        := param: step - Step to check if the model has overcome.
        := return: Boolean value that determines if the model has already computed a given step S.
        """

        # Define the Model Iteration that we want to check it exists
        file_name = f"model_step_{step}.zip"

        # Check if the folder exists and if the specific file is in it
        return os.path.exists(self.resultsFolder) and file_name in os.listdir(
            self.resultsFolder
        )

    def getLastestTrainedModelPath(self) -> Tuple[str, int]:
        """
        # Description
            -> This method helps obtain the lastest computed model path
            and the corresponding step.
        ----------------------------------------------------------------
        := return: Tuple containing the path to the most recent model and it's training step.
        """

        # Define a variable to store the path to the lastest model
        modelPath = None

        # Define a variable for the latest performed step
        lastestStep = 0

        for step in range(
            self.savingInterval, CONFIG["N_ITERATIONS"] + 1, self.savingInterval
        ):
            if self.hasComputedStep(step=step):
                # Update the model path
                modelPath = self.resultsFolder + f"model_step_{step}.zip"

                # Update the latest step
                lastestStep = step

        # Return fetched values
        return modelPath, lastestStep

    def train(self) -> Union[PPO, DQN]:
        """
        # Description
            -> This method allows to train a selected Reinforcement Learning Algorithm
            on a selected Environment according to the CONFIG Dictionary defined in the Configuration.py file.
        ------------------------------------------------------------------------------------------------------
        := return: Trained Model.
        """

        def registerAndMake():
            gym.register(
                id="MyLunarLander",
                entry_point=MyLunarLander,
            )
            return gym.make(self.envName)

        # Create a Environment
        envs = make_vec_env(
            registerAndMake, n_envs=CONFIG["N_ENVS"], seed=0, vec_env_cls=SubprocVecEnv
        )

        # Create a separate evaluation environment
        evalEnv = gym.make(self.envName)

        # Fetch the latest model computed and it's corresponding step
        lastestModelPath, latestModelStep = self.getLastestTrainedModelPath()

        # Check if the current configuration of the model has already been computed
        if lastestModelPath is None:
            # Define the Reinforcement Learning Model
            if self.algorithm == "PPO":
                model = PPO(
                    policy=self.policy,
                    env=envs,
                    device="cpu",
                    verbose=1,
                    **self.algorithmSettings,
                )
            else:
                model = DQN(
                    policy=self.policy,
                    env=envs,
                    device="cpu",
                    verbose=1,
                    **self.algorithmSettings,
                )
        else:
            # The model can be further trained
            if self.algorithm == "PPO":
                model = PPO.load(path=lastestModelPath, env=envs)
            else:
                model = DQN.load(path=lastestModelPath, env=envs)

            # Already trained the model for the designated duration
            if latestModelStep >= CONFIG["N_ITERATIONS"]:
                print(
                    "Already Trained the Model over the established TimeSteps (Defined inside the CONFIG Dictionary)."
                )

                # Close the Environments
                envs.close()
                evalEnv.close()

                # Return the Model
                return model

        # Set the callback's internal state to align with the loaded model's step count
        checkpointCallback = CustomCheckpointCallback(
            check_freq=self.savingInterval, save_path=self.resultsFolder
        )
        checkpointCallback.num_timesteps = latestModelStep
        eventCallback = EveryNTimesteps(
            n_steps=self.savingInterval, callback=checkpointCallback
        )
        eventCallback.n_calls = latestModelStep // self.savingInterval

        # Define the evaluation callback
        evalCallback = EvalCallback(
            evalEnv,
            best_model_save_path=self.resultsFolder + "/bestModel",
            log_path=self.resultsFolder + "/evalLogs",
            eval_freq=12_500,
            deterministic=True,
            render=False,
        )

        # Combine callbacks using CallbackList
        combinedCallbacks = CallbackList([evalCallback, eventCallback])

        try:
            # Train the Model
            model.learn(
                CONFIG["N_ITERATIONS"] - latestModelStep,
                callback=combinedCallbacks,
                reset_num_timesteps=False,
            )
        finally:
            # Close the Environment
            envs.close()
            evalEnv.close()

        # Return trained model
        return model

    def test(
        self, model: Union[PPO, DQN] = None, numEpisodes: int = CONFIG["N_EPISODES"]
    ) -> None:
        """
        # Description
            -> This method allows to test a given model upon the selected Environment.
        ------------------------------------------------------------------------------
        := param: model - Trained Model.
        := param: numEpisodes - Number of Episodes to Consider.
        := return: None, since we are simply testing the model.
        """

        # Define a Environment
        env = gym.make(self.envName, render_mode="human")

        # Check if a model was given
        if model is None:
            # Define Model Path
            bestModelPath = f"./ExperimentalResults/{self._envVersion}/{self.algorithm}/Settings-{self.settingsNumber}/bestModel/best_model.zip"

            # PPO Algorithm
            if self.algorithm == "PPO":
                model = PPO.load(path=bestModelPath, env=env)
            # DQN Algorithm
            elif self.algorithm == "DQN":
                model = DQN.load(path=bestModelPath, env=env)
            else:
                # Close the Environment
                env.close()
                # Raise Error
                raise ValueError("Unsupported Algorithm Chosen!")

        # Perform N Episodes
        for ep in range(numEpisodes):
            obs, info = env.reset()
            trunc = False
            while not trunc:
                # pass observation to model to get predicted action
                action, _states = model.predict(obs, deterministic=True)

                # pass action to env and get info back
                obs, rewards, trunc, done, info = env.step(action)

                # show the environment on the screen
                env.render()
                # print(ep, rewards, trunc)
                # print("---------------")

        # Close the Environment
        env.close()

    def testRandomAction(self) -> None:
        """
        # Description
            -> This method allows to randomly test a given model upon the selected Environment.
        ---------------------------------------------------------------------------------------
        := return: None, since we are simply testing the model.
        """

        # Create a new instance of the Environment
        env = gym.make(self.envName, render_mode="human")

        # Reset the Environment - To get the initial observation
        observation, info = env.reset()

        # Define a flag to determine if the episode is over or not
        episode_over = False

        # PerfORM a Episode
        while not episode_over:
            # Choose a random action
            action = (
                env.action_space.sample()
            )  # agent policy that uses the observation and info

            # Perform a Action / Step
            observation, reward, terminated, truncated, info = env.step(action)

            # Update if the episode is over
            episode_over = terminated or truncated

        # Close the Environment
        env.close()

    def checkResults(self) -> None:
        """
        # Description
            -> This method aims to evaluate the performance of the model after training.
        --------------------------------------------------------------------------------
        := return: None, since we are only plotting data.
        """

        # Define the path to the results
        resultsPath = f"./ExperimentalResults/{self._envVersion}/{self.algorithm}/Settings-{self.settingsNumber}/evalLogs/evaluations.npz"

        # Check if the file exists
        if not os.path.exists(resultsPath):
            # The results have not been computed
            raise ValueError("The Results have not yet been Computed!")

        # Creating the figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 3 rows, 1 column

        # Load the collected data
        data = np.load(resultsPath)

        # Extract mean and standard deviation for rewards
        meanRewards = np.mean(data["results"], axis=1)
        stdRewards = np.std(data["results"], axis=1)

        # Extract mean and standard deviation for episode lengths
        meanEpisodeLengths = np.mean(data["ep_lengths"], axis=1)
        stdEpisodeLengths = np.std(data["ep_lengths"], axis=1)

        # Plot Rewards with standard deviation
        axs[0].plot(data["timesteps"], meanRewards, label="Reward")
        axs[0].fill_between(
            data["timesteps"],
            meanRewards - stdRewards,
            meanRewards + stdRewards,
            color="blue",
            alpha=0.2,
            label="Std Dev",
        )
        axs[0].set_title("Reward")
        axs[0].set_xlabel("TimeSteps")
        axs[0].set_ylabel("Reward")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Episode Lengths with standard deviation
        axs[1].plot(
            data["timesteps"], meanEpisodeLengths, label="Episode Lengths", color="red"
        )
        axs[1].fill_between(
            data["timesteps"],
            meanEpisodeLengths - stdEpisodeLengths,
            meanEpisodeLengths + stdEpisodeLengths,
            color="red",
            alpha=0.2,
            label="Std Dev",
        )
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("TimeSteps")
        axs[1].set_ylabel("Episode Lengths")
        axs[1].legend()
        axs[1].grid(True)

        fig.suptitle(
            f"[{self._envVersion}] {self.algorithm} Performance Evaluation", fontsize=16
        )

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        plt.show()

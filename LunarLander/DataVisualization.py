from typing import (List, Tuple)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def pastelizeColor(c:tuple, weight:float=None) -> np.ndarray:
    """
    # Description
        -> Lightens the input color by mixing it with white, producing a pastel effect.
    -----------------------------------------------------------------------------------
    := param: c - Original color.
    := param: weight - Amount of white to mix (0 = full color, 1 = full white).
    """

    # Set a default weight
    weight = 0.5 if weight is None else weight

    # Initialize a array with the white color values to help create the pastel version of the given color
    white = np.array([1, 1, 1])

    # Returns a tuple with the values for the pastel version of the color provided
    return mcolors.to_rgba((np.array(mcolors.to_rgb(c)) * (1 - weight) + white * weight))

def generatePastelCmap(baseColors:list, numberColors:int, weight:float=0.5) -> List:
    """
    # Description
        -> Generate a list of n pastelized colors.
    ----------------------------------------------
    := param: baseColors - List of base colors to be pastelized.
    := param: numberColors - Number of colors to generate.
    := param: weight - The weight of the pastel effect (0 = no pastel, 1 = full white).
    := return: List of pastel colors.
    """
    # If baseColors is shorter than numberColors, interpolate it
    if len(baseColors) < numberColors:
        baseColors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, numberColors))
    else:
        baseColors = baseColors[:numberColors]
    
    # Apply the pastelizeColor function to each base color
    pastelColors = [pastelizeColor(c, weight) for c in baseColors]

    # Return the list with the pastel colors
    return pastelColors


def plotTrainingResultsEnvironment(results:Tuple[str, str, List[np.lib.npyio.NpzFile]]):
    """
    # Description
        -> This function aims to plot the best model results on a given Environment.
    --------------------------------------------------------------------------------
    := param: results - Tuple composed by the env name and algorithm alongside the List with all the evallogs obtained for the trained models.
    := return: None, since we are only plotting results.
    """
    
    # Get the number of results
    numberResults = len(results)

    # Create a pastel cmap based on the amount of the results for both the rewards and the average episode length
    rewardsColorMap = generatePastelCmap(baseColors=[plt.cm.Paired(5//(i + 1)) for i in range(numberResults)], numberColors=numberResults, weight=0)
    episodeLengthColorMap = generatePastelCmap(baseColors=[plt.cm.Accent(5//(i + 1)) for i in range(numberResults)], numberColors=numberResults, weight=0)

    # Creating the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Iterate through the results
    for idx, (algorithm, _, data) in enumerate(results):
        # Extract mean for rewards
        meanRewards = np.mean(data['results'], axis=1)

        # Extract mean for episode lengths
        meanEpisodeLengths = np.mean(data['ep_lengths'], axis=1)

        # Get the pastel color for the current plots
        rewardsColor = rewardsColorMap[idx]
        episodeLengthColor = episodeLengthColorMap[idx]

        # Plot Rewards with standard deviation
        axs[0].plot(data['timesteps'], meanRewards, label=f"{algorithm}", alpha=0.7, color=rewardsColor)
        axs[0].set_title("Reward")
        axs[0].set_xlabel("TimeSteps")
        axs[0].set_ylabel("Reward")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Episode Lengths with standard deviation
        axs[1].plot(data['timesteps'], meanEpisodeLengths, label=f"{algorithm}", alpha=0.7, color=episodeLengthColor)
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("TimeSteps")
        axs[1].set_ylabel("Episode Lengths")
        axs[1].legend()
        axs[1].grid(True)

    # Add a main title to the plot
    fig.suptitle(f"[{results[0][1]} Environment] Performance Evaluation", fontsize=16)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

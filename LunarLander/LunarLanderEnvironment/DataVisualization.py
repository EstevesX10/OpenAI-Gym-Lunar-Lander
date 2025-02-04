from typing import (List, Tuple, Dict)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Utils.Configuration import CONFIG

import scikit_posthocs as sp
from scipy import stats


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

def plotModelsTrainingPerformance(results:Tuple[str, str, str, List[np.lib.npyio.NpzFile]]) -> None:
    """
    # Description
        -> This function aims to plot the best model results on a given Environment.
    --------------------------------------------------------------------------------
    := param: results - Tuple composed by the env name, algorithm and settings alongside the List with all the evallogs obtained for the trained models.
    := return: None, since we are only plotting results.
    """
    
    # Get the number of results
    numberResults = len(results)

    # Create a Custom Color Map
    customColorMap = ['#29599c', '#f66b6e', '#4cb07a']

    # Verify if there are enough colors
    if (numberResults > len(customColorMap)):
        raise ValueError("Not enough colors! Please adapt the implementation")

    # Creating the figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))

    # Iterate through the results
    for idx, (algorithm, _, _, data) in enumerate(results):
        # Extract mean for rewards
        meanRewards = np.mean(data['results'], axis=1)

        # Extract mean for episode lengths
        meanEpisodeLengths = np.mean(data['ep_lengths'], axis=1)

        # Get the pastel color for the current plots
        rewardsColor = customColorMap[idx]
        episodeLengthColor = customColorMap[idx]

        # Plot Rewards with standard deviation
        axs[0].plot(data['timesteps'], meanRewards, label=f"{algorithm}", alpha=0.7, color=rewardsColor)
        axs[0].set_title("Reward")
        axs[0].set_xlabel("TimeSteps")
        axs[0].set_ylabel("Reward")
        axs[0].set_ylim(-500, 500)

        axs[0].legend()
        axs[0].grid(alpha=0.4, linestyle='dashed')

        # Plot Episode Lengths with standard deviation
        axs[1].plot(data['timesteps'], meanEpisodeLengths, label=f"{algorithm}", alpha=0.7, color=episodeLengthColor)
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("TimeSteps")
        axs[1].set_ylabel("Episode Lengths")
        axs[1].legend()
        axs[1].grid(alpha=0.4, linestyle='dashed')

        # Plot success rate
        successRate = (data["results"] > CONFIG["SUCCESS_THRESHOLD"]).sum(
            axis=1
        ) / data["results"].shape[1]
        axs[2].plot(successRate, label=f"{algorithm}", alpha=0.7, color=customColorMap[idx])
        axs[2].set_title("Success Rate")
        axs[2].set_xlabel("EvaluationSteps")
        axs[2].set_ylabel("Success Rate")
        axs[2].legend()
        axs[2].grid(alpha=0.4, linestyle="dashed")

    # Add a main title to the plot
    fig.suptitle(f"[{results[0][1]} Environment] Performance Evaluation", fontsize=16)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

def plotModelSettingsPerformance(results:Tuple[str, str, str, List[np.lib.npyio.NpzFile]]) -> None:
    """
    # Description
        -> This function aims to plot the performance of a given algorithm under multiple settings.
    -----------------------------------------------------------------------------------------------
    := param: results - Tuple composed by the env name, algorithm and setting alongside the List with all the evallogs obtained for the trained models.
    := return: None, since we are only plotting results.
    """
    
    # Get the number of results
    numberResults = len(results)

    # Create a Custom Color Map
    customColorMap = ['#29599c', '#f66b6e', '#4cb07a']

    # Verify if there are enough colors
    if (numberResults > len(customColorMap)):
        raise ValueError("Not enough colors! Please adapt the implementation")

    # Creating the figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))

    # Iterate through the results
    for idx, (_, _, setting, data) in enumerate(results):
        # Extract mean for rewards
        meanRewards = np.mean(data['results'], axis=1)

        # Extract mean for episode lengths
        meanEpisodeLengths = np.mean(data['ep_lengths'], axis=1)

        # Plot Rewards with standard deviation
        axs[0].plot(data['timesteps'], meanRewards, label=f"{setting}", alpha=0.7, color=customColorMap[idx])
        axs[0].set_title("Reward")
        axs[0].set_xlabel("TimeSteps")
        axs[0].set_ylabel("Reward")
        axs[0].set_ylim(-500, 500)

        axs[0].legend()
        axs[0].grid(alpha=0.4, linestyle='dashed')

        # Plot Episode Lengths with standard deviation
        axs[1].plot(data['timesteps'], meanEpisodeLengths, label=f"{setting}", alpha=0.7, color=customColorMap[idx])
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("TimeSteps")
        axs[1].set_ylabel("Episode Lengths")
        axs[1].legend()
        axs[1].grid(alpha=0.4, linestyle='dashed')

        # Plot success rate
        successRate = (data["results"] > CONFIG["SUCCESS_THRESHOLD"]).sum(
            axis=1
        ) / data["results"].shape[1]
        axs[2].plot(successRate, label=f"{setting}", alpha=0.7, color=customColorMap[idx])
        axs[2].set_title("Success Rate")
        axs[2].set_xlabel("EvaluationSteps")
        axs[2].set_ylabel("Success Rate")
        axs[2].legend()
        axs[2].grid(alpha=0.4, linestyle="dashed")

    # Add a main title to the plot
    fig.suptitle(f"[{results[0][1]} Environment] {results[0][0]} Settings Performance Evaluation", fontsize=16)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

def plotModelsOverallPerformances(originalEnvResults:Tuple[str, str, List[np.lib.npyio.NpzFile]], customEnvResults:Tuple[str, str, List[np.lib.npyio.NpzFile]]) -> None:
    """
    # Description
        -> This function aims to plot the overall performances of the used algorithms under the respective settings and environments.
    ---------------------------------------------------------------------------------------------------------------------------------
    := param: originalEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the original Environment.
    := param: customEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the custom Environment.
    := return: None, since we are only plotting results.
    """
    
    # Get the number of results
    numberResults = max(len(originalEnvResults), len(customEnvResults))

    # Create a Custom Color Map
    customColorMap = ['#29599c', '#f66b6e', '#4cb07a', '#f8946c']

    # Verify if there are enough colors
    if (numberResults > len(customColorMap)):
        raise ValueError("Not enough colors! Please adapt the implementation")

    # Creating the figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(17, 8))

    # Then manually add more space at the top and between rows
    fig.subplots_adjust(top=0.18, hspace=0.4)

    # Add a title for the top row (Original Environment)
    fig.text(0.5, 0.98, "[Original Environment]", ha='center', va='center', fontsize=14, fontweight='bold')

    # Add a title for the bottom row (Custom Environment)
    fig.text(0.5, 0.48, "[Custom Environment]", ha='center', va='center', fontsize=14, fontweight='bold')

    # Iterate through the original environment results
    for idx, (model, setting, data) in enumerate(originalEnvResults):
        # Extract mean for rewards
        meanRewards = np.mean(data['results'], axis=1)

        # Extract mean for episode lengths
        meanEpisodeLengths = np.mean(data['ep_lengths'], axis=1)

        # Plot Rewards with standard deviation
        axs[0, 0].plot(data['timesteps'], meanRewards, label=f"[{model}] {setting}", alpha=0.7, color=customColorMap[idx])
        axs[0, 0].set_title("Reward")
        axs[0, 0].set_xlabel("TimeSteps")
        axs[0, 0].set_ylabel("Reward")
        axs[0, 0].set_ylim(-500, 500)
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.4, linestyle='dashed')

        # Plot Episode Lengths with standard deviation
        axs[0, 1].plot(data['timesteps'], meanEpisodeLengths, label=f"[{model}] {setting}", alpha=0.7, color=customColorMap[idx])
        axs[0, 1].set_title("Episode Lengths")
        axs[0, 1].set_xlabel("TimeSteps")
        axs[0, 1].set_ylabel("Episode Lengths")
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.4, linestyle='dashed')

        # Plot success rate
        successRate = (data["results"] > CONFIG["SUCCESS_THRESHOLD"]).sum(
            axis=1
        ) / data["results"].shape[1]
        axs[0, 2].plot(successRate, label=f"[{model}] {setting}", alpha=0.7, color=customColorMap[idx])
        axs[0, 2].set_title("Success Rate")
        axs[0, 2].set_xlabel("EvaluationSteps")
        axs[0, 2].set_ylabel("Success Rate")
        axs[0, 2].legend()
        axs[0, 2].grid(alpha=0.4, linestyle="dashed")

    # Iterate through the custom environment results
    for idx, (model, setting, data) in enumerate(customEnvResults):

        # Extract mean for rewards
        meanRewards = np.mean(data['results'], axis=1)

        # Extract mean for episode lengths
        meanEpisodeLengths = np.mean(data['ep_lengths'], axis=1)

        # Plot Rewards with standard deviation
        axs[1, 0].plot(data['timesteps'], meanRewards, label=f"[{model}] {setting}", alpha=0.7, color=customColorMap[idx])
        axs[1, 0].set_title("Reward")
        axs[1, 0].set_xlabel("TimeSteps")
        axs[1, 0].set_ylabel("Reward")
        axs[1, 0].set_ylim(-500, 500)
        axs[1, 0].legend()
        axs[1, 0].grid(alpha=0.4, linestyle='dashed')

        # Plot Episode Lengths with standard deviation
        axs[1, 1].plot(data['timesteps'], meanEpisodeLengths, label=f"[{model}] {setting}", alpha=0.7, color=customColorMap[idx])
        axs[1, 1].set_title("Episode Lengths")
        axs[1, 1].set_xlabel("TimeSteps")
        axs[1, 1].set_ylabel("Episode Lengths")
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.4, linestyle='dashed')

        # Plot success rate
        successRate = (data["results"] > CONFIG["SUCCESS_THRESHOLD"]).sum(
            axis=1
        ) / data["results"].shape[1]
        axs[1, 2].plot(successRate, label=f"[{model}] {setting}", alpha=0.7, color=customColorMap[idx])
        axs[1, 2].set_title("Success Rate")
        axs[1, 2].set_xlabel("EvaluationSteps")
        axs[1, 2].set_ylabel("Success Rate")
        axs[1, 2].legend()
        axs[1, 2].grid(alpha=0.4, linestyle="dashed")

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

def plotOverallEvaluationResults(originalEnvResults: Tuple[str, str, Dict[str, List[float]]], customEnvResults: Tuple[str, str, Dict[str, List[float]]]) -> None:
    """
    # Description
        -> This function aims to plot the evaluation results of the used algorithms under the respective settings and environments.
    -------------------------------------------------------------------------------------------------------------------------------
    := param: originalEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the original Environment.
    := param: customEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the custom Environment.
    := return: None, since we are only plotting results.
    """
    
    # Create a Custom Color Map
    customColorMap = ['#29599c', '#f66b6e', '#4cb07a', '#f8946c']

    # Creating the figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(17, 8))

    # Then manually add more space at the top and between rows
    fig.subplots_adjust(top=0.9, hspace=0.4)

    # Add a title for the top row (Original Environment)
    fig.text(0.5, 0.94, "[Original Environment]", ha='center', va='center', fontsize=14, fontweight='bold')

    # Add a title for the bottom row (Custom Environment)
    fig.text(0.5, 0.48, "[Custom Environment]", ha='center', va='center', fontsize=14, fontweight='bold')

    def plot_error_charts(results, row):
        labels = [f"{model}\n{setting}" for model, setting, _ in results]

        # Rewards
        mean_rewards = [np.mean(data['rewards']) for _, _, data in results]
        std_rewards = [np.std(data['rewards']) for _, _, data in results]
        axs[row, 0].bar(labels, mean_rewards, yerr=std_rewards, align='center', ecolor='black', capsize=10, color=customColorMap[:len(results)])
        axs[row, 0].set_title("Average Reward")
        axs[row, 0].set_ylabel("Reward")

        # Episode Lengths
        mean_lengths = [np.mean(data['length']) for _, _, data in results]
        std_lengths = [np.std(data['length']) for _, _, data in results]
        axs[row, 1].bar(labels, mean_lengths, yerr=std_lengths, align='center', ecolor='black', capsize=10, color=customColorMap[:len(results)])
        axs[row, 1].set_title("Average Episode Length")
        axs[row, 1].set_ylabel("Episode Length")

        # Success Rates
        success_rates = [(np.array(data['rewards']) > CONFIG["SUCCESS_THRESHOLD"]).mean() for _, _, data in results]
        success_rates_std = [np.std((np.array(data['rewards']) > CONFIG["SUCCESS_THRESHOLD"]).astype(float)) for _, _, data in results]
        axs[row, 2].bar(labels, success_rates, yerr=success_rates_std, align='center', ecolor='black', capsize=10, color=customColorMap[:len(results)])
        axs[row, 2].set_title("Success Rate")
        axs[row, 2].set_ylabel("Success Rate")

    # Plot for original environment
    plot_error_charts(originalEnvResults, 0)

    # Plot for custom environment
    plot_error_charts(customEnvResults, 1)

    # Show the plot
    plt.show()

def plotOverallEvaluationResultsViolinplots(originalEnvResults: Tuple[str, str, Dict[str, List[float]]], customEnvResults: Tuple[str, str, Dict[str, List[float]]]) -> None:
    """
    # Description
        -> This function aims to plot the evaluation results within violin plots of the used algorithms under the respective settings and environments.
    ---------------------------------------------------------------------------------------------------------------------------------------------------
    := param: originalEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the original Environment.
    := param: customEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the custom Environment.
    := return: None, since we are only plotting results.
    """
    
    # Create a Custom Color Map
    customColorMap = ['#29599c', '#f66b6e', '#4cb07a', '#f8946c']

    # Creating the figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))

    # Then manually add more space at the top and between rows
    fig.subplots_adjust(top=0.9, hspace=0.4)

    # Add a title for the top row (Original Environment)
    fig.text(0.5, 0.94, "[Original Environment]", ha='center', va='center', fontsize=14, fontweight='bold')

    # Add a title for the bottom row (Custom Environment)
    fig.text(0.5, 0.48, "[Custom Environment]", ha='center', va='center', fontsize=14, fontweight='bold')

    def plot_violinplots(results, row):
        labels = [f"{model}\n{setting}" for model, setting, _ in results]

        # Rewards
        rewards_data = [data['rewards'] for _, _, data in results]
        axs[row, 0].violinplot(rewards_data, showmeans=True, showextrema=True, showmedians=True)
        axs[row, 0].set_title("Reward Distribution")
        axs[row, 0].set_ylabel("Reward")
        axs[row, 0].set_xticks(range(1, len(labels) + 1))
        axs[row, 0].set_xticklabels(labels)

        # Episode Lengths
        lengths_data = [data['length'] for _, _, data in results]
        axs[row, 1].violinplot(lengths_data, showmeans=True, showextrema=True, showmedians=True)
        axs[row, 1].set_title("Episode Length Distribution")
        axs[row, 1].set_ylabel("Episode Length")
        axs[row, 1].set_xticks(range(1, len(labels) + 1))
        axs[row, 1].set_xticklabels(labels)

        # Success Rates
        success_rates_data = [(np.array(data['rewards']) > CONFIG["SUCCESS_THRESHOLD"]).astype(float) for _, _, data in results]
        axs[row, 2].violinplot(success_rates_data, showmeans=True, showextrema=True, showmedians=True)
        axs[row, 2].set_title("Success Rate Distribution")
        axs[row, 2].set_ylabel("Success Rate")
        axs[row, 2].set_xticks(range(1, len(labels) + 1))
        axs[row, 2].set_xticklabels(labels)

        # Add color to violin plots
        for i, pc in enumerate(axs[row, 0].collections):
            pc.set_facecolor(customColorMap[i % len(customColorMap)])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        for i, pc in enumerate(axs[row, 1].collections):
            pc.set_facecolor(customColorMap[i % len(customColorMap)])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        for i, pc in enumerate(axs[row, 2].collections):
            pc.set_facecolor(customColorMap[i % len(customColorMap)])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
    # Plot for original environment
    plot_violinplots(originalEnvResults, 0)

    # Plot for custom environment
    plot_violinplots(customEnvResults, 1)

    # Show the plot
    plt.show()
    
def plotCriticalDifferenceDiagram(originalEnvResults: Tuple[str, str, Dict[str, List[float]]], customEnvResults: Tuple[str, str, Dict[str, List[float]]]) -> None:
    """
    # Description
        -> This function aims to plot the critical differences diagram of the used algorithms's performance (based on the rewards) under the respective settings and environments.
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    := param: originalEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the original Environment.
    := param: customEnvResults - Tuple composed by the algorithm and setting alongside the List with all the evallogs obtained for the trained models for the custom Environment.
    := return: None, since we are only plotting results.
    """
    
    # Create a initial dictionary to store the data
    data = {}

    # Update the data dictionary with the received data
    for algorithm, settings, envResults in originalEnvResults:
        data.update({f"[OriginalEnv]-{algorithm}-{settings}":envResults['rewards']})
    # Update the data dictionary with the received data
    for algorithm, settings, envResults in customEnvResults:
        data.update({f"[CustomEnv]-{algorithm}-{settings}":envResults['rewards']})

    # Create a matrix with the collected data
    matrix = pd.DataFrame(data)

    # Calculate ranks
    ranks = matrix.rank(axis=1, ascending=False).mean()

    # Step 3: Perform the Nemenyi post-hoc test
    nemenyi = sp.posthoc_nemenyi_friedman(matrix)

    # Adapt all the 0 values into a small value
    nemenyi = nemenyi.replace(0, 1e-8)

    # Styling parameters
    marker_props = {'marker': 'o', 'linewidth': 1}  # Markers
    label_props = {'backgroundcolor': '#ADD5F7', 'verticalalignment': 'top'}

    # Plot Critical Difference Diagram
    plt.figure(figsize=(10, 6))
    _ = sp.critical_difference_diagram(ranks=ranks, sig_matrix=nemenyi, marker_props=marker_props, label_props=label_props)

    # Title and layout
    plt.title("Critical Difference Diagram of Rewards")
    plt.tight_layout()
    plt.show()
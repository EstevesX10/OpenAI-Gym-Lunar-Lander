<div align="center">

# OpenAI Gym | Lunar Lander

</div>

<p align="center" width="100%">
    <img src="./LunarLander/Assets/Spacecraft.jpg" width="55%" height="55%" alt="Multi-Agent Autonomous Waste Collection System"/>
</p>

<div align="center">
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Python-bedcf5?style=for-the-badge&logo=Python&logoColor=bedcf5">
    </a>
    <a>
        <img src="https://img.shields.io/badge/Made%20with-SPADE-bedcf5?style=for-the-badge&logo=robotframework&logoColor=bedcf5">
    </a>
</div>

<br/>

<div align="center">
    <a href="https://github.com/EstevesX10/OpenAI-Gym-Lunar-Lander/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/EstevesX10/OpenAI-Gym-Lunar-Lander?style=flat&logo=gitbook&logoColor=bedcf5&label=License&color=bedcf5">
    </a>
    <a href="#">
        <img src="https://img.shields.io/github/repo-size/EstevesX10/OpenAI-Gym-Lunar-Lander?style=flat&logo=googlecloudstorage&logoColor=bedcf5&logoSize=auto&label=Repository%20Size&color=bedcf5">
    </a>
    <a href="#">
        <img src="https://img.shields.io/github/stars/EstevesX10/OpenAI-Gym-Lunar-Lander?style=flat&logo=adafruit&logoColor=bedcf5&logoSize=auto&label=Stars&color=bedcf5">
    </a>
    <a href="https://github.com/EstevesX10/OpenAI-Gym-Lunar-Lander/blob/main/DEPENDENCIES.md">
        <img src="https://img.shields.io/badge/Dependencies-DEPENDENCIES.md-white?style=flat&logo=anaconda&logoColor=bedcf5&logoSize=auto&color=bedcf5"> 
    </a>
</div>

## Project Overview

This project explores the impact of customizing an `OpenAI Gym Environment` on **reinforcement learning (RL) performance**. We modified an existing Gym environment - Lunar Lander - in order to train an RL agent using the Stable Baselines library, and later compare results between the **customized and original environments**.

The process involves:

- **Environment Customization**: **Implement changes** such as altered rewards or added challenges to the Environment.
- **Agent Training**: Train an RL agent with **algorithms like PPO** and further **tune their hyperparameters** to ensure optimal performance.
- **Evaluation**: **Compare agent performance** in both environments to analyze the effect of the customizations.

This project aims to analyse **how does the environment design influence the outcomes of a Reinforcement Learning Algorithm**.

## Project Results

Adopting a training strategy with **periodic saves and evaluations**, we aimed to identify the **best performing model** for each configuration of the **environment**, **algorithm** and respective **hyperparameters** over 10-30 million time steps, resulting in the **following outcomes**:

<table width="100%">
    <thead>
        <th></th>
        <th>
            <div align="center">
                Original Environment
            </div>
        </th>
        <th>
            <div align="center">
                Custom Environment
            </div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td width="10%">
                <p align="center" width="100%">
                    [PPO] Settings1
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/ExperimentalResults/OriginalEnvironment/PPO/Settings-1/recordings/rl-video-episode-0.gif" width="100%" height="100%" />
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/ExperimentalResults/CustomEnvironment/PPO/Settings-1/recordings/rl-video-episode-0.gif" width="100%" height="100%" />
                </p>
            </td>
        </tr>
        <tr>
            <td width="10%">
                <p align="center" width="100%">
                    [PPO] Settings2
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/Assets/Dash.png" width="30%" height="30%" />
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/ExperimentalResults/CustomEnvironment/PPO/Settings-2/recordings/rl-video-episode-0.gif" width="100%" height="100%" />
                </p>
            </td>
        </tr>
        <tr>
            <td width="10%">
                <p align="center" width="100%">
                    [DQN] Settings1
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/ExperimentalResults/OriginalEnvironment/DQN/Settings-1/recordings/rl-video-episode-0.gif" width="100%" height="100%" />
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/ExperimentalResults/CustomEnvironment/DQN/Settings-1/recordings/rl-video-episode-0.gif" width="100%" height="100%" />
                </p>
            </td>
        </tr>
        <tr>
            <td width="10%">
                <p align="center" width="100%">
                    [DQN] Settings2
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/Assets/Dash.png" width="30%" height="30%" />
                </p>
            </td>
            <td width="45%">
                <p align="center" width="100%">
                    <img src="./LunarLander/ExperimentalResults/CustomEnvironment/DQN/Settings-2/recordings/rl-video-episode-0.gif" width="100%" height="100%" />
                </p>
            </td>
        </tr>
    </tbody>
</table>

## Results Evaluation

Finally, we proceed to **compare all algorithms and configurations** to identify the **best-performing combination** for each environment.

<!-- Bar Plot -->

<table width="100%">
    <thead>
        <th>
            <div align="center">
                Bar Plot
            </div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td width="50%">
                <p align="center" width="100%">
                    <img src="./LunarLander/Assets/FinalBarPlot.png" width="100%" height="100%" />
                </p>
            </td>
        </tr>
    </tbody>
</table>

<!-- Violin Plot -->

<table width="100%">
    <thead>
        <th>
            <div align="center">
                Violin Plot
            </div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td width="100%">
                <p align="center" width="100%">
                    <img src="./LunarLander/Assets/FinalViolinPlot.png" width="100%" height="100%" />
                </p>
            </td>
        </tr>
    </tbody>
</table>

<!-- Critical Differences Diagram Plot -->

<table width="100%">
    <thead>
        <th>
            <div align="center">
                Critical Differences Diagram
            </div>
        </th>
    </thead>
    <tbody>
        <tr>
            <td width="100%">
                <p align="center" width="100%">
                    <img src="./LunarLander/Assets/CriticalDifferencesDiagram.png" width="100%" height="100%" />
                </p>
            </td>
        </tr>
    </tbody>
</table>

The **PPO algorithm consistently outperforms the DQN algorithm** in both the **Original and Custom Environments**. It's performance remains relatively **stable across different settings**, showing **minimal differences** between the Original Environment and the Custom Environment.

Regardless of the **applied configuration**, PPO demonstrates **consistent success** in **achieving high rewards**, **shorter episode lengths**, and **higher success rates**.

In contrast, the **DQN algorithm's performance** **deteriorates** significantly in the **Custom Environment**. While the Custom Environment does **lead to shorter episode lengths for DQN**, the results are **considerably worse** compared to the Original Environment.

This decline is primarily because, in most episodes, the **DQN agent struggles to manage its fuel efficiently**. Instead of progressing toward the goal, the agent often **expends excessive fuel** attempting to **stabilize its position**. Consequently, this behavior results in **frequent failure to reach the goal** and **negative rewards**.

## Authorship

- **Authors** &#8594; [Gonçalo Esteves](https://github.com/EstevesX10), [Nuno Gomes](https://github.com/NightF0x26) and [Pedro Afonseca](https://github.com/PsuperX)
- **Course** &#8594; Introduction to Intelligent Autonomous Systems [[CC3042](https://sigarra.up.pt/fcup/en/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=546531)]
- **University** &#8594; Faculty of Sciences, University of Porto

<div align="right">
<sub>

<!-- <sup></sup> -->

`README.md by Gonçalo Esteves`
</sub>

</div>

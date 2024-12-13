# Gomoku as a Markov Decision Process
**Team Members:** <br>
Ioana-Andreea Cristescu, Alyssa Mia Taliotis, Alissia Di Maria

## Project:
This project models the strategic game of Gomoku as a Markov Decision Process (MDP) to develop a Reinforcement Learning (RL) agent capable of learning competitive strategies. The focus is on creating a robust and dynamic training environment where agents learn by competing against a deterministic, rule-based player rather than relying on self-play, ensuring a consistent and structured learning process. By leveraging state-of-the-art RL algorithms, such as Proximal Policy Optimization (PPO) and Deep Q-Learning (DQN), the project aims to address the challenges of Gomoku’s large state and action spaces. It systematically explores the impact of various reward functions—ranging from default win/loss-based rewards to more granular configurations that encourage offensive and defensive gameplay—on the learning efficiency and strategic depth of the agents. Additionally, advanced RL techniques such as action masking, hyperparameter tuning, and reward shaping are implemented to optimize agent performance. The project also incorporates interpretability tools, including heatmaps, to visualize the agents’ decision-making processes, offering insights into learned strategies. Together, these efforts aim to advance RL applications in strategy games while providing a practical framework for training and analyzing competitive agents in Gomoku.

## Instructions to run our Gomoku MDP Game
This guide will walk you through setting up the environment, training the agents, running the game, and exploring its functionality.

### Environment
This section explains how to set up and work with the required Python environment for running the Gomoku MDP project.

#### First Time Setup
For the first time, you need to install the required dependencies and set up the Python environment. Follow these steps:
1. Install `pipenv` if not already installed: `pip install pipenv`
2. Navigate to the project directory: `cd /path/to/stat184_gomoku_mdp`
3. Install the dependencies using `pipenv`: `pipenv install`
4. Activate the virtual environment: `pipenv shell`

#### After the First Time
Once the environment is set up, you only need to activate the virtual environment to start working on the project:
1. Activate the virtual environment: `pipenv shell`
2. If dependencies change (e.g., after a repository update), update them: `pipenv install`

### Gomoku Game
This section provides instructions for training the RL agent, playing Gomoku using the trained agent, and testing the environment to ensure everything is working correctly. Make sure that the environmnet is already activated before running any of the commands below.

#### Train the Agents
Follow these steps to train the RL agents against a rule-based player:

1. Prepare the reward configuration:
   The reward configurations for training are defined in YAML files. Before running the training script, update the reward values in the YAML files as needed (e.g., adjusting win values to explore different training dynamics).

2. Run the training scripts:
    Both train_rule_based_dqn.py and train_rule_based_ppo.py scripts allow customization through command-line arguments. You can adjust these parameters to fine-tune the training process:
    - --num_episodes: Number of training episodes (default: 1000).
    - --board_size: Size of the Gomoku board (default: 8).
    - --config_name: Name of the reward configuration file (without .yml extension). Update this to match your chosen reward setup (e.g., rewards_offensive).
    - --device: Specify the computation device (cpu or cuda). If not specified, the script auto-detects the available device.
    - --model_save_path: Path to save the trained model (default: dqn_gomoku or equivalent for PPO).
    - --win_reward: Value for the win reward. This should correspond to the win reward set in the YAML configuration.

    Example commands:
    - Train the DQN agent:
        ``` python train_rule_based_dqn.py --num_episodes 500 --board_size 8 --config_name rewards_1 --device cuda --win_reward 10 ```
    - Train the PPO agent:
        ``` python train_rule_based_ppo.py --num_episodes 500 --board_size 8 --config_name rewards_1 --device cuda --win_reward 10 ```

3. Additional Training Modes: The same instructions apply for training agents in alternative setups:
    - Random-Based Player: use train_random_dqn.py and train_random_ppo.py with similar arguments to train agents against a random-based player.
    - Self-Play: use train_self_play_dqn.py and train_self_play_ppo.py with similar arguments to train agents using self-play.

#### Visualize Training Metrics
You can visualize the training metrics for agents trained against a rule-based player using the provided scripts. These visualizations help evaluate the agent’s learning performance, including win rates, losses, and rewards.

1. Run the Visualization Scripts
    Both vis_rule_based_dqn.py and vis_rule_based_ppo.py scripts allow customization through command-line arguments. You can adjust these parameters to tailor the visualization process:
    - --config_name: Name of the reward configuration file used during training (default: rewards_default). Update this to match your chosen reward setup (e.g., rewards_offensive).
    -  --win_reward: Value for the win reward used during training (default: 1). This should correspond to the win reward specified in the YAML configuration.

    Example Commands:
    - Visualize metrics for the DQN agent:
``` python vis_rule_based_dqn.py --config_name rewards_1 --win_reward 10 ```
    - Visualize metrics for the PPO agent:
``` python vis_rule_based_ppo.py --config_name rewards_2 --win_reward 5 ```

2. The same instructions apply for visualizing metrics from alternative setups (random-based training and sel-play).

#### Validate and Select Best Agents
To validate and compare the performance of trained agents, the project includes scripts to run evaluation matches between different agents. These matches allow for systematic comparison of agents trained with different algorithms, reward configurations, and training setups. The results help identify the best-performing agents and analyze their strategic effectiveness.

1. Evaluation Matches
    The following evaluation matches are supported:
    - Agent vs Rule-Based Player (`eval_agent_vs_rule_based.py`): Evaluate how well a trained agent performs against the rule-based player.
    - DQN vs DQN (`eval_dqn_vs_dqn_rule_based.py`): Compare two DQN agents trained with different configurations or setups.
    - PPO vs PPO (`eval_ppo_vs_ppo_rule_based.py`): Compare two PPO agents trained with different configurations or setups.
    - DQN vs PPO (`eval_dqn_vs_ppo_rule_based.py`): Evaluate the performance of a DQN agent against a PPO agent.
2. Arguments 
    Each validation script accepts command-line arguments similar to the ones used in training and visualization to select the preffered configuration.

#### Playing Gomoku with the Trained RL Agent and Generate Heatmaps
The project includes functionality to generate heatmaps that visualize the decision-making processes of RL agents. These heatmaps provide valuable insights into the strategies learned by the agents, highlighting their focus on potential moves during gameplay. Heatmaps can be generated for both DQN and PPO agents while playing against a human.

Customize the heatmap generation by using the following arguments:
- --mode: Set this to human to enable Agent vs. Human gameplay.
- --agent: Specify the agent type (ppo or dqn).
- --config_name: Specify the reward configuration file used during training (default: rewards_default).
- --win_reward: Set the win reward value used during training (default: 1).
- --generate_heatmaps: Add this flag to enable heatmap generation.

Example Command:

```python evaluate.py --mode human --agent ppo --config_name rewards_default --win_reward 10 --generate_heatmaps ```



This feature allows users to play Gomoku against a trained RL agent while generating heatmaps at every step. The heatmaps provide a visual representation of the agent's decision-making process, offering valuable insights into its learned strategies and helping players learn from its moves. Both DQN and PPO agents are supported in this interactive gameplay mode.

During gameplay, the agent dynamically evaluates the board state and highlights its focus on potential moves:
- DQN Agents: Heatmaps visualize Q-values for each valid move, with darker blue shades representing moves the agent considers most valuable.
- PPO Agents: Heatmaps display action probabilities for each valid move, with darker red shades indicating moves with higher probabilities.

1. How to Play and Generate Heatmaps:
    - Customize the gameplay and heatmap generation by using the following command-line arguments:
        - --mode: Set this to human to enable Agent vs. Human gameplay.
        - --agent: Specify the agent type (ppo or dqn).
        - --config_name: Specify the reward configuration file used during training (default: rewards_default).
        - --win_reward: Set the win reward value used during training (default: 1).
        - --generate_heatmaps: Add this flag to enable heatmap generation.

    - Example Command:
        - Play against a PPO agent with heatmap generation enabled:
        ```python evaluate.py --mode human --agent ppo --config_name rewards_default --win_reward 10 --generate_heatmaps ```

2. Features During Gameplay:
    - Interactive Moves: The human player competes as O, while the agent plays as X. Players input their moves in a human-friendly format (e.g., a1, b2).
    - Dynamic Heatmaps: At every step of the agent's turn, the heatmap is displayed, highlighting its evaluation of the current board. The visualizations include:
        - The current board state with X and O indicating moves.
        - A color bar showing the action probability or Q-value scale.
        - Labels for rows and columns to help interpret positions.




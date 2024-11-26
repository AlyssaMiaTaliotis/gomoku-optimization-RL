# Gomoku as a Markov Decision Process
**Team Members:** <br>
Ioana-Andreea Cristescu, Alyssa Mia Taliotis, Alissia Di Maria

## Project Organization

```
├── Readme.md
├── Pipfile
├── Pipfile.lock
├── gomoku_env.py
├── test_gomoku_env.py
└── rewards
    ├── rewards_1.yml
    ├── rewards_2.yml
    └── rewards_default.yml
```

## Project:
This project models the strategic game of Gomoku as a Markov Decision Process (MDP) to develop an Reinforcement Learning (RL) agent capable of learning competitive strategies. By leveraging self-play and various reinforcement learning algorithms, such as Proximal Policy Optimization (PPO) and Deep Q-Learning (DQN), the agent iteratively improves its decision-making capabilities. The project focuses on designing a robust game environment, experimenting with multiple reward functions to guide learning, and implementing advanced RL techniques to optimize performance. 

## Milestone Progress
`TBD`

## Files overview 

1. **`gomoku_env.py`**

    This class implements a Gomoku game environment following OpenAI Gym conventions, enabling the simulation of moves, validation of actions, and calculation of rewards for game strategies. It manages a configurable board size and supports alternating turns between two players. The class validates moves, checks for wins or draws, and updates the board state accordingly. Rewards are computed based on a YAML-configured system, supporting both forming rows (e.g., 2-in-a-row, 3-in-a-row, 4-in-a-row) and blocking opponent strategies (e.g., blocking 3-in-a-row, blocking 4-in-a-row). The environment includes utility functions to analyze sequences in all directions, detect double threats, and penalize isolated moves. Additionally, the class provides a rendering function to visualize the board state as a grid.

2. **`test_gomoku_env.py`**

    This script contains unit and integration test cases for validating the functionality of the GomokuEnvironment class. It tests key scenarios, including valid and invalid moves, forming rows of varying lengths, blocking opponent strategies, and handling game-ending conditions like wins and draws. 

## Instructions to run our Gomoku MDP Game
This guide will walk you through setting up the environment, running the game, and exploring its functionality.

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
#### Train the RL Agent
`TBD`
#### Playing Gomoku with the Trained RL Agent
`TBD`
#### Testing
To test the Gomoku environment run `python test_gomoku_env.py`. If no exceptions are raised, all test cases have passed successfully.

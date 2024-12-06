# import numpy as np
# import matplotlib.pyplot as plt
# import os

# def visualize_training(log_every=1, save_folder="plots"):
#     """
#     Visualizes the training loss, rewards, and win rates over episodes.

#     Args:
#         log_every (int): The logging interval used during training.
#     """
#     # Load the metrics
#     try:
#         win_rates = np.load("win_rates.npy")
#         losses = np.load("losses.npy")
#     except FileNotFoundError:
#         print("Metrics files not found. Ensure that 'win_rates.npy' and 'losses.npy' exist.")
#         return

#     # Compute the number of episodes
#     num_points = len(win_rates)
#     episodes = np.arange(log_every, (num_points + 1) * log_every, log_every)

#     # Separate win rates for each agent
#     agent1_win_rates = [wr[0] for wr in win_rates]
#     agent2_win_rates = [wr[1] for wr in win_rates]

#     # Plotting
#     plt.figure(figsize=(12, 8))

#     # Plot Win Rates
#     plt.subplot(3, 1, 1)
#     plt.plot(episodes, agent1_win_rates, label='Agent 1 Win Rate')
#     plt.plot(episodes, agent2_win_rates, label='Agent 2 Win Rate')
#     plt.xlabel('Episode')
#     plt.ylabel('Win Rate')
#     plt.title('Win Rates Over Episodes')
#     plt.legend()

#     # Plot Training Loss
#     plt.subplot(3, 1, 2)
#     plt.plot(np.arange(len(losses)), losses)
#     plt.xlabel('Training Steps')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Over Time')

#     # Plot Total Rewards per Episode
#     try:
#         episode_rewards = np.load("episode_rewards.npy")
#         plt.subplot(3, 1, 3)
#         plt.plot(np.arange(1, len(episode_rewards) + 1), episode_rewards)
#         plt.xlabel('Episode')
#         plt.ylabel('Total Reward')
#         plt.title('Total Rewards per Episode')
#     except FileNotFoundError:
#         print("Episode rewards not found. Skipping reward plot.")

#     plt.tight_layout()
#     # Save the plot as an image
#     if not os.path.exists(save_folder):  # Create the folder if it doesn't exist
#         os.makedirs(save_folder)  # Added to ensure the folder exists
#     save_path = os.path.join(save_folder, "training_metrics.png")
#     plt.savefig(save_path)  # Save the plot as a .png image
#     print(f"Plot saved to {save_path}")  # Added to confirm where the file was saved

#     plt.show()

# if __name__ == "__main__":
#     # Replace 'log_every' with the actual value used during training
#     visualize_training(log_every=1)

import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_training(log_every=1, save_folder="plots"):
    """
    Visualizes the training metrics: win rates, losses, and rewards over episodes for both agents.

    Args:
        log_every (int): The logging interval used during training.
        save_folder (str): The folder where the plot image will be saved.
    """
    # Load the metrics
    try:
        win_rates = np.load("win_rates.npy")
    except FileNotFoundError:
        print("Win rates file not found. Ensure 'win_rates.npy' exists.")
        return

    try:
        agent1_rewards = np.load("agent1_rewards.npy")
        agent2_rewards = np.load("agent2_rewards.npy")
    except FileNotFoundError:
        print("Reward files not found. Ensure 'agent1_rewards.npy' and 'agent2_rewards.npy' exist.")
        return

    try:
        agent1_losses = np.load("agent1_losses.npy")
        agent2_losses = np.load("agent2_losses.npy")
    except FileNotFoundError:
        print("Loss files not found. Ensure 'agent1_losses.npy' and 'agent2_losses.npy' exist.")
        return

    # Compute the number of episodes
    num_points = len(win_rates)
    episodes = np.arange(log_every, (num_points + 1) * log_every, log_every)

    # Ensure that the lengths of episodes and win_rates match
    if len(episodes) != len(win_rates):
        print("Mismatch in length between episodes and win_rates.")
        min_length = min(len(episodes), len(win_rates))
        episodes = episodes[:min_length]
        win_rates = win_rates[:min_length]

    # Separate win rates for each agent
    agent1_win_rates = [wr[0] for wr in win_rates]
    agent2_win_rates = [wr[1] for wr in win_rates]

    # Plotting
    plt.figure(figsize=(12, 12))

    # Plot Win Rates
    plt.subplot(3, 1, 1)
    plt.plot(episodes/10, agent1_win_rates, label='Agent 1 Win Rate', color='blue')
    plt.plot(episodes/10, agent2_win_rates, label='Agent 2 Win Rate', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rates Over Episodes')
    plt.legend()

    # Plot Training Losses
    # Assuming losses are averaged per episode
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(1, len(agent1_losses) + 1), agent1_losses, label='Agent 1 Loss', color='blue')
    plt.plot(np.arange(1, len(agent2_losses) + 1), agent2_losses, label='Agent 2 Loss', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Episodes')
    plt.legend()

    # Plot Total Rewards per Episode
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(1, len(agent1_rewards) + 1), agent1_rewards, label='Agent 1 Reward', color='blue')
    plt.plot(np.arange(1, len(agent2_rewards) + 1), agent2_rewards, label='Agent 2 Reward', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Over Episodes')
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "training_metrics.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    visualize_training(log_every=10)  # Use the same log_every as in training

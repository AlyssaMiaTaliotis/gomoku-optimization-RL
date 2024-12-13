import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_training_against_rule_based(rewards_type="rewards_default", log_every=1, suffix="1", save_folder="plots_rule_based"):
    """
    Visualizes the training metrics: win rates, losses, and rewards over episodes 
    for the DQN agent against the rule-based player.

    Args:
        rewards_type (str): The reward configuration type used during training.
        log_every (int): The logging interval used during training.
        save_folder (str): The folder where the plot image will be saved.
    """
    folder = f"rule_based_dqn/{rewards_type}"
    
    # Load the metrics
    try:
        win_rates = np.load(f"{folder}/win_rates_{suffix}.npy")
    except FileNotFoundError:
        print(f"Win rates file not found. Ensure '{folder}/win_rates_{suffix}.npy' exists.")
        return

    try:
        agent1_rewards = np.load(f"{folder}/agent1_rewards_{suffix}.npy")
    except FileNotFoundError:
        print(f"Reward file not found. Ensure '{folder}/agent1_rewards_{suffix}.npy' exists.")
        return

    try:
        agent1_losses = np.load(f"{folder}/agent1_losses_{suffix}.npy")
    except FileNotFoundError:
        print(f"Loss file not found. Ensure '{folder}/agent1_losses_{suffix}.npy' exists.")
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

    # Separate win rates for each player
    agent1_win_rates = [wr[0] for wr in win_rates]
    rule_based_win_rates = [wr[1] for wr in win_rates]

    # Adjust font sizes
    title_fontsize = 20
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    # Plotting
    plt.figure(figsize=(14, 8))

    # Plot Win Rates
    plt.subplot(3, 1, 1)
    plt.plot(episodes, agent1_win_rates, label='Agent 1 Win Rate (DQN)', color='blue')
    plt.plot(episodes, rule_based_win_rates, label='Rule-Based Player Win Rate', color='orange')
    plt.xlabel('Episode', fontsize=label_fontsize)
    plt.ylabel('Win Rate', fontsize=label_fontsize)
    plt.title('Win Rates Over Episodes', fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)


    # Plot Training Losses
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(1, len(agent1_losses) + 1), agent1_losses, label='Agent 1 Loss (DQN)', color='blue')
    plt.xlabel('Episode', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.title('Training Losses Over Episodes', fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Plot Total Rewards per Episode
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(1, len(agent1_rewards) + 1), agent1_rewards, label='Agent 1 Reward (DQN)', color='blue')
    plt.xlabel('Episode', fontsize=label_fontsize)
    plt.ylabel('Total Reward', fontsize=label_fontsize)
    plt.title('Total Rewards Over Episodes', fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    # plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    if not os.path.exists(f"{save_folder}/{rewards_type}"):
        os.makedirs(f"{save_folder}/{rewards_type}")
    save_path = os.path.join(f"{save_folder}/{rewards_type}", f"dqn_training_metrics_{suffix}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DQN Agents against Rule-based Agent in Gomoku")
    parser.add_argument("--log_every", type=int, default=1, help="Plotting points")
    parser.add_argument("--config_name", type=str, default="rewards_default", help="Name of the reward configuration file (without .yml extension)")
    parser.add_argument("--win_reward", type=str, default="1", help="Value for the win reward")

    args = parser.parse_args()
    
    visualize_training_against_rule_based(rewards_type=args.config_name, log_every=args.log_every, suffix=args.win_reward)

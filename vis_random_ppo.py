import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_training_against_random(rewards_type="rewards_default", log_every=1, save_folder="plots_random_ppo"):
    """
    Visualizes the training metrics: win rates, losses, and rewards over episodes 
    for the PPO agent against the random player.

    Args:
        rewards_type (str): The reward configuration type used during training.
        log_every (int): The logging interval used during training.
        save_folder (str): The folder where the plot image will be saved.
    """
    folder = f"random_ppo/{rewards_type}"
    
    # Load the metrics
    try:
        win_rates = np.load(f"{folder}/win_rates.npy")
    except FileNotFoundError:
        print(f"Win rates file not found. Ensure '{folder}/win_rates.npy' exists.")
        return

    try:
        agent1_rewards = np.load(f"{folder}/agent1_rewards.npy")
    except FileNotFoundError:
        print(f"Reward file not found. Ensure '{folder}/agent1_rewards.npy' exists.")
        return

    try:
        policy_losses = np.load(f"{folder}/policy_losses.npy")
        value_losses = np.load(f"{folder}/value_losses.npy")
    except FileNotFoundError:
        print(f"Loss files not found. Ensure '{folder}/policy_losses.npy' and '{folder}/value_losses.npy' exist.")
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
    random_agent_win_rates = [wr[1] for wr in win_rates]

    # Plotting
    plt.figure(figsize=(12, 10))

    # Plot Win Rates
    plt.subplot(3, 1, 1)
    plt.plot(episodes, agent1_win_rates, label='Agent 1 Win Rate (PPO)', color='blue')
    plt.plot(episodes, random_agent_win_rates, label='Random Agent Win Rate', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rates Over Episodes')
    plt.legend()

    # Plot Training Losses (Policy and Value Losses)
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(1, len(policy_losses) + 1), policy_losses, label='Policy Loss', color='red')
    plt.plot(np.arange(1, len(value_losses) + 1), value_losses, label='Value Loss', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Episodes')
    plt.legend()

    # Plot Total Rewards per Episode
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(1, len(agent1_rewards) + 1), agent1_rewards, label='Agent 1 Reward (PPO)', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Over Episodes')
    plt.legend()

    plt.tight_layout()

    # Save the plot as an image
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f"ppo_training_metrics_{rewards_type}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PPO Agent against Random Agent in Gomoku")
    parser.add_argument("--log_every", type=int, default=1, help="Plotting points")
    parser.add_argument("--config_name", type=str, default="rewards_default", help="Name of the reward configuration file (without .yml extension)")

    args = parser.parse_args()
    
    visualize_training_against_random(rewards_type=args.config_name, log_every=args.log_every)

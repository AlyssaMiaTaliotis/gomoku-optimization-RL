import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_training(save_folder="plots"):
    """
    Visualizes the training metrics: win rates, losses, and rewards over episodes for both agents.

    Args:
        save_folder (str): The folder where the plot image will be saved.
    """
    try:
        win_rates = np.load("win_rates_ppo.npy")
        episode_rewards = np.load("episode_rewards_ppo.npy")
        policy_losses = np.load("policy_losses_ppo.npy")
        value_losses = np.load("value_losses_ppo.npy")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    num_points = len(win_rates)
    episodes = np.arange(1, num_points + 1)

    agent1_win_rates = [wr[0] for wr in win_rates]
    agent2_win_rates = [wr[1] for wr in win_rates]
    agent1_rewards = [er[0] for er in episode_rewards]
    agent2_rewards = [er[1] for er in episode_rewards]

    plt.figure(figsize=(12, 10))

    # Win Rates
    plt.subplot(3, 1, 1)
    plt.plot(episodes, agent1_win_rates, label="Agent 1 Win Rate", color="blue")
    plt.plot(episodes, agent2_win_rates, label="Agent 2 Win Rate", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rates Over Episodes")
    plt.legend()

    # Total Rewards
    plt.subplot(3, 1, 2)
    plt.plot(episodes, agent1_rewards, label="Agent 1 Reward", color="blue")
    plt.plot(episodes, agent2_rewards, label="Agent 2 Reward", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards Over Episodes")
    plt.legend()

    # Losses
    plt.subplot(3, 1, 3)
    plt.plot(policy_losses, label="Policy Loss", color="red")
    plt.plot(value_losses, label="Value Loss", color="green")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Losses Over Time")
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
    visualize_training()

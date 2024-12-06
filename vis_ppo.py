import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_training(log_every=1, save_folder="plots"):
    """
    Visualizes the training metrics for PPO: win rates, losses, and rewards.

    Args:
        log_every (int): The logging interval used during training.
    """

    # load the metrics
    try:
        win_rates = np.load("win_rates_ppo.npy")
    except FileNotFoundError:
        print("Win rates file not found. Ensure 'win_rates_ppo.npy' exists.")
        return

    try:
        policy_losses = np.load("policy_losses.npy")
        value_losses = np.load("value_losses.npy")
    except FileNotFoundError:
        print("Losses files not found. Ensure 'policy_losses.npy' and 'value_losses.npy' exist.")
        return

    # load rewards
    try:
        episode_rewards = np.load("episode_rewards_ppo.npy")
    except FileNotFoundError:
        print("Episode rewards file not found. Skipping reward plot.")
        episode_rewards = None

    # compute the number of episodes
    num_points = len(win_rates)
    episodes = np.arange(log_every, (num_points + 1) * log_every, log_every)

    plt.figure(figsize=(12, 10))

    # plot Win Rates
    plt.subplot(3, 1, 1)
    plt.plot(episodes, [wr[0] for wr in win_rates], label='Agent 1 Win Rate', color='blue')
    plt.plot(episodes, [wr[1] for wr in win_rates], label='Agent 2 Win Rate', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rate Over Episodes')
    plt.legend()

    # plot Losses
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(policy_losses)), policy_losses, label='Policy Loss', color='red')
    plt.plot(np.arange(len(value_losses)), value_losses, label='Value Loss', color='green')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Losses Over Time')
    plt.legend()

    # plot total rewards per episode
    if episode_rewards is not None:
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(1, len(episode_rewards) + 1), [er[0] for er in episode_rewards], label='Agent 1 Total Reward', color='blue')
        plt.plot(np.arange(1, len(episode_rewards) + 1), [er[1] for er in episode_rewards], label='Agent 2 Total Reward', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Rewards per Episode')
        plt.legend()

    plt.tight_layout()
    # Save the plot as an image
    if not os.path.exists(save_folder):  # Create the folder if it doesn't exist
        os.makedirs(save_folder)  # Added to ensure the folder exists
    save_path = os.path.join(save_folder, "training_metrics.png")
    plt.savefig(save_path)  # Save the plot as a .png image
    print(f"Plot saved to {save_path}")  # Added to confirm where the file was saved

    plt.show()

if __name__ == "__main__":
    visualize_training(log_every=10)


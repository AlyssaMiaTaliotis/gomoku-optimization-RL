import numpy as np
import matplotlib.pyplot as plt

def visualize_training(log_every=1):
    """
    Visualizes the training loss, rewards, and win rates over episodes.

    Args:
        log_every (int): The logging interval used during training.
    """
    # Load the metrics
    try:
        win_rates = np.load("win_rates.npy")
        losses = np.load("losses.npy")
    except FileNotFoundError:
        print("Metrics files not found. Ensure that 'win_rates.npy' and 'losses.npy' exist.")
        return

    # Compute the number of episodes
    num_points = len(win_rates)
    episodes = np.arange(log_every, (num_points + 1) * log_every, log_every)

    # Separate win rates for each agent
    agent1_win_rates = [wr[0] for wr in win_rates]
    agent2_win_rates = [wr[1] for wr in win_rates]

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot Win Rates
    plt.subplot(3, 1, 1)
    plt.plot(episodes, agent1_win_rates, label='Agent 1 Win Rate')
    plt.plot(episodes, agent2_win_rates, label='Agent 2 Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rates Over Episodes')
    plt.legend()

    # Plot Training Loss
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')

    # Plot Total Rewards per Episode
    try:
        episode_rewards = np.load("episode_rewards.npy")
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(1, len(episode_rewards) + 1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Rewards per Episode')
    except FileNotFoundError:
        print("Episode rewards not found. Skipping reward plot.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace 'log_every' with the actual value used during training
    visualize_training(log_every=1)



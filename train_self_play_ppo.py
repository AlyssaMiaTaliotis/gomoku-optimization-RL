import argparse
import numpy as np
import torch
from gomoku_env import GomokuEnvironment
from ppo_agent import PPOAgent

print("Script has started executing.")

def train_ppo_self_play(
    num_episodes: int = 100, 
    board_size: int = 15, 
    gamma: float = 0.99,
    epsilon: float = 0.2,
    lr: float = 1e-3,
    rollout_steps: int = 64,
    epochs: int = 2,
    batch_size: int = 64,
    device: str = None,
    config_path: str = "rewards/rewards_2.yml",
):
    """
    Trains two PPO agents through self-play in the Gomoku environment.

    Args:
        num_episodes (int): Number of training episodes.
        board_size (int): Size of the Gomoku board.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Clipping parameter for PPO.
        lr (float): Learning rate for optimizers.
        rollout_steps (int): Number of steps to collect before updating.
        epochs (int): Number of epochs to optimize policy and value networks per update.
        batch_size (int): Mini-batch size for updates.
        device (str): Device to run computations on ('cpu' or 'cuda').
        config_path (str): Path to the reward configuration YAML file.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize the Gomoku environment
    env = GomokuEnvironment(board_size=board_size, config_path=config_path)

    # Initialize PPO agents for Player 1 and Player 2
    agent1 = PPOAgent(board_size=board_size, gamma=gamma, epsilon=epsilon, lr=lr, device=device)
    agent2 = PPOAgent(board_size=board_size, gamma=gamma, epsilon=epsilon, lr=lr, device=device)

    # Metrics for tracking
    agent1_wins, agent2_wins, draws = 0, 0, 0
    win_rates, episode_rewards = [], []
    policy_losses, value_losses = [], []

    for episode in range(1, num_episodes + 1):
        # Reset the environment for a new game
        state = env.reset()
        done = False
        agent1_reward, agent2_reward = 0, 0
        step_count = 0

        # Track episode data
        states, actions, rewards, dones, action_probs = [], [], [], [], []

        while not done:
            # Determine which agent is playing
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2

            # Select action based on the policy
            action, action_prob = agent.select_action(state)
            row, col = divmod(action, board_size)
            action_coordinates = (row, col)

            # Execute action
            next_state, reward, done, info = env.step(action_coordinates)

            # Track rewards separately for each agent
            if current_player == 1:
                agent1_reward += reward
            else:
                agent2_reward += reward

            # Store the transition data for training
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            action_probs.append(action_prob)

            # Update state and step count
            state = next_state
            step_count += 1

        # Update win rates
        if "Player 1 wins" in info.get("info", ""):
            agent1_wins += 1
            agent2_reward -= 1  # Penalize losing agent
        elif "Player 2 wins" in info.get("info", ""):
            agent2_wins += 1
            agent1_reward -= 1  # Penalize losing agent
        else:
            draws += 1

        total_games = agent1_wins + agent2_wins + draws
        agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0
        agent2_win_rate = agent2_wins / total_games if total_games > 0 else 0
        win_rates.append((agent1_win_rate, agent2_win_rate))
        episode_rewards.append((agent1_reward, agent2_reward))

        # Compute advantages and returns
        values = [agent1.value_net(torch.FloatTensor(s).unsqueeze(0).unsqueeze(0).to(device)).item() for s in states]
        advantages, returns = agent1.compute_advantages(rewards, values, dones)

        # Perform multiple epochs of training for each collected batch of episode data
        for epoch in range(epochs):
            for start in range(0, len(states), batch_size):
                end = start + batch_size

                # Extract data for the current batch
                batch_states = torch.FloatTensor(np.array(states[start:end])).unsqueeze(1).to(device)
                batch_actions = torch.LongTensor(actions[start:end]).to(device)
                batch_action_probs = torch.FloatTensor(action_probs[start:end]).to(device)
                batch_returns = returns[start:end]
                batch_advantages = advantages[start:end]

                # Perform a single PPO update
                policy_loss, value_loss = agent1.update(batch_states, batch_actions, batch_action_probs, batch_returns, batch_advantages)

                # Track losses
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

        # Log progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}: Agent1 Reward: {agent1_reward}, Agent2 Reward: {agent2_reward}, "
                  f"Win Rates -> Agent1: {agent1_win_rate:.2f}, Agent2: {agent2_win_rate:.2f}")

    # Save metrics
    np.save("win_rates_ppo.npy", win_rates)
    np.save("episode_rewards_ppo.npy", episode_rewards)
    np.save("policy_losses_ppo.npy", policy_losses)
    np.save("value_losses_ppo.npy", value_losses)
    print("Training completed and metrics saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agents via Self-Play in Gomoku")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--board_size", type=int, default=15, help="Size of the Gomoku board")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizers")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per PPO update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computations ('cpu' or 'cuda')")
    parser.add_argument("--config_name", type=str, default="rewards_2", help="Name of the reward configuration file (without .yml extension)")

    args = parser.parse_args()
    config_path = f"rewards/{args.config_name}.yml"

    train_ppo_self_play(
        num_episodes=args.num_episodes,
        board_size=args.board_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        config_path=config_path,
    )



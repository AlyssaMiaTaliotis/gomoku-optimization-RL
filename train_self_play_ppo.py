import argparse
import numpy as np
import torch
import os
from gomoku_env import GomokuEnvironment
from ppo_agent import PPOAgent

print("Script has started executing.")

def train_ppo_self_play(
    num_episodes: int = 1000,
    board_size: int = 8,
    gamma: float = 0.99,
    epsilon: float = 0.05,
    lr: float = 1e-5,
    epochs: int = 4,
    batch_size: int = 128,
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
    agent1_losses, agent2_losses = [], []  # Track losses for Agent 1 and Agent 2

    for episode in range(1, num_episodes + 1):
        # Reset the environment for a new game
        state = env.reset()
        done = False
        agent1_reward, agent2_reward = 0, 0
        step_count = 0

        # Track episode data separately for each agent
        states_agent1, actions_agent1, rewards_agent1, dones_agent1, action_probs_agent1 = [], [], [], [], []
        states_agent2, actions_agent2, rewards_agent2, dones_agent2, action_probs_agent2 = [], [], [], [], []

        while not done:
            # Determine which agent is playing
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2

            # Get valid moves and valid action indices
            valid_moves = env.get_valid_moves()
            valid_action_indices = [r * board_size + c for r, c in valid_moves]

            # Select action based on the policy
            action, action_prob = agent.select_action(state, valid_action_indices)
            row, col = divmod(action, board_size)
            action_coordinates = (row, col)

            # Execute action
            next_state, reward, done, info = env.step(action_coordinates)

            # Track rewards and transitions separately for each agent
            if current_player == 1:
                agent1_reward += reward
                states_agent1.append(state)
                actions_agent1.append(action)
                rewards_agent1.append(reward)
                dones_agent1.append(done)
                action_probs_agent1.append(action_prob)
            else:
                agent2_reward += reward
                states_agent2.append(state)
                actions_agent2.append(action)
                rewards_agent2.append(reward)
                dones_agent2.append(done)
                action_probs_agent2.append(action_prob)

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

        # Compute advantages and returns for both agents
        # Agent 1
        if states_agent1:
            values_agent1 = [agent1.value_net(torch.FloatTensor(s).unsqueeze(0).unsqueeze(0).to(device)).item() for s in states_agent1]
            advantages_agent1, returns_agent1 = agent1.compute_advantages(rewards_agent1, values_agent1, dones_agent1)

            # Perform multiple epochs of training for Agent 1
            episode_policy_loss_agent1, episode_value_loss_agent1 = 0, 0
            for epoch in range(epochs):
                for start in range(0, len(states_agent1), batch_size):
                    end = start + batch_size

                    batch_states = torch.FloatTensor(np.array(states_agent1[start:end])).unsqueeze(1).to(device)
                    batch_actions = torch.LongTensor(actions_agent1[start:end]).to(device)
                    batch_action_probs = torch.FloatTensor(action_probs_agent1[start:end]).to(device)
                    batch_returns = returns_agent1[start:end]
                    batch_advantages = advantages_agent1[start:end]

                    # Update Agent 1
                    policy_loss_agent1, value_loss_agent1 = agent1.update(batch_states, batch_actions, batch_action_probs, batch_returns, batch_advantages)
                    episode_policy_loss_agent1 += policy_loss_agent1
                    episode_value_loss_agent1 += value_loss_agent1

            # Log average losses for Agent 1
            avg_agent1_loss = (episode_policy_loss_agent1 + episode_value_loss_agent1) / (epochs * max(len(states_agent1), 1))
            agent1_losses.append(avg_agent1_loss)
        else:
            agent1_losses.append(0)

        # Agent 2
        if states_agent2:
            values_agent2 = [agent2.value_net(torch.FloatTensor(s).unsqueeze(0).unsqueeze(0).to(device)).item() for s in states_agent2]
            advantages_agent2, returns_agent2 = agent2.compute_advantages(rewards_agent2, values_agent2, dones_agent2)

            # Perform multiple epochs of training for Agent 2
            episode_policy_loss_agent2, episode_value_loss_agent2 = 0, 0
            for epoch in range(epochs):
                for start in range(0, len(states_agent2), batch_size):
                    end = start + batch_size

                    batch_states = torch.FloatTensor(np.array(states_agent2[start:end])).unsqueeze(1).to(device)
                    batch_actions = torch.LongTensor(actions_agent2[start:end]).to(device)
                    batch_action_probs = torch.FloatTensor(action_probs_agent2[start:end]).to(device)
                    batch_returns = returns_agent2[start:end]
                    batch_advantages = advantages_agent2[start:end]

                    # Update Agent 2
                    policy_loss_agent2, value_loss_agent2 = agent2.update(batch_states, batch_actions, batch_action_probs, batch_returns, batch_advantages)
                    episode_policy_loss_agent2 += policy_loss_agent2
                    episode_value_loss_agent2 += value_loss_agent2

            # Log average losses for Agent 2
            avg_agent2_loss = (episode_policy_loss_agent2 + episode_value_loss_agent2) / (epochs * max(len(states_agent2), 1))
            agent2_losses.append(avg_agent2_loss)
        else:
            agent2_losses.append(0)

        # Log progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}: Agent1 Reward: {agent1_reward}, Agent2 Reward: {agent2_reward}, "
                  f"Win Rates -> Agent1: {agent1_win_rate:.2f}, Agent2: {agent2_win_rate:.2f}, "
                  f"Agent1 Loss: {agent1_losses[-1]:.4f}, Agent2 Loss: {agent2_losses[-1]:.4f}")

    # Save metrics
    folder = "self_play_ppo"
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(f"{folder}/win_rates.npy", win_rates)
    np.save(f"{folder}/episode_rewards.npy", episode_rewards)
    np.save(f"{folder}/agent1_losses.npy", agent1_losses)
    np.save(f"{folder}/agent2_losses.npy", agent2_losses)
    print("Training completed and metrics saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agents via Self-Play in Gomoku")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--board_size", type=int, default=8, help="Size of the Gomoku board")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Clipping parameter for PPO")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for optimizers")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs per PPO update")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
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

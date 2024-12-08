import argparse
import numpy as np
import torch
import os
from gomoku_env import GomokuEnvironment
from ppo_agent import PPOAgent
from utils import smartest_rule_based_move

print("Script has started executing.")

def train_rule_based_ppo(
    num_episodes: int = 100,
    board_size: int = 15,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 2,
    batch_size: int = 64,
    device: str = None,
    rewards_type: str = "rewards_default",
    model_save_path: str = "ppo_gomoku.pth",
    log_every: int = 10,
):
    """
    Trains a PPO agent against a rule-based agent in the Gomoku environment.

    Args:
        num_episodes (int): Number of training episodes.
        board_size (int): Size of the Gomoku board.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Clipping parameter for PPO.
        lr (float): Learning rate for optimizers.
        epochs (int): Number of epochs to optimize policy and value networks per update.
        batch_size (int): Mini-batch size for updates.
        device (str): Device to run computations ('cpu' or 'cuda').
        rewards_type (str): Name of the reward configuration YAML file (without extension).
        model_save_path (str): Path to save the trained PPO model.
        log_every (int): Number of episodes after which to log progress.
    """
    config_path = f"rewards/{rewards_type}.yml"

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize the Gomoku environment
    env = GomokuEnvironment(board_size=board_size, config_path=config_path)

    # Initialize PPO agent for Player 1
    agent1 = PPOAgent(board_size=board_size, gamma=gamma, epsilon=epsilon, lr=lr, device=device)

    # Metrics for tracking performance
    agent1_wins, rule_based_wins, draws = 0, 0, 0
    win_rates = []
    agent1_rewards_list = []
    policy_losses_per_episode, value_losses_per_episode = [], []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        agent1_reward, rule_based_reward = 0, 0
        step_count = 0

        # Track episode data
        states, actions, rewards, dones, action_probs = [], [], [], [], []

        while not done:
            current_player = env.current_player
            if current_player == 1:
                # PPO Agent's turn
                action, action_prob = agent1.select_action(state)
                row, col = divmod(action, board_size)
                action_coordinates = (row, col)
            else:
                # Rule-based Agent's turn
                action_coordinates = smartest_rule_based_move(env)

            # Execute action
            next_state, reward, done, info = env.step(action_coordinates)

            # Track rewards and transitions for PPO agent
            if current_player == 1:
                agent1_reward += reward
                # Store the transition data for training PPO agent
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                action_probs.append(action_prob)
            else:
                rule_based_reward += reward

            # Update state and step count
            state = next_state
            step_count += 1

        # Update win rates
        if "Player 1 wins" in info.get("info", ""):
            agent1_wins += 1
        elif "Player 2 wins" in info.get("info", ""):
            rule_based_wins += 1
            agent1_reward -= 1  # Penalize the PPO agent
        else:
            draws += 1

        total_games = agent1_wins + rule_based_wins + draws
        agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0
        rule_based_win_rate = rule_based_wins / total_games if total_games > 0 else 0
        win_rates.append((agent1_win_rate, rule_based_win_rate))
        agent1_rewards_list.append(round(agent1_reward, 1))

        # Compute advantages and returns for PPO agent
        values = [agent1.value_net(torch.FloatTensor(s).unsqueeze(0).unsqueeze(0).to(device)).item() for s in states]
        advantages, returns = agent1.compute_advantages(rewards, values, dones)

        # Perform multiple epochs of training for PPO agent
        episode_policy_loss, episode_value_loss = 0, 0
        batch_updates = 0
        for epoch in range(epochs):
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_updates += 1

                # Extract data for the current batch
                batch_states = torch.FloatTensor(np.array(states[start:end])).unsqueeze(1).to(device)
                batch_actions = torch.LongTensor(actions[start:end]).to(device)
                batch_action_probs = torch.FloatTensor(action_probs[start:end]).to(device)
                batch_returns = returns[start:end]
                batch_advantages = advantages[start:end]

                # Perform a single PPO update
                policy_loss, value_loss = agent1.update(batch_states, batch_actions, batch_action_probs, batch_returns, batch_advantages)

                # Accumulate losses for the episode
                episode_policy_loss += policy_loss
                episode_value_loss += value_loss

        # Track average loss for Agent 1 over the episode
        avg_policy_loss = episode_policy_loss / batch_updates
        avg_value_loss = episode_value_loss / batch_updates
        policy_losses_per_episode.append(avg_policy_loss)
        value_losses_per_episode.append(avg_value_loss)

        # Log progress every 10 episodes
        if episode % log_every == 0:
            print(f"Episode {episode}: Agent1 Reward: {agent1_reward}, Rule-Based Reward: {rule_based_reward}, "
                  f"Win Rates -> PPO: {agent1_win_rate:.2f}, Rule-Based: {rule_based_win_rate:.2f}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

    # Save metrics
    folder = f"rule_based_ppo/{rewards_type}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(f"{folder}/win_rates.npy", win_rates)
    np.save(f"{folder}/agent1_rewards.npy", agent1_rewards_list)
    np.save(f"{folder}/policy_losses.npy", policy_losses_per_episode)
    np.save(f"{folder}/value_losses.npy", value_losses_per_episode)

    # Save the final model
    agent1.save_model(f"{folder}/{model_save_path}")
    print("Training completed and metrics saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent against Rule-Based Agent in Gomoku")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--board_size", type=int, default=15, help="Size of the Gomoku board")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizers")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per PPO update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computations ('cpu' or 'cuda')")
    parser.add_argument("--config_name", type=str, default="rewards_default", help="Name of the reward configuration file (without extension)")
    parser.add_argument("--model_save_path", type=str, default="ppo_gomoku.pth", help="Path to save the model")
    args = parser.parse_args()

    train_rule_based_ppo(
        num_episodes=args.num_episodes,
        board_size=args.board_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        rewards_type=args.config_name,
        model_save_path=args.model_save_path,
    )

import argparse
import numpy as np
import torch
import os
import random  # Ensure random is imported at the top
from gomoku_env import GomokuEnvironment
from ppo_agent import PPOAgent

print("Script has started executing.")

def train_ppo_random(
    num_episodes: int = 1000,
    board_size: int = 8,
    gamma: float = 0.99,
    epsilon: float = 0.05, # changed from 0.1 -> 0.05
    lr: float = 1e-5, # changed from 1e-3 to 1e-5
    epochs: int = 4, # changed from 2 to 4
    batch_size: int = 128, # changed from 512 to 128
    device: str = None,
    rewards_type: str = "rewards_2", # changed from default to rewards_2 to help agent learn better strategies
    model_save_path: str = "ppo_gomoku.pth",
    log_every: int = 10,
):
    """
    Trains a PPO agent against a random agent in the Gomoku environment.
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
    agent1_wins, random_agent_wins, draws = 0, 0, 0
    win_rates = []
    agent1_rewards_list = []
    policy_losses_per_episode, value_losses_per_episode = [], []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        agent1_reward = 0
        step_count = 0

        # Track episode data
        states, actions, rewards, dones, action_probs = [], [], [], [], []

        while not done:
            current_player = env.current_player
            if current_player == 1:
                # PPO Agent's turn
                valid_moves = env.get_valid_moves()
                valid_action_indices = [r * board_size + c for r, c in valid_moves]
                action, action_prob = agent1.select_action(state, valid_action_indices)
                row, col = divmod(action, board_size)
                action_coordinates = (row, col)
            else:
                # Random agent's turn
                valid_moves = env.get_valid_moves()
                action_coordinates = random.choice(valid_moves)

            # Execute action
            next_state, reward, done, info = env.step(action_coordinates)

            if current_player == 1:
                # Track rewards and transitions for PPO agent
                agent1_reward += reward
                states.append(state.copy())
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                action_probs.append(action_prob)

            # Prepare for next step
            state = next_state
            step_count += 1

            if done:
                if 'info' in info:
                    # Update win counts based on the result
                    if "Player 1 wins" in info.get("info", ""):
                        agent1_wins += 1
                    elif "Player 2 wins" in info.get("info", ""):
                        random_agent_wins += 1
                        agent1_reward -= 1  # Penalize the PPO agent
                    else:
                        draws += 1
                break  # End the game

        # Update win rates
        total_games = agent1_wins + random_agent_wins + draws
        agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0
        random_agent_win_rate = random_agent_wins / total_games if total_games > 0 else 0
        win_rates.append((agent1_win_rate, random_agent_win_rate))
        agent1_rewards_list.append(round(agent1_reward, 1))

        # Compute advantages and returns for PPO agent
        if states:
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
        else:
            # No valid states collected; append zeros
            policy_losses_per_episode.append(0)
            value_losses_per_episode.append(0)

        # Log progress every 'log_every' episodes
        if episode % log_every == 0:
            print(f"Episode {episode}: Agent1 Reward: {agent1_reward}, "
                  f"Win Rates -> PPO Agent: {agent1_win_rate:.2f}, Random Agent: {random_agent_win_rate:.2f}, "
                  f"Policy Loss: {policy_losses_per_episode[-1]:.4f}, Value Loss: {value_losses_per_episode[-1]:.4f}")
            # env.render()  # Uncomment to visualize the board

    # Save metrics
    folder = f"random_ppo/{rewards_type}"
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
    parser = argparse.ArgumentParser(description="Train PPO Agent against Random Agent in Gomoku")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--board_size", type=int, default=8, help="Size of the Gomoku board")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Clipping parameter for PPO")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for optimizers")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs per PPO update")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computations ('cpu' or 'cuda')")
    parser.add_argument("--config_name", type=str, default="rewards_2", help="Name of the reward configuration file (without extension)")
    parser.add_argument("--model_save_path", type=str, default="ppo_gomoku.pth", help="Path to save the model")
    args = parser.parse_args()

    train_ppo_random(
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


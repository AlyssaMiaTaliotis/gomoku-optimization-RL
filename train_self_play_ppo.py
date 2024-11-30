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
    rollout_steps: int = 64, # 128
    epochs: int = 2, # 4
    batch_size: int = 64, # 32
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

    # print("Initializing training parameters...")

    # auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # print(f"Using device: {device}")

    # initialize the Gomoku environment
    # print("Initializing Gomoku environment...")
    env = GomokuEnvironment(board_size=board_size, config_path=config_path)
    # print("Environment initialized successfully.")

    # initialize two PPO agents (for player 1 and player 2)
    # print("Initializing PPO agents...")
    agent1 = PPOAgent(board_size=board_size, gamma=gamma, epsilon=epsilon, lr=lr, device=device)
    agent2 = PPOAgent(board_size=board_size, gamma=gamma, epsilon=epsilon, lr=lr, device=device)
    # print("PPO agents initialized successfully.")

    # metrics for tracking performance
    # print("Setting up metrics for tracking...")
    agent1_wins, agent2_wins, draws = 0, 0, 0
    win_rates, episode_rewards = [], []

    # initialize lists for tracking losses 
    policy_losses = []
    value_losses = []
    
    # print("Starting training loop...")
    for episode in range(1, num_episodes + 1): 
        # print(f"Starting Episode {episode}...")

        # reset the environment for a new game
        state = env.reset()
        # print("Environment reset.")
        done = False
        step_count = 0

        # track episode data
        states, actions, rewards, dones, action_probs = [], [], [], [], []

        while not done: 
            # determine which agent is playing (player 1 or player 2)
                current_player = env.current_player
                agent = agent1 if current_player == 1 else agent2
                # print(f"Current Player: {current_player}")

                # get action and action probability from the policy network
                action, action_prob = agent.select_action(state)
                #print(f"Action selected: {action}, Action Probability: {action_prob:.4f}")
                
                row, col = divmod(action, board_size)
                action_coordinates = (row, col)
                # print(f"Action coordinates: {action_coordinates}")

                # execute the action in the environment
                next_state, reward, done, info = env.step(action_coordinates)
                # print(f"Step result -> Reward: {reward}, Done: {done}, Info: {info}")

                # penalize invalid moves (agent must retry)
                while info.get("info") == "Invalid move":
                    # print("Invalid move detected. Retrying...")
                    states.append(state) 
                    rewards.append(reward)
                    dones.append(done)

                    # retry action 
                    action, action_prob = agent.select_action(state)
                    row, col = divmod(action, board_size)
                    action_coordinates = (row, col)
                    next_state, reward, done, info = env.step(action_coordinates)

                    # update actions and action_probs to match
                    actions.append(action)
                    action_probs.append(action_prob)

                # store the transition data for training
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                action_probs.append(action_prob)

                # update the current state
                state = next_state
                step_count += 1
        
        
        # print(f"Episode {episode} finished. Steps: {step_count}")

        # track metrics
        if "info" in info:
            if "Player 1 wins" in info["info"]:
                agent1_wins += 1
            elif "Player 2 wins" in info["info"]:
                agent2_wins += 1
            elif "Draw" in info["info"]:
                draws += 1

        # compute advantages and returns
        # print("Computing advantages and returns...")
        values = [agent1.value_net(torch.FloatTensor(s).unsqueeze(0).unsqueeze(0).to(device)).item() for s in states]
        # print(f"Lengths -> Rewards: {len(rewards)}, Values: {len(values)}, Dones: {len(dones)}")

        advantages, returns = agent1.compute_advantages(rewards, values, dones)
        # print("Advantages and returns computed.")

        # update both agents (player 1 and player 2)
        # perform multiple epochs of training for each collected batch of episode data
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs}...")

            # loop thru the data in mini-batches of size 'batch_size'
            for start in range(0, len(states), batch_size):
                # define the end of the current batch
                end = start + batch_size

                # extract the states for the current batch
                # states[start:end]` selects the states in the range [start, end)
                # convert the states into a PyTorch tensor with an additional channel dimension (unsqueeze(1))
                batch_states = torch.FloatTensor(np.array(states[start:end])).unsqueeze(1).to(device)
                
                # extract the actions taken for the current batch and convert to a PyTorch tensor
                batch_actions = torch.LongTensor(actions[start:end]).to(device)

                # extract the action probabilities (from the old policy) for the current batch
                batch_action_probs = torch.FloatTensor(action_probs[start:end]).to(device)
                
                # extract the discounted returns (G_t) for the current batch
                # these represent the actual rewards accumulated over the episode
                batch_returns = returns[start:end]

                # extract the advantages (A_t) for the current batch
                # these represent how much better each action was compared to the policy's baseline
                batch_advantages = advantages[start:end]

                # print(f"Batch shapes -> States: {batch_states.shape}, Actions: {batch_actions.shape}, Action Probs: {batch_action_probs.shape}, Returns: {len(batch_returns)}, Advantages: {len(batch_advantages)}")

                
                # perform a single PPO update using the extracted batch
                # the `update` method adjusts the policy and value networks to reduce the loss
                policy_loss, value_loss = agent1.update(batch_states, batch_actions, batch_action_probs, batch_returns, batch_advantages)
                
                # track losses
                policy_losses.append(policy_loss)
                value_losses.append(value_loss) 

                # print policy loss and value loss for each batch
                # print(f"Batch [{start}:{end}] -> Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        # save losses
        np.save("policy_losses.npy", policy_losses)
        np.save("value_losses.npy", value_losses)

        # logging
        total_games = agent1_wins + agent2_wins + draws
        win_rate_agent1 = agent1_wins / total_games if total_games > 0 else 0
        win_rate_agent2 = agent2_wins / total_games if total_games > 0 else 0
        win_rates.append((win_rate_agent1, win_rate_agent2))
        
        reward_agent1 = sum(reward for i, reward in enumerate(rewards) if (i % 2) == 0)  # rewards for Agent 1
        reward_agent2 = sum(reward for i, reward in enumerate(rewards) if (i % 2) == 1)  # rewards for Agent 2
        episode_rewards.append((reward_agent1, reward_agent2))


        if episode % 10 == 0:
            print(f"Episode {episode}: Agent1 Wins: {agent1_wins}, Agent2 Wins: {agent2_wins}, Draws: {draws}")

    # save metrics
    print("Saving metrics...")
    np.save("win_rates_ppo.npy", win_rates)
    np.save("episode_rewards_ppo.npy", episode_rewards)
    print("Training completed! Metrics saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agents via Self-Play in Gomoku")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--board_size", type=int, default=15, help="Size of the Gomoku board")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizers")
    parser.add_argument("--rollout_steps", type=int, default=64, help="Number of steps to collect before updating")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs per PPO update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computations ('cpu' or 'cuda')")

    ## should change this to rewards_2
    parser.add_argument("--config_name", type=str, default="rewards_2", help="Name of the reward configuration file (without .yml extension)")
    args = parser.parse_args()

    config_path = f"rewards/{args.config_name}.yml"

    train_ppo_self_play(
        num_episodes=args.num_episodes,
        board_size=args.board_size,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        config_path=config_path,
    )

    


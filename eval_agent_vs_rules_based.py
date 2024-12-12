import argparse
import torch
import numpy as np
from gomoku_env import GomokuEnvironment
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from utils import smartest_rule_based_move
import os
import warnings
warnings.filterwarnings("ignore")

print("Script has started executing.")

def evaluate_agent_vs_rule(
    num_episodes: int,
    board_size: int,
    agent_type: str,
    rewards_type: str,
    suffix: str,
    device: str = None,
    log_every: int = 10,
):
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize environment
    env = GomokuEnvironment(board_size=board_size)

    # Initialize the agent based on the provided agent_type
    if agent_type.lower() == "dqn":
        agent = DQNAgent(board_size=board_size, device=device)
    elif agent_type.lower() == "ppo":
        agent = PPOAgent(board_size=board_size, device=device)
    else:
        raise ValueError("Invalid agent type. Must be 'dqn' or 'ppo'.")

    # Load the trained model
    agent.load_model(f"rule_based_{agent_type}/{rewards_type}/{agent_type}_gomoku_{suffix}.pth")

    # Metrics
    agent_wins = 0
    rule_based_wins = 0
    draws = 0
    game_results = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == 1:
                # Agent's turn
                valid_moves = env.get_valid_moves()
                if agent_type.lower() == "ppo":
                    valid_action_indices = [row * board_size + col for row, col in valid_moves]
                    action_index, _ = agent.select_action(state, valid_action_indices, exploit_only=True)
                    row, col = divmod(action_index, board_size)
                else:  # DQN agent
                    action_index = agent.select_action(state, valid_moves, exploit_only=True)
                    row, col = agent.action_index_to_coordinates(action_index)
                action = (row, col)
            else:
                # Rule-based player's turn
                action = smartest_rule_based_move(env)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Prepare for next step
            state = next_state

            if done:
                if "Player 1 wins" in info.get("info", ""):
                    agent_wins += 1
                    game_results.append(1)
                elif "Player 2 wins" in info.get("info", ""):
                    rule_based_wins += 1
                    game_results.append(2)
                else:
                    draws += 1
                    game_results.append(0)
                break

        # Log progress
        if episode % log_every == 0:
            total_games = agent_wins + rule_based_wins + draws
            agent_win_rate = agent_wins / total_games if total_games > 0 else 0
            rule_based_win_rate = rule_based_wins / total_games if total_games > 0 else 0
            print(f"Episode {episode}/{num_episodes}: Agent Wins: {agent_wins}, Rule-Based Wins: {rule_based_wins}, Draws: {draws}, "
                  f"Agent Win Rate: {agent_win_rate:.2f}, Rule-Based Win Rate: {rule_based_win_rate:.2f}")

    # Final results
    print("\nFinal Results:")
    print(f"Agent Wins: {agent_wins}")
    print(f"Rule-Based Wins: {rule_based_wins}")
    print(f"Draws: {draws}")

    # Save results
    # results_folder = f"{agent_type.lower()}_vs_rule_results"
    # if not os.path.exists(results_folder):
    #     os.makedirs(results_folder)
    # np.save(os.path.join(results_folder, "game_results.npy"), game_results)
    # print(f"Game results saved to {results_folder}/game_results.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an Agent against Rule-Based Player in Gomoku")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of games to play")
    parser.add_argument("--board_size", type=int, default=8, help="Size of the Gomoku board")
    parser.add_argument("--agent_type", type=str, choices=["dqn", "ppo"], required=True, help="Type of agent ('dqn' or 'ppo')")
    parser.add_argument("--config_name", type=str, default="rewards_1", help="Name of the reward configuration file")
    parser.add_argument("--win_reward", type=str, default="10", help="Value for the win reward")
    # parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    evaluate_agent_vs_rule(
        num_episodes=args.num_episodes,
        board_size=args.board_size,
        agent_type=args.agent_type,
        rewards_type=args.config_name,
        suffix=args.win_reward,
        device=args.device,
    )

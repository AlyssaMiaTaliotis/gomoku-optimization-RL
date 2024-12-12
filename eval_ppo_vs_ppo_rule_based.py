import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import os
from gomoku_env import GomokuEnvironment
from ppo_agent import PPOAgent

print("Script has started executing.")

def ppo_vs_ppo(
    num_games: int = 100,
    board_size: int = 8,
    device: str = None,
    log_every: int = 10,
    config_name_agent1: str = "rewards_1",
    win_reward_agent1: str = "1",
    config_name_agent2: str = "rewards_1",
    win_reward_agent2: str = "1",
):
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize environment.
    # Using agent1's config to initialize the environment. Adjust if needed.
    config_path = f"rewards/{config_name_agent1}.yml"
    env = GomokuEnvironment(board_size=board_size, config_path=config_path)

    # Initialize two PPO agents
    ppo_agent1 = PPOAgent(board_size=board_size, device=device)
    ppo_agent2 = PPOAgent(board_size=board_size, device=device)

    # Construct model paths for PPO agents
    ppo_model_path_agent1 = f"rule_based_ppo/{config_name_agent1}/ppo_gomoku_{win_reward_agent1}.pth"
    ppo_model_path_agent2 = f"rule_based_ppo/{config_name_agent2}/ppo_gomoku_{win_reward_agent2}.pth"

    # Load trained models
    ppo_agent1.load_model(ppo_model_path_agent1)
    ppo_agent2.load_model(ppo_model_path_agent2)

    # Metrics
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    game_results = []

    for game in range(1, num_games + 1):
        state = env.reset()
        done = False

        while not done:
            current_player = env.current_player
            valid_moves = env.get_valid_moves()
            valid_action_indices = [r * board_size + c for r, c in valid_moves]

            if current_player == 1:
                # PPO Agent 1's turn
                action_index, _ = ppo_agent1.select_action(state, valid_action_indices, exploit_only=True)
            else:
                # PPO Agent 2's turn
                action_index, _ = ppo_agent2.select_action(state, valid_action_indices, exploit_only=True)

            row, col = divmod(action_index, board_size)
            action = (row, col)

            # Execute action
            next_state, reward, done, info = env.step(action)
            state = next_state

            if done:
                if "Player 1 wins" in info.get("info", ""):
                    agent1_wins += 1
                    game_results.append(1)
                elif "Player 2 wins" in info.get("info", ""):
                    agent2_wins += 1
                    game_results.append(2)
                else:
                    draws += 1
                    game_results.append(0)
                break

        # Log progress
        if game % log_every == 0:
            total_games = agent1_wins + agent2_wins + draws
            agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0
            agent2_win_rate = agent2_wins / total_games if total_games > 0 else 0
            print(f"Game {game}/{num_games}: Agent1 Wins: {agent1_wins}, Agent2 Wins: {agent2_wins}, Draws: {draws}, "
                  f"Agent1 Win Rate: {agent1_win_rate:.2f}, Agent2 Win Rate: {agent2_win_rate:.2f}")

    # Final results
    print("\nFinal Results:")
    print(f"PPO Agent1 Wins: {agent1_wins}")
    print(f"PPO Agent2 Wins: {agent2_wins}")
    print(f"Draws: {draws}")

    # Save results
    results_folder = "ppo_vs_ppo_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    np.save(os.path.join(results_folder, "game_results.npy"), game_results)
    print(f"Game results saved to {results_folder}/game_results.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Agent against PPO Agent in Gomoku")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--board_size", type=int, default=8, help="Size of the Gomoku board")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--config_name_agent1", type=str, default="rewards_1",
                        help="Name of the reward configuration for agent 1 (without .yml extension)")
    parser.add_argument("--win_reward_agent1", type=str, default="1",
                        help="Suffix for the model file for agent 1")
    parser.add_argument("--config_name_agent2", type=str, default="rewards_1",
                        help="Name of the reward configuration for agent 2 (without .yml extension)")
    parser.add_argument("--win_reward_agent2", type=str, default="1",
                        help="Suffix for the model file for agent 2")
    parser.add_argument("--log_every", type=int, default=10, help="Log progress every N games")

    args = parser.parse_args()

    ppo_vs_ppo(
        num_games=args.num_games,
        board_size=args.board_size,
        device=args.device,
        log_every=args.log_every,
        config_name_agent1=args.config_name_agent1,
        win_reward_agent1=args.win_reward_agent1,
        config_name_agent2=args.config_name_agent2,
        win_reward_agent2=args.win_reward_agent2,
    )

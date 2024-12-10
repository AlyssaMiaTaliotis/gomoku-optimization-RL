import argparse
import numpy as np
import torch
from gomoku_env import GomokuEnvironment
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
import os

print("Script has started executing.")

def dqn_vs_ppo(
    num_games: int = 100,
    board_size: int = 8,
    device: str = None,
    dqn_model_path: str = "rule_based_dqn/rewards_default/dqn_gomoku.pth",
    ppo_model_path: str = "rule_based_ppo/rewards_default/ppo_gomoku.pth",
    log_every: int = 10,
):
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize environment without specifying reward configuration
    env = GomokuEnvironment(board_size=board_size)

    # Initialize agents without specifying any parameters
    dqn_agent = DQNAgent(
        board_size=board_size,
        device=device,
    )
    ppo_agent = PPOAgent(
        board_size=board_size,
        device=device,
    )

    # Load trained models
    dqn_agent.load_model(dqn_model_path)
    ppo_agent.load_model(ppo_model_path)

    # Metrics
    dqn_wins = 0
    ppo_wins = 0
    draws = 0
    game_results = []

    for game in range(1, num_games + 1):
        state = env.reset()
        done = False

        while not done:
            current_player = env.current_player

            if current_player == 1:
                # DQN Agent's turn
                valid_moves = env.get_valid_moves()
                action_index = dqn_agent.select_action(state, valid_moves, exploit_only=True)
                row, col = dqn_agent.action_index_to_coordinates(action_index)
                action = (row, col)
            else:
                # PPO Agent's turn
                valid_moves = env.get_valid_moves()
                valid_action_indices = [row * board_size + col for row, col in valid_moves]
                action_index, _ = ppo_agent.select_action(state, valid_action_indices, exploit_only=True)
                row, col = divmod(action_index, board_size)
                action = (row, col)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Prepare for next step
            state = next_state

            if done:
                if "Player 1 wins" in info.get("info", ""):
                    dqn_wins += 1
                    game_results.append(1)
                elif "Player 2 wins" in info.get("info", ""):
                    ppo_wins += 1
                    game_results.append(2)
                else:
                    draws += 1
                    game_results.append(0)
                break

        # Log progress
        if game % log_every == 0:
            total_games = dqn_wins + ppo_wins + draws
            dqn_win_rate = dqn_wins / total_games if total_games > 0 else 0
            ppo_win_rate = ppo_wins / total_games if total_games > 0 else 0
            print(f"Game {game}/{num_games}: DQN Wins: {dqn_wins}, PPO Wins: {ppo_wins}, Draws: {draws}, "
                  f"DQN Win Rate: {dqn_win_rate:.2f}, PPO Win Rate: {ppo_win_rate:.2f}")

    # Final results
    print("\nFinal Results:")
    print(f"DQN Agent Wins: {dqn_wins}")
    print(f"PPO Agent Wins: {ppo_wins}")
    print(f"Draws: {draws}")

    # Save results
    results_folder = "dqn_vs_ppo_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    np.save(os.path.join(results_folder, "game_results.npy"), game_results)
    print(f"Game results saved to {results_folder}/game_results.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Agent against PPO Agent in Gomoku")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--board_size", type=int, default=8, help="Size of the Gomoku board")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--dqn_model_path", type=str, default="rule_based_dqn/rewards_default/dqn_gomoku.pth", help="Path to the trained DQN model")
    parser.add_argument("--ppo_model_path", type=str, default="rule_based_ppo/rewards_default/ppo_gomoku.pth", help="Path to the trained PPO model")
    args = parser.parse_args()

    dqn_vs_ppo(
        num_games=args.num_games,
        board_size=args.board_size,
        device=args.device,
        dqn_model_path=args.dqn_model_path,
        ppo_model_path=args.ppo_model_path,
    )

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import os
from gomoku_env import GomokuEnvironment
from dqn_agent import DQNAgent

print("Script has started executing.")

def dqn_vs_dqn(
    num_games: int = 100,
    board_size: int = 8,
    device: str = None,
    # Paths will be formed from arguments
    log_every: int = 10,
    config_name_agent1: str = "rewards_1",
    win_reward_agent1: str = "10",
    config_name_agent2: str = "rewards_1",
    win_reward_agent2: str = "10",
):
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize environment without specifying reward configuration
    # Since both are DQN and you want to specify possibly different rewards,
    # you can choose one config to initialize the environment or set a default.
    # Typically, the environment might not need separate configs for each agent.
    # We'll just use agent1's config for environment initialization.
    config_path = f"rewards/{config_name_agent1}.yml"
    env = GomokuEnvironment(board_size=board_size, config_path=config_path)

    # Initialize two DQN agents
    dqn_agent1 = DQNAgent(
        board_size=board_size,
        device=device,
    )
    dqn_agent2 = DQNAgent(
        board_size=board_size,
        device=device,
    )

    # Construct model paths using the arguments
    dqn_model_path_agent1 = f"rule_based_dqn/{config_name_agent1}/dqn_gomoku_{win_reward_agent1}.pth"
    dqn_model_path_agent2 = f"rule_based_dqn/{config_name_agent2}/dqn_gomoku_{win_reward_agent2}.pth"

    # Load trained models
    dqn_agent1.load_model(dqn_model_path_agent1)
    dqn_agent2.load_model(dqn_model_path_agent2)

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

            if current_player == 1:
                # Agent1's turn (DQN Agent 1)
                valid_moves = env.get_valid_moves()
                action_index = dqn_agent1.select_action(state, valid_moves, exploit_only=True)
                row, col = dqn_agent1.action_index_to_coordinates(action_index)
                action = (row, col)
            else:
                # Agent2's turn (DQN Agent 2)
                valid_moves = env.get_valid_moves()
                action_index = dqn_agent2.select_action(state, valid_moves, exploit_only=True)
                row, col = dqn_agent2.action_index_to_coordinates(action_index)
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
    print(f"DQN Agent1 Wins: {agent1_wins}")
    print(f"DQN Agent2 Wins: {agent2_wins}")
    print(f"Draws: {draws}")

    # Save results
    results_folder = "dqn_vs_dqn_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    np.save(os.path.join(results_folder, "game_results.npy"), game_results)
    print(f"Game results saved to {results_folder}/game_results.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Agent against DQN Agent in Gomoku")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--board_size", type=int, default=8, help="Size of the Gomoku board")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--config_name_agent1", type=str, default="rewards_1",
                        help="Name of the reward configuration for agent 1 (without .yml extension)")
    parser.add_argument("--win_reward_agent1", type=str, default="10",
                        help="Suffix for the model file for agent 1")
    parser.add_argument("--config_name_agent2", type=str, default="rewards_1",
                        help="Name of the reward configuration for agent 2 (without .yml extension)")
    parser.add_argument("--win_reward_agent2", type=str, default="10",
                        help="Suffix for the model file for agent 2")
    parser.add_argument("--log_every", type=int, default=10, help="Log progress every N games")

    args = parser.parse_args()

    dqn_vs_dqn(
        num_games=args.num_games,
        board_size=args.board_size,
        device=args.device,
        log_every=args.log_every,
        config_name_agent1=args.config_name_agent1,
        win_reward_agent1=args.win_reward_agent1,
        config_name_agent2=args.config_name_agent2,
        win_reward_agent2=args.win_reward_agent2,
    )

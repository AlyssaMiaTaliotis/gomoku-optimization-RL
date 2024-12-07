import argparse
import re
import numpy as np
import torch
from gomoku_env import GomokuEnvironment
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from utils import smartest_rule_based_move, evaluate_board, parse_human_move, get_human_name


def display_intro():
    """
    Displays the intro message at the start of the program.
    """
    print("-----------------------------------------")
    print("            WELCOME TO GOMOKU!           ")
    print("-----------------------------------------")


# def parse_human_move(move: str) -> tuple:
#     """
#     Parses a human player's move and converts it into board coordinates.
#     """
#     move = move.strip().lower()
#     match = re.match(r"^([a-o])(\d+)$|^(\d+)([a-o])$", move)
#     if not match:
#         raise ValueError("Invalid move format. Use formats like 'a1', '1a', 'A1', or '1A'.")
    
#     if match[1] and match[2]:
#         col = ord(match[1]) - ord('a')
#         row = int(match[2]) - 1
#     elif match[3] and match[4]:
#         row = int(match[3]) - 1
#         col = ord(match[4]) - ord('a')
#     else:
#         raise ValueError("Invalid move format.")
    
#     return row, col


# def get_human_name() -> str:
#     """
#     Prompts the human player to input their name.
#     """
#     while True:
#         name = input("Enter your name: ").strip()
#         if len(name) > 0 and name.isalnum():
#             return name
#         print("Invalid name. Please enter a valid alphanumeric name.")


# def smartest_rule_based_move(env: GomokuEnvironment) -> tuple:
#     """
#     Determines the best move for the rule-based player based on heuristics.
#     """
#     best_score = -float("inf")
#     best_move = None
#     opponent = 3 - env.current_player

#     for r in range(env.board_size):
#         for c in range(env.board_size):
#             if env.board[r, c] == 0:
#                 # Simulate placing the current player's stone
#                 env.board[r, c] = env.current_player
#                 score = evaluate_board(env, env.current_player)

#                 # Check if it blocks the opponent's winning move
#                 env.board[r, c] = opponent
#                 score -= evaluate_board(env, opponent)

#                 # Revert the board
#                 env.board[r, c] = 0

#                 if score > best_score:
#                     best_score = score
#                     best_move = (r, c)

#     return best_move


# def evaluate_board(env: GomokuEnvironment, player: int) -> int:
#     """
#     Evaluates the board for a specific player and assigns a score.
#     """
#     score = 0
#     for r in range(env.board_size):
#         for c in range(env.board_size):
#             if env.board[r, c] == 0:
#                 continue
#             counts = env.count_in_a_row_all_directions((r, c), player)
#             if counts["horizontal"] == 4 or counts["vertical"] == 4 or counts["diagonal"] == 4 or counts["anti_diagonal"] == 4:
#                 score += 1000  # Prioritize winning
#             elif counts["horizontal"] == 3 or counts["vertical"] == 3 or counts["diagonal"] == 3 or counts["anti_diagonal"] == 3:
#                 score += 100
#             elif counts["horizontal"] == 2 or counts["vertical"] == 2 or counts["diagonal"] == 2 or counts["anti_diagonal"] == 2:
#                 score += 10
#     return score


def computer_vs_human(env: GomokuEnvironment, agent, human_name: str, agent_type: str):
    """
    Runs the game where the computer plays against a human player.
    """
    state = env.reset()
    print(f"Welcome, {human_name}! You will play as 'O' (Player 2). The computer is 'X' (Player 1).")
    env.render()
    
    while not env.done:
        if env.current_player == 1:
            # Computer's turn
            print("Computer's turn:")
            if agent_type == "ppo":
                action, _ = agent.select_action(state)
            else:  # DQN agent
                action = agent.select_action(state)
            row, col = divmod(action, env.board_size)
            action = (row, col)
            _, _, _, info = env.step(action)
            print(f"Computer placed at {chr(col + ord('a')).upper()}{row + 1}")
            env.render()
        else:
            # Human's turn
            while True:
                try:
                    move = input(f"{human_name}, enter your move: ")
                    row, col = parse_human_move(move)
                    _, _, _, info = env.step((row, col))
                    break
                except ValueError as e:
                    print(e)
                except Exception as e:
                    print("Invalid move. Try again.")
            
            print(f"{human_name} placed at {chr(col + ord('a')).upper()}{row + 1}")
            env.render()
    
    if "Player 1 wins" in info["info"]:
        print("Game over! Computer wins!")
    elif "Player 2 wins" in info["info"]:
        print(f"Game over! {human_name} wins!")
    elif "Draw" in info["info"]:
        print("Game over! It's a draw!")


def rule_based_vs_computer(env: GomokuEnvironment, agent, agent_type: str):
    """
    Runs a game where the computer plays against a smarter rule-based player.
    """
    state = env.reset()
    print("Starting a rule-based player vs. computer match. The computer is 'X' (Player 1).")
    env.render()
    
    while not env.done:
        if env.current_player == 1:
            # Computer's turn
            print("Computer's turn:")
            if agent_type == "ppo":
                action, _ = agent.select_action(state)
            else:  # DQN agent
                action = agent.select_action(state)
            row, col = divmod(action, env.board_size)
            action = (row, col)
            _, _, _, info = env.step(action)
            print(f"Computer placed at {chr(col + ord('a')).upper()}{row + 1}")
        else:
            # Smarter rule-based player's turn
            print("Rule-based player's turn:")
            action = smartest_rule_based_move(env)
            _, _, _, info = env.step(action)
            print(f"Rule-based player placed at {chr(action[1] + ord('a')).upper()}{action[0] + 1}")
        
        env.render()
    
    if "Player 1 wins" in info["info"]:
        print("Game over! Computer wins!")
    elif "Player 2 wins" in info["info"]:
        print("Game over! Rule-based player wins!")
    elif "Draw" in info["info"]:
        print("Game over! It's a draw!")


def main():
    display_intro()
    parser = argparse.ArgumentParser(description="Evaluate Gomoku agents with different modes.")
    parser.add_argument("--mode", type=str, required=True, choices=["rule-based", "human"],
                        help="Mode of evaluation: 'rule-based' or 'human'.")
    parser.add_argument("--agent", type=str, required=True, choices=["ppo", "dqn"],
                        help="Agent to use: 'ppo' or 'dqn'.")
    parser.add_argument("--config_name", type=str, default="rewards_default",
                        help="Name of the reward configuration file (without .yml extension).")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computations ('cpu' or 'cuda').")
    args = parser.parse_args()

    # Load the Gomoku environment
    config_path = f"rewards/{args.config_name}.yml"
    env = GomokuEnvironment(config_path=config_path)

    # Auto-detect device if not specified
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load the specified agent
    if args.agent == "ppo":
        agent = PPOAgent(board_size=env.board_size, device=device)
    elif args.agent == "dqn":
        agent = DQNAgent(board_size=env.board_size, device=device)
    else:
        raise ValueError("Invalid agent type. Choose 'ppo' or 'dqn'.")

    # Evaluate based on the mode
    if args.mode == "human":
        human_name = get_human_name()
        computer_vs_human(env, agent, human_name, args.agent)
    elif args.mode == "rule-based":
        rule_based_vs_computer(env, agent, args.agent)


if __name__ == "__main__":
    main()


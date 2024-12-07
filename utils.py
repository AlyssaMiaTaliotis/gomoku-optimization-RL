import argparse
import re
import numpy as np
import torch
from gomoku_env import GomokuEnvironment
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent


# ---------------------------------------------------------------------------------

#                              RULE BASED AGENT FUNCTIONS

# ---------------------------------------------------------------------------------

def smartest_rule_based_move(env: GomokuEnvironment) -> tuple:
    """
    Determines the best move for the rule-based player based on heuristics.
    """
    best_score = -float("inf")
    best_move = None
    opponent = 3 - env.current_player

    for r in range(env.board_size):
        for c in range(env.board_size):
            if env.board[r, c] == 0:
                # Simulate placing the current player's stone
                env.board[r, c] = env.current_player
                score = evaluate_board(env, env.current_player)

                # Check if it blocks the opponent's winning move
                env.board[r, c] = opponent
                score -= evaluate_board(env, opponent)

                # Revert the board
                env.board[r, c] = 0

                if score > best_score:
                    best_score = score
                    best_move = (r, c)

    return best_move


def evaluate_board(env: GomokuEnvironment, player: int) -> int:
    """
    Evaluates the board for a specific player and assigns a score.
    """
    score = 0
    for r in range(env.board_size):
        for c in range(env.board_size):
            if env.board[r, c] == 0:
                continue
            counts = env.count_in_a_row_all_directions((r, c), player)
            if counts["horizontal"] == 4 or counts["vertical"] == 4 or counts["diagonal"] == 4 or counts["anti_diagonal"] == 4:
                score += 1000  # Prioritize winning
            elif counts["horizontal"] == 3 or counts["vertical"] == 3 or counts["diagonal"] == 3 or counts["anti_diagonal"] == 3:
                score += 100
            elif counts["horizontal"] == 2 or counts["vertical"] == 2 or counts["diagonal"] == 2 or counts["anti_diagonal"] == 2:
                score += 10
    return score


# ---------------------------------------------------------------------------------

#                              HUMAN PLAYER FUNCTIONS

# ---------------------------------------------------------------------------------

def parse_human_move(move: str) -> tuple:
    """
    Parses a human player's move and converts it into board coordinates.
    """
    move = move.strip().lower()
    match = re.match(r"^([a-o])(\d+)$|^(\d+)([a-o])$", move)
    if not match:
        raise ValueError("Invalid move format. Use formats like 'a1', '1a', 'A1', or '1A'.")
    
    if match[1] and match[2]:
        col = ord(match[1]) - ord('a')
        row = int(match[2]) - 1
    elif match[3] and match[4]:
        row = int(match[3]) - 1
        col = ord(match[4]) - ord('a')
    else:
        raise ValueError("Invalid move format.")
    
    return row, col


def get_human_name() -> str:
    """
    Prompts the human player to input their name.
    """
    while True:
        name = input("Enter your name: ").strip()
        if len(name) > 0 and name.isalnum():
            return name
        print("Invalid name. Please enter a valid alphanumeric name.")
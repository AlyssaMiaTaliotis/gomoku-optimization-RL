import argparse
import re
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from gomoku_env import GomokuEnvironment
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from utils import smartest_rule_based_move, parse_human_move, get_human_name


def display_intro():
    """
    Displays the intro message at the start of the program.
    """
    print("-----------------------------------------")
    print("            WELCOME TO GOMOKU!           ")
    print("-----------------------------------------")





def generate_heatmap(env: GomokuEnvironment, agent, agent_type: str, state: np.ndarray, config_name: str):
    board_size = env.board_size
    valid_moves = env.get_valid_moves()
    valid_action_indices = [r * board_size + c for r, c in valid_moves]

    if agent_type == "ppo":
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action_logits = agent.policy_net(state_tensor)
            action_probs = torch.softmax(action_logits, dim=1).cpu().numpy().flatten()
        masked_probs = np.zeros_like(action_probs)
        masked_probs[valid_action_indices] = action_probs[valid_action_indices]
        heatmap_data = masked_probs.reshape((board_size, board_size))
        cmap = plt.cm.Reds
    else:  # DQN agent
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor).cpu().numpy().flatten()
        masked_q_values = np.full_like(q_values, np.nan)
        masked_q_values[valid_action_indices] = q_values[valid_action_indices]
        heatmap_data = masked_q_values.reshape((board_size, board_size))
        cmap = plt.cm.Blues

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap_data, cmap=cmap, interpolation='nearest', origin='upper', alpha=0.8)
    for r in range(board_size):
        for c in range(board_size):
            if env.board[r, c] == 1:
                plt.text(c, r, 'X', ha='center', va='center', color='black', fontsize=16, fontweight='bold')
            elif env.board[r, c] == 2:
                plt.text(c, r, 'O', ha='center', va='center', color='black', fontsize=16, fontweight='bold')

    plt.xticks(range(board_size), [chr(c + ord('a')) for c in range(board_size)])
    plt.yticks(range(board_size), range(1, board_size + 1))
    plt.gca().xaxis.tick_top()
    plt.colorbar(label='Action Probability' if agent_type == "ppo" else 'Q-value')
    plt.title(f"{agent_type.upper()} Agent Heatmap ({config_name})")
    plt.tight_layout()
    plt.show()


def computer_vs_human(env: GomokuEnvironment, agent, human_name: str, agent_type: str, config_name: str, generate_heatmaps=False):
    state = env.reset()
    print(f"Welcome, {human_name}! You will play as 'O' (Player 2). The computer is 'X' (Player 1).")
    env.render()
    
    while not env.done:
        if env.current_player == 1:
            # Computer's turn
            print("Computer's turn:")
            if generate_heatmaps:
                generate_heatmap(env, agent, agent_type, state, config_name)
                
            if agent_type == "ppo":
                valid_moves = env.get_valid_moves()
                valid_action_indices = [r * env.board_size + c for r, c in valid_moves]
                action, _ = agent.select_action(state, valid_action_indices, exploit_only=True)
            else:  # DQN agent
                valid_moves = env.get_valid_moves()
                action = agent.select_action(state, valid_moves, exploit_only=True)
            row, col = divmod(action, env.board_size)
            action = (row, col)
            _, _, _, info = env.step(action)
            print(f"Computer placed at {chr(col + ord('a')).lower()}{row + 1}")
        else:
            # Human's turn
            env.render()
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
                
            print(f"{human_name} placed at {chr(col + ord('a')).lower()}{row + 1}")
            state = env.board.copy()
        
        # Update state after each move
        state = env.board.copy()
    
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
    parser.add_argument("--generate_heatmaps", action='store_true',
                        help="Include this flag to generate heatmaps during the game.")
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
        # Load the PPO model based on the reward configuration
        ppo_model_path = f"rule_based_ppo/{args.config_name}/ppo_gomoku.pth"
        agent.load_model(ppo_model_path)
    elif args.agent == "dqn":
        agent = DQNAgent(board_size=env.board_size, device=device)
        # Load the DQN model based on the reward configuration
        dqn_model_path = f"rule_based_dqn/{args.config_name}/dqn_gomoku.pth"
        agent.load_model(dqn_model_path)
    else:
        raise ValueError("Invalid agent type. Choose 'ppo' or 'dqn'.")

    # Evaluate based on the mode
    if args.mode == "human":
        human_name = get_human_name()
        computer_vs_human(env, agent, human_name, args.agent, args.config_name, generate_heatmaps=args.generate_heatmaps)
    elif args.mode == "rule-based":
        rule_based_vs_computer(env, agent, args.agent)


if __name__ == "__main__":
    main()















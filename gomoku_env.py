import numpy as np
import yaml

class GomokuEnvironment:
    def __init__(self, board_size: int = 15, config_path: str = "rewards/rewards_default.yml"):
        """
        Initializes the Gomoku environment.
        Args:
            board_size (int): The size of the board (default is 15x15).
            config_path (str): Path to the reward configuration YAML file.
        Returns: None
        """
        self.board_size = board_size
        self.reward_config = self.load_config(config_path)
        self.reset()
    
    def load_config(self, config_path: str) -> dict:
        """
        Loads the reward configuration from a YAML file.
        """

        with open(config_path, "r") as file:
            return yaml.safe_load(file)
        
    def reset(self) -> np.ndarray:
        """
        Resets the environment to its initial state.
        Returns:
            np.ndarray: The initial game board, which is a zero-filled 2D array of shape (board_size, board_size).
        """

        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  
        self.done = False
        return self.board

    def calculate_reward(self, action: tuple[int, int], win: bool = False, draw: bool = False, invalid: bool = False) -> float:
        """
        Calculates the reward based on the action and the game state.
        """
        if invalid:
            return self.reward_config.get("invalid_move", 0)
        if win:
            return self.reward_config.get("win", 0)
        if draw:
            return self.reward_config.get("draw", 0)

        # Default reward if no intermediate conditions match
        reward = self.reward_config.get("default", 0)

        # Intermediate Rewards
        intermediate_rewards = self.reward_config.get("intermediate_rewards", {})
        blocking_rewards = intermediate_rewards.get("blocking_opponent_approach", {})
        action_rewards = intermediate_rewards.get("action_approach", {})

        # Handle blocking-based rewards for the opponent
        if blocking_rewards:
            # Get counts for all directions
            opponent_counts = self.count_in_a_row_all_directions(action, player = 3 - self.current_player)

            for direction, count in opponent_counts.items():
                if count - 1 == 4:
                    reward += blocking_rewards.get("block_four", 0)
                elif count - 1 == 3:
                    reward += blocking_rewards.get("block_three", 0)

        if action_rewards:
            # Get counts for all directions
            counts = self.count_in_a_row_all_directions(action)

            for direction, count in counts.items():
                if count == 4:
                    reward += action_rewards.get("four_in_a_row", 0)
                elif count == 3:
                    reward += action_rewards.get("three_in_a_row", 0)
                elif count == 2:
                    reward += action_rewards.get("two_in_a_row", 0)

            # Check for additional conditions like double threat
            if self.creates_double_threat(action):
                reward += action_rewards.get("double_threat", 0)
            if self.places_far_from_current_group(action) and reward == 0:
                reward += action_rewards.get("far_stone_penalty", 0)

        return reward

    
    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, dict]:
        """
        Takes an action and updates the environment.
        Args:
            action (tuple[int, int]): The action to execute, represented as a (row, col) tuple.
        Returns:
            tuple:
                - np.ndarray: The current state of the board.
                - float: The reward associated with the action.
                - bool: Whether the game has ended.
                - dict: Additional information (e.g., reasons for terminal states or invalid moves).
        """
        # For invalid action -> agent will be penalized and prompted to try again
        if not self.is_valid_action(action):
            reward = self.calculate_reward(action, invalid=True)
            return self.board, reward, False, {"info": "Invalid move"}
        
        # Update the board with the current player's move
        row, col = action
        self.board[row, col] = self.current_player
        
        # Check for terminal conditions
        if self.check_win(row, col):
            reward = self.calculate_reward(action, win=True)
            self.done = True
            return self.board, reward, self.done, {"info": f"Player {self.current_player} wins"}

        if self.check_draw():
            reward = self.calculate_reward(action, draw=True)
            self.done = True
            return self.board, reward, self.done, {"info": "Draw"}
        
        # Calculate intermediate rewards
        reward = self.calculate_reward(action)

        # Switch turns
        self.current_player = 3 - self.current_player  

        return self.board, reward, self.done, {}
    
    def is_valid_action(self, action: tuple[int, int]) -> bool:
        """
        Checks if an action is valid.
        """

        row, col = action
        # Checking if the action is in the bounds of the board is for the human player, not the agent
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == 0
    
    def check_win(self, row: int, col: int) -> bool:
        """
        Checks if the current player has won the game.
        """
        # Get counts for all directions for the current player
        counts = self.count_in_a_row_all_directions((row, col))

        # Check if any direction has 5 or more stones in a row
        return any(count >= 5 for count in counts.values())

        # return self.count_in_a_row((row, col)) >= 5
    
    def check_draw(self) -> bool:
        """
        Checks if the game is a draw (i.e., the board is full).
        """
        return np.all(self.board != 0)
    
    def count_in_a_row_all_directions(self, action: tuple[int, int], player: int = None) -> dict:
        """
        Counts the number of consecutive stones for the player in all directions.
        """
        if player is None:
            player = self.current_player
        row, col = action

        # Directions: horizontal, vertical, diagonal (top-left to bottom-right), anti-diagonal
        directions = {
            "horizontal": (0, 1),
            "vertical": (1, 0),
            "diagonal": (1, 1),
            "anti_diagonal": (1, -1),
        }

        counts = {}
        for direction, (dr, dc) in directions.items():
            count = 1  
            # Forward direction
            for step in range(1, 5):
                r, c = row + dr * step, col + dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break

            # Backward direction
            for step in range(1, 5):
                r, c = row - dr * step, col - dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break

            counts[direction] = count

        return counts
    
    def creates_double_threat(self, action: tuple[int, int]) -> bool:
        """
        Checks if the current action creates a "double threat" - a line of 4 with space on both sides.
        """
        row, col = action
        player = self.board[row, col]
         # Vertical, horizontal, diagonals
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)] 

        for dr, dc in directions:
            count = 1
            empty_before = False
            empty_after = False

            # Forward direction
            for step in range(1, 5):
                r, c = row + dr * step, col + dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.board[r, c] == player:
                        count += 1
                    elif self.board[r, c] == 0:
                        empty_after = True
                        break
                    else:
                        break
            
            # Backward direction
            for step in range(1, 5):
                r, c = row - dr * step, col - dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.board[r, c] == player:
                        count += 1
                    elif self.board[r, c] == 0:
                        empty_before = True
                        break
                    else:
                        break

            if count == 4 and empty_before and empty_after:
                return True

        return False
    
    def places_far_from_current_group(self, action: tuple[int, int]) -> bool:
        """
        Checks if the action places a stone far from the player's current group of stones.
        """
        row, col = action
        player = self.current_player
        for r in range(max(0, row - 4), min(self.board_size, row + 5)):
            for c in range(max(0, col - 4), min(self.board_size, col + 5)):
                if self.board[r, c] == player and not (r == row and c == col):
                    return False
        return True

    
    def render(self) -> None:
        """
        Prints the current state of the board as a grid with cell outlines.
        Returns: None
        """
        column_labels = "     " + "   ".join("abcdefghijklmno"[:self.board_size])  
        print(column_labels) 

        horizontal_line = "   +" + "---+" * self.board_size  
        for i, row in enumerate(self.board):
            if i == 0:
                print(horizontal_line)  
            row_label = f"{i + 1:2}"  
            row_content = " | ".join("X" if x == 1 else "O" if x == 2 else " " for x in row)
            print(f"{row_label} | {row_content} |")  
            print(horizontal_line)  
        print("\n")

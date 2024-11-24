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
        Args:
            config_path (str): Path to the YAML file containing reward configurations.
        Returns:
            dict: A dictionary of reward settings loaded from the YAML file.
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
    
    def is_valid_action(self, action: tuple[int, int]) -> bool:
        """
        Checks if an action is valid.
        Args:
            action (tuple[int, int]): A tuple representing the row and column of the action.
        Returns:
            bool: True if the action is valid, False otherwise.
        """

        row, col = action
        # Checking if the action is in the bounds of the board is for the human player, not the agent
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == 0
    
    def calculate_reward(self, action: tuple[int, int], win: bool = False, draw: bool = False, invalid: bool = False) -> float:
        """
        Calculates the reward based on the action and the game state.
        Args:
            action (tuple[int, int]): The action taken, represented as a (row, col) tuple.
            win (bool): Whether the action results in a win.
            draw (bool): Whether the game ends in a draw.
            invalid (bool): Whether the action is invalid.
        Returns:
            float: The calculated reward based on the reward configuration.
        """

        if invalid:
            return self.reward_config["invalid_move"]
        if win:
            return self.reward_config["win"]
        if draw:
            return self.reward_config["draw"]
        
        # Add more intermediate rewards if specified in the configuration
        # if self.reward_config.get("intermediate_rewards"):
        #     # Example: Check if the move blocks an opponent's threat
        #     if self.check_block_opponent(action):
        #         return self.reward_config["intermediate_rewards"].get("block_opponent", 0)
            
        return self.reward_config["default"]
    
    # For invalid action -> agent will be penalized and prompted to try again
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

        # Switch turns
        self.current_player = 3 - self.current_player  
        reward = self.calculate_reward(action)
        return self.board, reward, self.done, {}
    
    def check_win(self, row: int, col: int) -> bool:
        """
        Checks if the current player has won the game.
        Args:
            row (int): The row index of the most recent move.
            col (int): The column index of the most recent move.
        Returns:
            bool: True if the current player has won, False otherwise.
        """

        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Count consecutive stones in the forward direction
            for step in range(1, 5): 
                r, c = row + dr * step, col + dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break

             # Count consecutive stones in the backward direction
            for step in range(1, 5): 
                r, c = row - dr * step, col - dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                return True
        
        return False
    
    def check_draw(self) -> bool:
        """
        Checks if the game is a draw (i.e., the board is full).
        Returns:
            bool: True if the game ends in a draw, False otherwise.
        """
        return np.all(self.board != 0)
    
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


# Example usage
env = GomokuEnvironment()
state = env.reset()
env.render()

# Simulate a few moves
env.step((7, 7))  # Player 1 places a stone
env.render()
env.step((7, 8))  # Player 2 places a stone
env.render()

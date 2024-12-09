import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Neural Network Architecture Definition
class DQN(nn.Module):
    def __init__(self, board_size: int):
        super(DQN, self).__init__()
        self.board_size=board_size
        # Convolutional Layers 
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # processes input with 32 filters, kernel size 3 and padding to preserve spatial dimesnions
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, padding=1) # processes output of conv1 with 64 filters
        self.conv3=nn.Conv2d(64, 128, kernel_size=3, padding=1) # processes output of conv2 with 128 filters
        # Fully Connected Layers
        self.fc1=nn.Linear(128 * board_size * board_size, 512) # linear layer that takes flattened output of convolutional layers and reduces it to 512 features
        self.fc2=nn.Linear(512, board_size * board_size) # final layer that utputs Q-values for ach possible action

    def forward(self, x):
        # Apply 3 convolutional layers, each followed by a ReLu activation function to introduce non-linearity
        x=torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))
        x=torch.relu(self.conv3(x))
        x=x.view(x.size(0), -1) # output fro last convolutional layer lattened to a 1D tensor to be fed into fully connected layers
        x=torch.relu(self.fc1(x))
        x=self.fc2(x) # outputs raw Q-values for each action
        return x
    # Note: these layers help in extracting spatial features from the board, such as patterns that are advantageous or disadvantageous


class DQNAgent:
    def __init__(self, board_size: int=8, memory_size: int=10000, batch_size: int=512, gamma: float=0.99, epsilon_start: float=1.0, epsilon_end: float=0.0, epsilon_decay: float=0.9995, learning_rate: float=1e-3, update_target_every: int=50, device: str="cpu"):
            """
        Initializes the DQN Agent.
        Args:
            board_size (int): The size of the Gomoku board.
            memory_size (int): The maximum size of the replay buffer.
            batch_size (int): The batch size for training.
            gamma (float): Discount factor for future rewards.
            epsilon_start (float): Initial epsilon for the epsilon-greedy policy.
            epsilon_end (float): Minimum epsilon after decay.
            epsilon_decay (float): Decay rate for epsilon.
            learning_rate (float): Learning rate for the optimizer.
            update_target_every (int): Number of episodes after which to update the target network.
            device (str): Device to run the computations on ('cpu' or 'cuda').
        """
            
            self.board_size=board_size
            self.action_size=board_size * board_size
            self.device=torch.device(device)

            # Replay memory
            self.memory=deque(maxlen=memory_size)
            self.batch_size=batch_size

            # Discount factor
            self.gamma=gamma

            # Epsilon-greedy parameters
            self.epsilon=epsilon_start
            self.epsilon_end=epsilon_end
            self.epsilon_decay=epsilon_decay

            # Networks
            self.policy_net=DQN(board_size).to(self.device)
            self.target_net=DQN(board_size).to(self.device)
            self.update_target_network()

            # Optimizer 
            self.optimizer=optim.Adam(self.policy_net.parameters(), lr=learning_rate)

            # Counter for updating target network 
            self.update_target_every=update_target_every
            self.steps_done=0

    # def select_action(self, state: np.ndarray) -> int:    # with invalid moves 
    #     """
    #     Selects an action using an epsilon-greedy policy without considering valid actions.
    #     """
    #     self.steps_done += 1
    #     if random.random() < self.epsilon:
    #         # Explore: select a random action from all possible actions
    #         action = random.randint(0, self.action_size - 1)
    #     else:
    #         # Exploit: select the action with highest Q-value
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
    #         with torch.no_grad():
    #             q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
    #         action = np.argmax(q_values)
    #     return action

    def select_action(self, state: np.ndarray, valid_moves: list, exploit_only: bool = False) -> int:
        """
        Selects an action using an epsilon-greedy policy, considering only valid moves.
        Args:
            state (np.ndarray): The current state of the board.
            valid_moves (list): A list of valid moves (row, col tuples).
        Returns:
            int: The action index corresponding to the selected move.
        """
        self.steps_done += 1
        if exploit_only:
            epsilon = 0.0  # Force exploitation
        else:
            epsilon = self.epsilon
        if random.random() < epsilon:
            # Explore: select a random action from valid moves
            valid_action_indices = [self.coordinates_to_action_index(row, col) for (row, col) in valid_moves]
            action = random.choice(valid_action_indices)
        else:
            # Exploit: select the action with highest Q-value among valid moves
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
            # Mask invalid actions by setting their Q-values to a very low number
            q_values_filtered = np.full_like(q_values, -np.inf)
            valid_action_indices = [self.coordinates_to_action_index(row, col) for (row, col) in valid_moves]
            q_values_filtered[valid_action_indices] = q_values[valid_action_indices]
            # Select the action with the highest Q-value among valid actions
            action = np.argmax(q_values_filtered)
        return action


    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Stores a transition in the replay buffer.
        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode is done.
        """
        
        self.memory.append((state, action, reward, next_state, done))

    def update_model(self):
        """
        Samples a batch from the replay buffer and updates the policy network.
        """

        if len(self.memory)< self.batch_size:
            return 
        
        minibatch=random.sample(self.memory, self.batch_size)
        states=torch.FloatTensor(np.array([m[0] for m in minibatch])).unsqueeze(1).to(self.device)
        actions=torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards=torch.FloatTensor([m[2] for m in minibatch]).unsqueeze(1).to(self.device)
        next_states=torch.FloatTensor(np.array([m[3] for m in minibatch])).unsqueeze(1).to(self.device)
        dones=torch.FloatTensor([float(m[4]) for m in minibatch]).unsqueeze(1).to(self.device)

        # Compute Q(s_t, a)
        q_values=self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            q_next=self.target_net(next_states).max(1)[0].unsqueeze(1)

        # Compute target Q-values
        q_targets=rewards+(self.gamma * q_next * (1-dones))

        # Compute loss
        loss=nn.functional.mse_loss(q_values, q_targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.decay_epsilon()

        return loss.item()

    def update_target_network(self):
        """
        Updates the target network to have the same weights as the policy network.
        """

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """
        Decays the epsilon parameter for the epsilon-greedy policy.
        """
        if self.epsilon>self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def action_index_to_coordinates(self, action: int) -> tuple:
        """
        Converts an action index to board coordinates.
        Args:
            action (int): The action index.
        Returns:
            tuple: (row, column) coordinates on the board.
        """

        row=action//self.board_size
        col=action % self.board_size
        return row, col
    
    def coordinates_to_action_index(self, row: int, col: int) -> int:
        """
        Converts board coordinates to an action index.
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            int: The action index.
        """

        return row * self.board_size + col
    
    def save_model(self, filepath: str):
        save_data = {
            "state_dict": self.policy_net.state_dict(),
            "board_size": self.board_size,
            # Add other parameters when needed
        }
        torch.save(save_data, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.board_size = checkpoint["board_size"]
        # Reinitialize the network with the correct board_size
        self.policy_net = DQN(self.board_size).to(self.device)
        self.policy_net.load_state_dict(checkpoint["state_dict"])










        


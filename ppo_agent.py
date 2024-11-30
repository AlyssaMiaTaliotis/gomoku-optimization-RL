import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Policy Network 
class PolicyNetwork(nn.Module): 
    def __init__(self, board_size: int):
        super(PolicyNetwork, self).__init__()
        self.board_size=board_size

        # convolutional layers to extract spatial patterns from the board 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # fully conected layers to output action probabilities
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)  # action logits

    def forward(self, x): 
        # forward pass thru convolutional layers w ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        action_logits = self.fc2(x)  # raw action logits
        return action_logits
    
class ValueNetwork(nn.Module): 
    def __init__(self, board_size=int):
        super(ValueNetwork, self).__init__()
        self.board_size = board_size

        # convolutional layers to extract spatial patterns
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # fully connected layers to estimate state values
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, 1)  # single value output

    def forward(self, x):
        # forward pass through convolutional layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)  # output state value
        return value

# PPO agent 
class PPOAgent: 
    def __init__(self, board_size: int, lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.2, device: str = 'cpu'):
        """
        Initializes the PPO agent with separate policy and value networks.
        Args:
            board_size (int): The size of the Gomoku board.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Clipping parameter for PPO updates.
            device (str): Device to run the computations on ('cpu' or 'cuda').
        """

        self.board_size = board_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # clipping parameter
        self.device = torch.device(device)

        # initialize policy and value networks
        self.policy_net = PolicyNetwork(board_size).to(self.device)
        self.value_net = ValueNetwork(board_size).to(self.device)

        # optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)


    # selects an action based on the policy network's output probabilities 
    # this action is chosen stochastically, which allows for exploration
    def select_action(self, state: np.ndarray):
            """
            Selects an action based on the current policy (stochastic).
            Args:
                state (np.ndarray): The current state of the board.
            Returns:
                action (int): The index of the selected action.
                action_prob (float): The probability of the selected action.
            """

            # convert the state (2D board) into a tensor 
            # - add a batch dimension (unsqueeze(0)) and a channel dimension (unsqueeze(1)) for CNN
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)

            # pass the state through the policy network to get logits for all actions
            action_logits = self.policy_net(state_tensor)

            # apply softmax to convert logits into probabilities 
            action_probs = torch.softmax(action_logits, dim=1)

            # create a categorical distribution over actions using the probabilities 
            action_distribution = torch.distributions.Categorical(action_probs)

            # sample an action from the distribution (stochastic action selection)
            action = action_distribution.sample()
            
            # return the selected action and its probability 
            return action.item(), action_probs[0, action.item()].item()
    
    # computes the advantages and returns for each state-action pair
    # advantages represent how much better an action is compared to the baseline (value function)
    def compute_advantages(self, rewards, values, dones):
        """
        Computes advantages using the Generalized Advantage Estimation (GAE) formula.
        Args:
            rewards (list): List of rewards.
            values (list): List of state values.
            dones (list): List of done flags (game over).
        Returns:
            advantages (list): Computed advantages for each time step.
            returns (list): Computed returns (discounted cumulative rewards).
        """
        advantages = []
        returns = []
        gae = 0
        next_value = 0 # future value for terminal states is 0 

        # iterate backwards thru the episode
        for t in reversed(range(len(rewards))):
            # temporal difference (TD) error: 
            # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
         
            # if the game is over (dones[t] = 1):
            #     δ_t = r_t - V(s_t)
            # the TD error only considers the immediate reward r_t, and the future reward term γ ⋅ V(s_{t+1}) is ignored
    
            # if the game is ongoing (dones[t] = 0):
            #     δ_t = r_t + γ ⋅ V(s_{t+1}) - V(s_t)
            # the TD error accounts for both the immediate reward and the discounted future value
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE: combines TD errors for all future steps 
            # GAE_t = δ_t + γ ⋅ (1 - dones[t]) ⋅ GAE_{t+1}
    
            # if the game is over (dones[t] = 1):
            #     GAE_t = δ_t
            # there is no future advantage to propagate because the game has ended
            #
            # if the game is ongoing (dones[t] = 0):
            #     GAE_t = δ_t + γ ⋅ GAE_{t+1}
            # future advantages are added recursively, weighted by the discount factor γ
            gae = delta + self.gamma * (1 - dones[t]) * gae

            # insert GAE and discounted return at the beginning 
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            # update next_value for the next iteration
            next_value = values[t]

        return torch.tensor(advantages, dtype=torch.float32).to(self.device), torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    # performs the PPO update: 
    # - updates the policy network using the clipped objective
    # - updates the value network to minimize prediction error 
    def update(self, states, actions, old_action_probs, returns, advantages):
        """
        Updates the policy and value networks using PPO loss functions.
        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions taken.
            old_action_probs (torch.Tensor): Probabilities of actions under the old policy.
            returns (torch.Tensor): Discounted cumulative rewards (returns).
            advantages (torch.Tensor): Advantages for each state-action pair.
        """
        # forward pass thru the policy network to get new action probabilities 
        action_logits = self.policy_net(states)
        action_probs = torch.softmax(action_logits, dim=1)
        action_distribution = torch.distributions.Categorical(action_probs)
        new_action_probs = action_distribution.log_prob(actions)

        # compute the probability ratio: r_t = new_prob / old_prob
        ratio = torch.exp(new_action_probs - torch.log(old_action_probs))  # probability ratio
        
        # PPO clipping objective: minimize the smaller of unclipped and clipped terms 
        # L_CLIP = min(r_t * A_t, clip(r_t, 1 - epsilon, 1 + epsilon) * A_t)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # update the policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # value loss:
        # L_value = (1 / N) * sum_{i=1}^N (V(s_i) - G_i)^2
        value_preds = self.value_net(states).view(-1) # reshape predictions to a 1D tensor
        value_loss = nn.MSELoss()(value_preds, returns)
        
        # update the value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return policy_loss.item(), value_loss.item()
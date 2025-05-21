import numpy as np
import random

class QAgent:
    def __init__(self, 
                 grid_dim_x: int,               # Full x-dimension of the Q-table (e.g., 13 for FourRooms)
                 grid_dim_y: int,               # Full y-dimension of the Q-table (e.g., 13 for FourRooms)
                 max_initial_packages: int,     # Max number of packages at the start of an episode (e.g., 1 for S1, 4 for S2/S3)
                 num_actions: int,              # Number of possible actions (e.g., 4 for UP, DOWN, LEFT, RIGHT)
                 learning_rate: float = 0.1,    # Alpha: how much new information overrides old
                 discount_factor: float = 0.99, # Gamma: importance of future rewards
                 epsilon_start: float = 1.0,    # Initial exploration rate
                 epsilon_end: float = 0.05,     # Minimum exploration rate
                 epsilon_decay_rate: float = 0.0001, # Rate for linear decay of epsilon per epoch
                 q_initial_value: float = 0.0   # Initial value for Q-table entries (0.0 for standard, >0 for optimistic)
                ):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = self.epsilon_start
        self.num_actions = num_actions

        # Q-table dimensions:
        # x_coord, y_coord: Use grid_dim_x, grid_dim_y directly (e.g., 0-12 for a 13x13 grid)
        # packages_left_k: from 0 to max_initial_packages. So, size is max_initial_packages + 1.
        self.k_dimension_q_table = max_initial_packages + 1
        
        self.q_table = np.full(
            (grid_dim_x, grid_dim_y, self.k_dimension_q_table, num_actions), 
            float(q_initial_value) # Ensure it's float
        )
        
        # For terminal states where all packages are collected (k=0), 
        # Q-values should ideally be 0 as there's no future reward from these states.
        # This is especially important if using optimistic initialization.
        if self.k_dimension_q_table > 0: # Check if k=0 is a valid index
             self.q_table[:, :, 0, :] = 0.0

    def choose_action(self, state: tuple) -> int:
        """
        Chooses an action based on the current state using epsilon-greedy strategy.
        State is (x, y, k), where x,y are 1-based from environment, k is packages remaining.
        """
        x, y, k = state
        k_idx = int(k) # k is directly used as the index for the 'packages remaining' dimension

        # Boundary check for state validity against Q-table dimensions.
        # Assumes x, y from environment are within [0, grid_dim_x-1] and [0, grid_dim_y-1]
        # For FourRooms, x,y are 1-11, so q_table dims 13x13 is fine.
        if not (0 <= x < self.q_table.shape[0] and \
                0 <= y < self.q_table.shape[1] and \
                0 <= k_idx < self.q_table.shape[2]):
            # This case should ideally not be hit if environment interactions are correct.
            # print(f"Warning: State {state} (k_idx {k_idx}) out of Q-table bounds in choose_action. Choosing random action.")
            return random.randrange(self.num_actions)

        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)  # Explore: choose a random action
        else:
            # Exploit: choose the best action from Q-table for the current state
            return np.argmax(self.q_table[x, y, k_idx, :])

    def update_q_table(self, state: tuple, action: int, reward: float, next_state: tuple, is_env_terminal: bool):
        """
        Updates the Q-value for the (state, action) pair.
        State and next_state are (x, y, k).
        is_env_terminal: True if the environment signals the end of the episode.
        """
        x, y, k = state
        next_x, next_y, next_k = next_state
        
        k_idx = int(k)
        next_k_idx = int(next_k)

        # Boundary checks before Q-table access
        if not (0 <= x < self.q_table.shape[0] and 0 <= y < self.q_table.shape[1] and 0 <= k_idx < self.q_table.shape[2] and \
                0 <= next_x < self.q_table.shape[0] and 0 <= next_y < self.q_table.shape[1] and 0 <= next_k_idx < self.q_table.shape[2]):
            # print(f"Warning: State or Next State out of Q-table bounds during Q-update. Skipping. State: {state}, Next: {next_state}")
            return

        current_q_value = self.q_table[x, y, k_idx, action]
        
        max_future_q = 0.0
        if not is_env_terminal: # If not a terminal state, consider future rewards
            max_future_q = np.max(self.q_table[next_x, next_y, next_k_idx, :])
            
        # Q-learning update rule
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q_value)
        self.q_table[x, y, k_idx, action] = new_q_value

    def decay_epsilon(self, current_epoch: int):
        """
        Linearly decays epsilon based on the current epoch.
        """
        self.epsilon = self.epsilon_start - current_epoch * self.epsilon_decay_rate
        self.epsilon = max(self.epsilon_end, self.epsilon) # Clamp at epsilon_end

    def get_greedy_action(self, state: tuple) -> int:
        """
        Returns the action with the highest Q-value for the given state (pure exploitation).
        """
        x, y, k = state
        k_idx = int(k)

        # Boundary check
        if not (0 <= x < self.q_table.shape[0] and \
                0 <= y < self.q_table.shape[1] and \
                0 <= k_idx < self.q_table.shape[2]):
            # print(f"Warning: State {state} (k_idx {k_idx}) out of Q-table bounds in get_greedy_action. Defaulting to action 0.")
            return 0 # Fallback action

        # If k=0 (all packages collected), Q-values are 0. np.argmax on zeros returns 0.
        return np.argmax(self.q_table[x, y, k_idx, :])
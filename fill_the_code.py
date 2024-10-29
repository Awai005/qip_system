import numpy as np
import math

class Controller:
    def __init__(self, state_bins=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], action_size=15, epsilon=1, 
                 epsilon_min=0.1, alpha=0.1, gamma=0.95, a_max=60, pretrained_q_table_path=None, training=True):
        """
        Initialize the Q-learning controller with varying state discretizations and LQR control.

        Args:
            state_bins (list): Number of bins for each state variable.
            action_size (int): Number of discretized actions.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum epsilon.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            a_max (float): Maximum action magnitude.
            pretrained_q_table_path (str, optional): Path to a pre-trained Q-table file.
            training (bool): Flag indicating whether to allow Q-table updates.
        """
        # Load system dynamics matrices
        self.A = np.load('A_matrix.npy')
        self.B = np.load('B_matrix.npy')

        # Set the goal state (equilibrium)
        self.goal_state = np.zeros(self.A.shape[0])

        # System parameters
        self.cart_mass = 1.3  # kg
        self.link_masses = [0.2000, 0.2500, 0.4300, 0.8100]  # kg
        self.link_lengths = [0.1500, 0.2000, 0.4000, 0.8000]  # m
        self.moi = [0.0015, 0.0033, 0.0229, 0.1728]  # Moment of inertia

        # Action space
        self.a_max = a_max  # Max force
        self.actions = np.linspace(-a_max, a_max, action_size)  # Discretized force

        # Q-learning parameters
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum epsilon
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor



        # Initialize Q-table based on the varying bin sizes
        self.state_bins = [np.linspace(-10, 10, bins) for bins in state_bins]
        q_table_shape = [len(bins) for bins in self.state_bins] + [action_size]
        
        if pretrained_q_table_path:
            try:
                self.q_table = np.load(pretrained_q_table_path)
                print(f"Loaded pre-trained Q-table from {pretrained_q_table_path}")
                # Set epsilon to minimum to prioritize exploitation
                self.epsilon = self.epsilon_min
            except FileNotFoundError:
                print(f"Pre-trained Q-table file {pretrained_q_table_path} not found. Initializing new Q-table.")
                self.q_table = np.zeros(q_table_shape)
        else:
            self.q_table = np.zeros(q_table_shape)

        # LQR controller setup
        self.K = self.solve_lqr(A=self.A, B=self.B, Q=np.eye(self.A.shape[0]), R=np.eye(self.B.shape[1]))

        # Set the bounds for cart movement
        self.cart_min_pos = -5
        self.cart_max_pos = 5

        # Threshold to switch to LQR control (in radians)
        self.lqr_threshold_angle = np.radians(5)  # 45 degrees threshold
        self.lqr_switch_back_threshold = np.radians(30)  # 90 degrees threshold for switching back

        # LQR mode flag
        self.in_lqr_mode = False

        # Training flag
        self.training = training

    def solve_lqr(self, A, B, Q, R, num_iters=100):
        """
        Solve the discrete-time algebraic Riccati equation (DARE) iteratively
        to find the optimal LQR gain matrix K.
        """
        P = Q
        for _ in range(num_iters):
            P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            if np.allclose(P, P_next):
                break
            P = P_next
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def wrap_angle(self, angle):
        """Wrap the angle between [-π, π] using math.remainder."""
        return math.remainder(angle, 2 * np.pi)

    def discretize_state(self, state):
        """
        Discretize continuous state into a tuple of discrete indices based on varying bin sizes.
        """
        state_discrete = []
        for i, s in enumerate(state):
            # For angles, ensure they are wrapped before discretization
            if i in [2, 4, 6, 8]:  # Indices for angles
                s = self.wrap_angle(s)
            # Clip the state to the bin range to prevent out-of-bounds
            s = np.clip(s, self.state_bins[i][0], self.state_bins[i][-1])
            state_discrete.append(np.digitize(s, self.state_bins[i]) - 1)
        # Ensure indices are within Q-table bounds
        state_discrete = tuple(np.clip(s, 0, len(self.state_bins[i])-1) for i, s in enumerate(state_discrete))
        #print(f"Discretized State: {state_discrete}")  # Debugging
        return state_discrete



    def select_action_q_learning(self, state):
        """
        Select an action using an epsilon-greedy policy for Q-Learning.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: Random action
            return np.random.choice(self.actions)
        else:
            # Exploitation: Choose action with the highest Q-value
            state_discrete = self.discretize_state(state)
            action_index = np.argmax(self.q_table[state_discrete])
            return self.actions[action_index]

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning update rule.
        """
        if not self.training:
            return  # Skip updates if not training

        state_discrete = self.discretize_state(state)
        next_state_discrete = self.discretize_state(next_state)
        action_indices = np.where(self.actions == action)[0]

        if len(action_indices) == 0:
            print(f"Action {action} not found in action space.")
            return

        action_index = action_indices[0]

        best_next_action_index = np.argmax(self.q_table[next_state_discrete])
        target = reward + self.gamma * self.q_table[next_state_discrete + (best_next_action_index,)]
        self.q_table[state_discrete + (action_index,)] += self.alpha * (target - self.q_table[state_discrete + (action_index,)])

        # Decay epsilon for exploration-exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.995

    def compute_reward(self, state, action):
        """
        Compute the immediate reward based on the current state and action.
        Customize this method based on your specific reward structure.
        """
        cart_pos = state[0]
        cart_vel = state[1]

        angular_vels = state[3::2]  # Assuming angular velocities are at odd indices
        
        state[2] = self.wrap_angle(state[2])
        state[4] = self.wrap_angle(state[4])
        state[6] = self.wrap_angle(state[6])
        state[8] = self.wrap_angle(state[8])
        angles = state[2::2]
        # Reward components
        reward = 0

        reward -= 0.1 * (abs(cart_pos)) 
      
        
        reward -= 0.1 * sum(abs(angular_vel) for angular_vel in angular_vels)
        

        reward += 1 * sum((1 + math.cos(angle)) / 2 for angle in angles)  # Encourage upright positions
        
        if all(abs(angle - math.pi) < 0.1 for angle in angles):  # You can adjust the threshold (0.1) as needed
            reward -= 5 


        return reward

    def compute_force(self, state: list, reward=None, next_state=None) -> float:
        """
        Compute and return the action (force) based on the state and update if needed.
        Switch to LQR control if all link angles are within the threshold.
        """
        theta1 = self.wrap_angle(state[2])
        theta2 = self.wrap_angle(state[4])
        theta3 = self.wrap_angle(state[6])
        theta4 = self.wrap_angle(state[8])
        
        if self.in_lqr_mode:
            # Check if we need to switch back to Q-learning (if the first link angle exceeds 90 degrees)
            if abs(theta1) > self.lqr_switch_back_threshold:
                self.in_lqr_mode = False
                print("Switching back to Q-Learning")
            else:
                # Continue using LQR
                state_array = np.array(state)
                action_array = -np.dot(self.K, (state_array - self.goal_state))
                action = action_array[0]  # Assuming the first element is the force applied to the cart
                print("Continuing with LQR")
                return action

        # Check if we should switch to LQR (if all link angles are within 45 degrees)
        if all(abs(angle) < self.lqr_threshold_angle for angle in [theta1, theta2, theta3, theta4]):
            self.in_lqr_mode = True
            print("Switching to LQR")
            state_array = np.array(state)
            action_array = -np.dot(self.K, (state_array - self.goal_state))
            action = action_array[0]  # Assuming the first element is the force applied to the cart
        else:
            # Use Q-learning for swing-up
            action = self.select_action_q_learning(state)

        cart_position = state[0]

        # Limit cart movement to be within the range [-5, 5]
        if cart_position <= self.cart_min_pos:
            action = self.a_max  # Prevent moving further left
        elif cart_position >= self.cart_max_pos:
            action = -self.a_max  # Prevent moving further right

        # Clip the action to the system's force constraints [-60, 60]
        action = np.clip(action, -self.a_max, self.a_max)

        # Apply the threshold: If |force| < 0.1, set force to 0
        if abs(action) < 0.1:
            action = 0.0

        # If we have both reward and next_state (during training), update the Q-tables
        if reward is not None and next_state is not None:
            if not self.in_lqr_mode:
                # Update Q-Learning Controller's Q-table
                computed_reward = self.compute_reward(state, action)
                self.update_q_table(state, action, computed_reward, next_state)

        return action

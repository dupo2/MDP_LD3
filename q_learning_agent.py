import random
import time
from typing import List, Tuple, Dict

from mdp_environment import GridWorldMDP, State, Action

# --- Type Aliases ---
QTable = Dict[State, Dict[Action, float]]
Policy = Dict[State, Action]
Path = List[Tuple[State, Action, float]]

class QLearningAgent:
    """An agent that solves the MDP using the Q-Learning algorithm."""
    
    def __init__(self, mdp: GridWorldMDP, gamma: float, alpha: float, epsilon: float):
        self.mdp = mdp
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.Q: QTable = {s: {a: 0.0 for a in self.mdp.actions} for s in self.mdp.states}
        
    def choose_action(self, state: State) -> Action:
        """Chooses an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)

    def learn(self, state: State, action: Action, reward: float, next_state: State):
        """Updates the Q-value for a given state-action pair using the Q-learning rule."""
        old_q_value = self.Q[state][action]
        
        if self.mdp.is_terminal(next_state):
            max_q_next = 0.0
        else:
            max_q_next = max(self.Q[next_state].values())
            
        temporal_difference = reward + self.gamma * max_q_next - old_q_value
        new_q_value = old_q_value + self.alpha * temporal_difference
        
        self.Q[state][action] = new_q_value

    def get_policy(self) -> Policy:
        """Extracts the optimal policy from the learned Q-table."""
        policy: Policy = {}
        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                policy[state] = None
            else:
                policy[state] = max(self.Q[state], key=self.Q[state].get)
        return policy

    # --- MODIFIED SIMULATE_PATH METHOD ---
    def simulate_path(self, start_state: State, max_steps: int = 50) -> Path:
        """Simulates a path using the derived optimal policy in a DETERMINISTIC way."""
        print("\n--- Q-LEARNING OPTIMAL PATH SIMULATION LOG ---")
        policy = self.get_policy()
        total_reward = 0
        path = [(start_state, "Start", total_reward)]
        
        current_state = start_state
        
        for i in range(max_steps):
            if self.mdp.is_terminal(current_state):
                print(f"Goal reached in {i} steps!")
                break
            
            action = policy.get(current_state)
            if action is None:
                print(f"Error: No action defined for non-terminal state {current_state}.")
                break

            # Use the deterministic flag to visualize the pure policy
            next_state, step_reward = self.mdp.step(current_state, action, deterministic=True)

            total_reward += step_reward
            path.append((next_state, action, total_reward))
            print(f"Step {i+1}: Action='{action}', Step Reward={step_reward}, Total Reward={total_reward}")
            current_state = next_state
        
        print(f"\n--- FINAL TOTAL REWARD (Q-Learning): {total_reward} ---\n")
        return path
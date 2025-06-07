# approx_q_agent.py
import random
import numpy as np
from typing import List, Tuple, Dict, Optional

# ... (other imports are the same)
from mdp_environment import GridWorldMDP, State, Action
from feature_extractor import FeatureExtractor

Path = List[Tuple[State, Action, float]]

class ApproxQLearningAgent:
    # __init__ is the same, just updates the num_features
    def __init__(self, mdp: GridWorldMDP, feature_extractor: FeatureExtractor,
                 gamma: float, alpha: float, epsilon: float):
        self.mdp = mdp
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        num_features = self.feature_extractor.get_num_features()
        self.weights = np.zeros(num_features)

    # --- MODIFIED: Pass last_pos to get_features ---
    def get_q_value(self, state: State, action: Action, last_pos: Optional[Tuple[int, int]]) -> float:
        features = self.feature_extractor.get_features(state, action, last_pos)
        return np.dot(self.weights, features)

    # --- MODIFIED: Pass last_pos when choosing an action ---
    def choose_action(self, state: State, last_pos: Optional[Tuple[int, int]]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions) 
        else:
            return self.get_best_action(state, last_pos)

    # --- MODIFIED: Pass last_pos when finding the best action ---
    def get_best_action(self, state: State, last_pos: Optional[Tuple[int, int]]) -> Action:
        q_values = {action: self.get_q_value(state, action, last_pos) for action in self.mdp.actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    # --- MODIFIED: Pass state/position info to the learn function ---
    def learn(self, state: State, action: Action, reward: float, next_state: State, last_pos: Optional[Tuple[int, int]]):
        # We need the current position to calculate the Q-value of the next state
        current_pos = state[0]
        
        prediction = self.get_q_value(state, action, last_pos)
        
        if self.mdp.is_terminal(next_state):
            target = reward
        else:
            # When considering the next state, its "last_pos" is our "current_pos"
            q_values_next = [self.get_q_value(next_state, a_next, current_pos) for a_next in self.mdp.actions]
            target = reward + self.gamma * max(q_values_next)
            
        difference = target - prediction
        
        features = self.feature_extractor.get_features(state, action, last_pos)
        self.weights += self.alpha * difference * np.array(features)
    
    def simulate_path(self, start_state: State, max_steps: int = 500) -> Path:
        # --- MODIFIED: Track last_pos during simulation ---
        print("\n--- APPROX-QL OPTIMAL PATH SIMULATION LOG ---")
        path = [(start_state, "Start", 0)]
        current_state = start_state
        last_pos = None
        total_reward = 0

        for i in range(max_steps):
            if self.mdp.is_terminal(current_state):
                print(f"Goal reached in {i} steps!")
                break
            
            action = self.get_best_action(current_state, last_pos)
            
            # Update last_pos before taking the step
            last_pos_for_next_step = current_state[0]
            
            next_state, step_reward = self.mdp.step(current_state, None, action, deterministic=True)
            total_reward += step_reward
            path.append((next_state, action, total_reward))
            
            current_state = next_state
            last_pos = last_pos_for_next_step # Update last_pos for the next iteration
        
        return path
# File: approx_q_agent.py

import random
import numpy as np
from typing import List, Tuple, Dict

# --- Imports from other project files ---
from mdp_environment import GridWorldMDP, State, Action
from feature_extractor import FeatureExtractor

# --- DEFINE THE 'Path' TYPE ALIAS HERE ---
Path = List[Tuple[State, Action, float]]


class ApproxQLearningAgent:
    def __init__(self, mdp: GridWorldMDP, feature_extractor: FeatureExtractor,
                 gamma: float, alpha: float, epsilon: float):
        self.mdp = mdp
        self.feature_extractor = feature_extractor
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # We need a sample state to determine the number of features.
        dummy_state = ((1, 1), tuple([False] * len(mdp.boxes)))
        num_features = len(self.feature_extractor.get_features(dummy_state))
        
        self.weights: Dict[Action, np.ndarray] = {
            action: np.zeros(num_features) for action in self.mdp.actions
        }

    def get_q_value(self, state: State, action: Action) -> float:
        features = self.feature_extractor.get_features(state)
        return np.dot(self.weights[action], features)

    def choose_action(self, state: State) -> Action:
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            q_values = {action: self.get_q_value(state, action) for action in self.mdp.actions}
            return max(q_values, key=q_values.get)

    def learn(self, state: State, action: Action, reward: float, next_state: State):
        prediction = self.get_q_value(state, action)
        
        if self.mdp.is_terminal(next_state):
            target = reward
        else:
            q_values_next = [self.get_q_value(next_state, a) for a in self.mdp.actions]
            target = reward + self.gamma * max(q_values_next)
            
        difference = target - prediction
        features = self.feature_extractor.get_features(state)
        self.weights[action] += self.alpha * difference * np.array(features)

    def simulate_path(self, start_state: State, max_steps: int = 500) -> Path:
        print("\n--- APPROX-QL OPTIMAL PATH SIMULATION LOG ---")
        
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        total_reward = 0
        path = [(start_state, "Start", total_reward)]
        current_state = start_state
        
        for i in range(max_steps):
            if self.mdp.is_terminal(current_state):
                print(f"Goal reached in {i} steps!")
                break
            
            action = self.choose_action(current_state)
            next_state, step_reward = self.mdp.step(current_state, action, deterministic=True)
            total_reward += step_reward
            path.append((next_state, action, total_reward))
            current_state = next_state
        
        print(f"\n--- FINAL TOTAL REWARD (Approx Q-Learning): {total_reward:.2f} ---\n")
        self.epsilon = original_epsilon
        return path
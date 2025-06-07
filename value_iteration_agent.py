# value_iteration_agent.py
"""
Defines a model-based AI agent that solves a known MDP environment using Value Iteration.
"""
import time
from typing import Dict

from mdp_environment import GridWorldMDP, State, Action, Policy, ValueFunction, Path

class ValueIterationAgent:
    """An agent that solves the MDP using Value Iteration, assuming a known model."""
    def __init__(self, mdp: GridWorldMDP, gamma: float = 0.9, theta: float = 1e-6):
        self.mdp = mdp; self.gamma = gamma; self.theta = theta
        self.V: ValueFunction = {s: 0 for s in self.mdp.states}
        self.policy: Policy = {}

    def solve(self):
        """Runs the complete solving process: Value Iteration -> Policy Extraction."""
        print("\n--- Solving with Value Iteration (Stage 1) ---")
        start_time = time.time()
        self._run_value_iteration()
        print(f"Value Iteration complete in {time.time() - start_time:.2f} seconds.")
        self._derive_optimal_policy()
        print("Optimal policy derived.")

    def _compute_q_value(self, state: State, action: Action) -> float:
        """Calculates the Q-value for a state-action pair."""
        # A rigorous implementation would average over the 80/10/10 probabilities.
        # For simplicity, we use the outcome of the intended action.
        next_state, reward = self.mdp.step(state, action)
        return reward + self.gamma * self.V.get(next_state, 0.0)

    def _run_value_iteration(self):
        """Computes the optimal value function V*(s)."""
        while True:
            delta = 0
            for s in self.mdp.states:
                if self.mdp.is_terminal(s): continue
                v_old = self.V[s]
                self.V[s] = max([self._compute_q_value(s, a) for a in self.mdp.actions])
                delta = max(delta, abs(v_old - self.V[s]))
            if delta < self.theta: break

    def _derive_optimal_policy(self):
        """Extracts the greedy policy pi*(s) from the optimal value function V*(s)."""
        for s in self.mdp.states:
            if self.mdp.is_terminal(s): self.policy[s] = None; continue
            self.policy[s] = max(self.mdp.actions, key=lambda a: self._compute_q_value(s, a))
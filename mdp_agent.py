import time
from typing import List, Tuple, Dict

# Import the MDP environment
from mdp_environment import GridWorldMDP, State, Action

# Define types for clarity
Policy = Dict[State, Action]
ValueFunction = Dict[State, float]
Path = List[Tuple[State, Action, float]]

# ValueIterationAgent
class ValueIterationAgent:
    # A Value Iteration Agent for solving Grid World MDPs.
    def __init__(self, mdp: GridWorldMDP, gamma: float = 0.9, theta: float = 1e-6):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.V: ValueFunction = {s: 0 for s in self.mdp.states}
        self.policy: Policy = {}

    # Solve the MDP using value iteration
    def solve(self):
        print("\nStarting Value Iteration...")
        start_time = time.time()
        self._run_value_iteration()
        end_time = time.time()
        print(f"Value Iteration complete in {end_time - start_time:.2f} seconds.")

        print("Deriving optimal policy...")
        self._derive_optimal_policy()
        print("Optimal policy derived.")
    
    # Run the value iteration algorithm to compute the optimal value function
    def _run_value_iteration(self):
        # Initialize the value function for all states
        while True:
            delta = 0
            for s in self.mdp.states:
                if self.mdp.is_terminal(s):
                    continue
                
                # Store the old value for convergence check
                v = self.V[s]
                action_values = [
                    self.mdp.step(s, a)[1] + self.gamma * self.V[self.mdp.step(s, a)[0]]
                    for a in self.mdp.actions
                ]
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            if delta < self.theta:
                break
    
    # Derive the optimal policy from the computed value function
    def _derive_optimal_policy(self):
        """Extracts the optimal policy pi*(s) from the value function V*(s)."""
        for s in self.mdp.states:
            if self.mdp.is_terminal(s):
                self.policy[s] = None
                continue

            # Find the action that maximizes the expected value
            best_action = max(
                self.mdp.actions,
                key=lambda a: self.mdp.step(s, a)[1] + self.gamma * self.V[self.mdp.step(s, a)[0]]
            )
            self.policy[s] = best_action

    # Simulate a path using the derived optimal policy
    def simulate_path(self, start_state: State, max_steps: int = 50) -> Path:
        """Simulates a path using the derived optimal policy."""
        print("\n--- OPTIMAL PATH SIMULATION LOG ---")
        total_reward = 0
        path = [(start_state, "Start", total_reward)]
        
        current_state = start_state
        
        # Iterate through the maximum number of steps
        for i in range(max_steps):
            if self.mdp.is_terminal(current_state):
                print(f"Goal reached in {i} steps!")
                break

            # Get the action from the policy
            action = self.policy.get(current_state)
            if action is None:
                print(f"Error: No action defined for non-terminal state {current_state}.")
                break
            
            # Take the action and get the next state and reward
            next_state, step_reward = self.mdp.step(current_state, action)
            total_reward += step_reward
            # Log the step
            path.append((next_state, action, total_reward))
            print(f"Step {i+1}: Action='{action}', Step Reward={step_reward}, Total Reward={total_reward}")
            current_state = next_state
        
        # Print the final total reward
        print(f"\n--- FINAL TOTAL REWARD: {total_reward} ---\n")
        return path
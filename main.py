import time
from mdp_environment import GridWorldMDP, generate_random_box_locations
from mdp_agent import ValueIterationAgent
from q_learning_agent import QLearningAgent
from environment_visualizer import visualize_path

def main():
    # --- Common Environment Setup ---
    grid_size = 4
    obstacles = {(1, 3), (2, 1), (3, 2)}
    initial_robot_pos = (1, 1)
    num_boxes = 4
    
    print("--- Setting up Environment ---")
    random_boxes = generate_random_box_locations(
        grid_size=grid_size,
        num_boxes=num_boxes,
        obstacles=obstacles,
        start_pos=initial_robot_pos
    )
    print(f"Boxes placed at: {list(random_boxes.keys())}")
    initial_state = (initial_robot_pos, tuple([False] * num_boxes))

    # =========================================================================
    # STAGE 1: VALUE ITERATION (KNOWN, DETERMINISTIC MODEL)
    # =========================================================================
    print("\n\n--- STAGE 1: VALUE ITERATION (KNOWN MODEL) ---")
    mdp_env_vi = GridWorldMDP(grid_size=grid_size, obstacles=obstacles, boxes=random_boxes, intended_action_prob=1.0)
    vi_agent = ValueIterationAgent(mdp=mdp_env_vi, gamma=0.9)
    vi_agent.solve()
    vi_path = vi_agent.simulate_path(start_state=initial_state)

    # =========================================================================
    # STAGE 2: Q-LEARNING (UNKNOWN, STOCHASTIC MODEL)
    # =========================================================================
    print("\n\n--- STAGE 2: Q-LEARNING (MODEL-FREE) ---")
    mdp_env_ql = GridWorldMDP(grid_size=grid_size, obstacles=obstacles, boxes=random_boxes, intended_action_prob=0.8)
    
    # --- MODIFIED: Q-Learning Hyperparameters with Epsilon Decay ---
    ql_agent = QLearningAgent(
        mdp=mdp_env_ql,
        gamma=0.9,
        alpha=0.1,
        epsilon=1.0  # Start with high exploration, will be decayed
    )
    
    # Training parameters
    num_episodes = 30000  # Increased episodes for better convergence
    max_steps_per_episode = 100
    
    # Epsilon decay parameters
    epsilon_start = 1.0
    epsilon_end = 0.01  # Ensure a minimum amount of exploration
    epsilon_decay_rate = 0.9997 # Adjust for desired decay speed

    print(f"Starting Q-Learning training for {num_episodes} episodes...")
    start_time = time.time()
    current_epsilon = epsilon_start

    # --- MODIFIED: Training Loop with Epsilon Decay ---
    for episode in range(num_episodes):
        current_state = initial_state
        ql_agent.epsilon = current_epsilon  # Set agent's current exploration rate

        for step in range(max_steps_per_episode):
            action = ql_agent.choose_action(current_state)
            next_state, reward = mdp_env_ql.step(current_state, action)
            ql_agent.learn(current_state, action, reward, next_state)
            current_state = next_state
            
            if mdp_env_ql.is_terminal(current_state):
                break
        
        # Decay epsilon after each episode
        current_epsilon = max(epsilon_end, current_epsilon * epsilon_decay_rate)
        
        if (episode + 1) % 5000 == 0:
            print(f"Completed episode {episode + 1}/{num_episodes}, Current Epsilon: {current_epsilon:.4f}")

    end_time = time.time()
    print(f"Q-Learning training finished in {end_time - start_time:.2f} seconds.")

    # Get the final path from the trained Q-Learning agent
    ql_path = ql_agent.simulate_path(start_state=initial_state)

    # =========================================================================
    # FINAL VISUALIZATION & COMPARISON
    # =========================================================================
    print("\n\n--- PATH COMPARISON ---")

    print("Visualizing optimal path from VALUE ITERATION...")
    visualize_path(vi_path, random_boxes)

    print("Visualizing optimal path from Q-LEARNING...")
    visualize_path(ql_path, random_boxes)
    
    print("Comparison finished.")

if __name__ == "__main__":
    main()
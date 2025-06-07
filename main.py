from mdp_environment import GridWorldMDP, generate_random_box_locations
from mdp_agent import ValueIterationAgent
from environment_visualizer import visualize_path

def main():
    grid_size = 4
    obstacles = {(1, 3), (2, 1), (3, 2)}
    initial_robot_pos = (1, 1)

    # 1. Generate random box locations for this run
    print("--- Setting up Environment ---")
    random_boxes = generate_random_box_locations(
        grid_size=grid_size,
        num_boxes=4,
        obstacles=obstacles,
        start_pos=initial_robot_pos
    )
    print(f"Boxes placed at: {list(random_boxes.keys())}")
    
    # 2. Create the environment instance with the randomized boxes
    mdp_env = GridWorldMDP(grid_size=grid_size, obstacles=obstacles, boxes=random_boxes)

    # 3. Create the agent and solve the MDP
    agent = ValueIterationAgent(mdp=mdp_env, gamma=0.9)
    agent.solve()

    # 4. Simulate the optimal path from the start state
    initial_state = (initial_robot_pos, (False, False, False, False))
    final_path = agent.simulate_path(start_state=initial_state)

    # 5. Visualize the result, passing the random box locations to the visualizer
    print("Starting visualization...")
    visualize_path(final_path, random_boxes)
    print("Visualization finished.")


if __name__ == "__main__":
    main()
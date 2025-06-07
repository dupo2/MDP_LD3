# main.py

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Set, Tuple, Dict, List, Optional

# --- Import your other project files ---
from mdp_environment import GridWorldMDP, generate_random_box_locations
from environment_visualizer import visualize_path
from feature_extractor import FeatureExtractor
from approx_q_agent import ApproxQLearningAgent

def generate_random_obstacles(grid_size: int, num_obstacles: int, start_pos: Tuple[int, int], box_locs: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Generates a set of random, valid obstacle locations."""
    possible_locations = {
        (x, y) for x in range(1, grid_size + 1) for y in range(1, grid_size + 1)
    }
    invalid_locations = {start_pos} | box_locs
    possible_locations -= invalid_locations
    
    if len(possible_locations) < num_obstacles:
        raise ValueError("Not enough valid locations to place all obstacles.")
    
    return set(random.sample(list(possible_locations), num_obstacles))

def plot_rewards(rewards_per_episode, window_size=100, title="Rewards per Episode"):
    """Smoothes the rewards and plots them."""
    if len(rewards_per_episode) < window_size:
        print("Not enough data to plot.")
        return
    smoothed_rewards = [np.mean(rewards_per_episode[i-window_size:i]) for i in range(window_size, len(rewards_per_episode))]
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_rewards)
    plt.title(title)
    plt.xlabel(f"Episode (smoothed over {window_size} episodes)")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()

def main():
    print("\n--- STAGE 3: Q-LEARNING WITH FUNCTION APPROXIMATION ---")

    # 1. Set up the environment with MORE obstacles
    grid_size = 30 # Increased grid size for more space with double obstacles
    num_boxes = 5
    
    # --- THIS IS THE LINE TO CHANGE ---
    # We'll double the number of obstacles again for a very challenging environment.
    num_obstacles = 60
    
    initial_robot_pos = (1, 1)

    random_boxes = generate_random_box_locations(
        grid_size, num_boxes, set(), initial_robot_pos
    )
    
    obstacles = generate_random_obstacles(
        grid_size, num_obstacles, initial_robot_pos, set(random_boxes.keys())
    )
    print(f"Generated {len(obstacles)} random obstacles in a {grid_size}x{grid_size} grid.")
    
    initial_state = (initial_robot_pos, tuple([False] * num_boxes))

    mdp_env = GridWorldMDP(
        grid_size=grid_size, obstacles=obstacles, boxes=random_boxes, intended_action_prob=0.8
    )

    # 2. Create the agent
    feature_ext = FeatureExtractor(
        grid_size=grid_size, num_boxes=num_boxes, boxes_data=random_boxes, obstacles=obstacles
    )
    agent = ApproxQLearningAgent(
        mdp=mdp_env,
        feature_extractor=feature_ext,
        gamma=0.99,
        alpha=0.01,
        epsilon=1.0
    )

    # 3. Run the training loop
    num_episodes = 5000
    max_steps_per_episode = 600
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9996
    all_rewards = []
    
    print(f"Starting training for {num_episodes} episodes...")
    start_time = time.time()
    current_epsilon = epsilon_start

    for episode in range(num_episodes):
        agent.epsilon = current_epsilon
        current_state = initial_state
        episode_reward = 0
        last_pos: Optional[Tuple[int, int]] = None

        for step in range(max_steps_per_episode):
            action = agent.choose_action(current_state, last_pos)
            next_state, reward = mdp_env.step(current_state, None, action)
            agent.learn(current_state, action, reward, next_state, last_pos)
            episode_reward += reward
            last_pos = current_state[0]
            current_state = next_state
            
            if mdp_env.is_terminal(current_state):
                break
        
        all_rewards.append(episode_reward)
        current_epsilon = max(epsilon_end, current_epsilon * epsilon_decay)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {current_epsilon:.4f}")

    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
    plot_rewards(all_rewards, window_size=200)

    # 4. Visualize the final policy
    final_path = agent.simulate_path(initial_state)
    visualize_path(final_path, random_boxes, obstacles, grid_size)

if __name__ == "__main__":
    main()
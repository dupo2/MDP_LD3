# mdp_environment.py

import random
from typing import List, Tuple, Dict, Set

State = Tuple[Tuple[int, int], Tuple[bool, ...]]
Action = str

def generate_random_box_locations(grid_size: int, num_boxes: int, obstacles: Set, start_pos: Tuple) -> Dict[Tuple[int, int], int]:
    # This function is unchanged and correct.
    possible_locations = {
        (x, y) for x in range(1, grid_size + 1) for y in range(1, grid_size + 1)
        if (x, y) not in obstacles and (x, y) != start_pos
    }
    if len(possible_locations) < num_boxes:
        raise ValueError("Not enough valid locations to place all boxes.")
    chosen_locations = random.sample(list(possible_locations), num_boxes)
    return {pos: i for i, pos in enumerate(chosen_locations)}

class GridWorldMDP:
    def __init__(self, grid_size: int, obstacles: Set[Tuple[int, int]], boxes: Dict[Tuple[int, int], int], intended_action_prob: float = 1.0):
        # This is unchanged
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.boxes = boxes
        self.box_locations = {v: k for k, v in boxes.items()}
        self.actions = ['up', 'down', 'left', 'right']
        self.intended_action_prob = intended_action_prob
        self.perpendicular_actions = {
            'up': ['left', 'right'], 'down': ['left', 'right'],
            'left': ['up', 'down'], 'right': ['up', 'down']
        }

    def is_terminal(self, state: State) -> bool:
        return all(state[1])

    # --- MODIFIED STEP FUNCTION WITH "STUCK" PENALTY ---
    def step(self, state: State, last_state: State, action: Action, deterministic: bool = False) -> Tuple[State, float]:
        if self.is_terminal(state):
            return state, 0

        position, box_statuses = state
        
        actual_action = action
        if not deterministic and random.random() > self.intended_action_prob:
            actual_action = random.choice(self.perpendicular_actions[action])
        
        move_map = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
        dx, dy = move_map.get(actual_action, (0, 0))
        next_position = (position[0] + dx, position[1] + dy)
        
        # Check if the move is invalid (into a wall or obstacle)
        nx, ny = next_position
        is_stuck = False
        if not (1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size) or next_position in self.obstacles:
            # If the move is invalid, the agent stays in the same position
            final_position = position
            is_stuck = True
        else:
            final_position = next_position

        new_box_statuses = list(box_statuses)
        just_collected_a_box = False
        if final_position in self.boxes:
            box_index = self.boxes[final_position]
            if not new_box_statuses[box_index]:
                new_box_statuses[box_index] = True
                just_collected_a_box = True
        next_state = (final_position, tuple(new_box_statuses))

        # --- NEW REWARD LOGIC ---
        if just_collected_a_box:
            return next_state, 500.0 if self.is_terminal(next_state) else 100.0
        
        # Apply a harsh penalty if the agent tried to move but couldn't
        if is_stuck:
            return next_state, -10.0 # Harsh penalty for being stuck

        # Small cost for every valid, non-box-collecting move
        return next_state, -1.0
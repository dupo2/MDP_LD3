import random
from typing import Tuple, Dict, Set

State = Tuple[Tuple[int, int], Tuple[bool, ...]]
Action = str

# Generate random box locations for the grid world.
def generate_random_box_locations(grid_size: int, num_boxes: int, obstacles: Set, start_pos: Tuple) -> Dict[Tuple[int, int], int]:
    
    # Generate a set of possible locations for boxes, excluding obstacles and the start position
    possible_locations = {
        (x, y) for x in range(1, grid_size + 1) for y in range(1, grid_size + 1)
        if (x, y) not in obstacles and (x, y) != start_pos
    }
    # Ensure we have enough locations to place the boxes
    if len(possible_locations) < num_boxes:
        raise ValueError("Not enough valid locations to place all boxes.")
    chosen_locations = random.sample(list(possible_locations), num_boxes)
    # Create a mapping of box locations to their indices
    return {pos: i for i, pos in enumerate(chosen_locations)}

# This class represents the GridWorld MDP environment with a modified step function
class GridWorldMDP:
    def __init__(self, grid_size: int, obstacles: Set[Tuple[int, int]], boxes: Dict[Tuple[int, int], int], intended_action_prob: float = 1.0):
        # Initialize the GridWorld MDP environment
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.boxes = boxes
        self.box_locations = {v: k for k, v in boxes.items()}
        # self.actionts is a set of all possible actions the agent can take
        self.actions = ['up', 'down', 'left', 'right']
        self.intended_action_prob = intended_action_prob

        # self.perpendicular_actions is a mapping of actions to their perpendicular alternatives
        self.perpendicular_actions = {
            'up': ['left', 'right'], 'down': ['left', 'right'],
            'left': ['up', 'down'], 'right': ['up', 'down']
        }

    # is_terminal checks if the state is terminal (i.e., all boxes are collected)
    def is_terminal(self, state: State) -> bool:
        return all(state[1])


    def step(self, state: State, last_state: State, action: Action, deterministic: bool = False) -> Tuple[State, float]:
        if self.is_terminal(state):
            # If the state is terminal, return the state and a reward of 0
            return state, 0

        # state defines the current position of the agent and the status of the boxes
        position, box_statuses = state
        
        actual_action = action
        if not deterministic and random.random() > self.intended_action_prob:
            actual_action = random.choice(self.perpendicular_actions[action])

        # move_map defines how each action translates to a change in position
        move_map = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
        dx, dy = move_map.get(actual_action, (0, 0))
        next_position = (position[0] + dx, position[1] + dy)
        
        # Check if the move is invalid (into a wall or obstacle)
        nx, ny = next_position

        # If the next position is out of bounds or an obstacle, the agent is considered stuck
        is_stuck = False
        if not (1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size) or next_position in self.obstacles:
            # If the move is invalid, the agent stays in the same position
            final_position = position
            is_stuck = True
        else:
            final_position = next_position

        # new_box_statuses is a copy of the current box statuses
        new_box_statuses = list(box_statuses)
        just_collected_a_box = False

        # Check if the agent is trying to collect a box
        if final_position in self.boxes:
            box_index = self.boxes[final_position]
            # If the box is not already collected, mark it as collected
            if not new_box_statuses[box_index]:
                new_box_statuses[box_index] = True
                just_collected_a_box = True

        # If the agent is not collecting a box, it just moves without changing box statuses
        next_state = (final_position, tuple(new_box_statuses))

        # If the agent collected a box, return a high reward
        if just_collected_a_box:
            return next_state, 500.0 if self.is_terminal(next_state) else 100.0
        
        # Apply a harsh penalty if the agent tried to move but couldn't
        if is_stuck:
            return next_state, -10.0

        # Small cost for every valid, non-box-collecting move
        return next_state, -5.0
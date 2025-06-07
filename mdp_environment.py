import itertools
import random
from typing import List, Tuple, Dict, Set

# --- Type Aliases specific to the environment ---
State = Tuple[Tuple[int, int], Tuple[bool, ...]]
Action = str

def generate_random_box_locations(grid_size: int, num_boxes: int, obstacles: Set, start_pos: Tuple) -> Dict[Tuple[int, int], int]:
    """Generates a dictionary of random, valid box locations."""
    possible_locations = {
        (x, y)
        for x in range(1, grid_size + 1)
        for y in range(1, grid_size + 1)
        if (x, y) not in obstacles and (x, y) != start_pos
    }
    if len(possible_locations) < num_boxes:
        raise ValueError("Not enough valid locations to place all boxes.")
    
    chosen_locations = random.sample(list(possible_locations), num_boxes)
    return {pos: i for i, pos in enumerate(chosen_locations)}

class GridWorldMDP:
    """Encapsulates the rules, states, and transitions of the Grid World environment."""
    def __init__(self, grid_size: int, obstacles: Set[Tuple[int, int]], boxes: Dict[Tuple[int, int], int], intended_action_prob: float = 1.0):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.boxes = boxes
        self.actions = ['up', 'down', 'left', 'right']
        self.states = self._generate_all_states()
        
        self.intended_action_prob = intended_action_prob
        self.perpendicular_actions = {
            'up': ['left', 'right'],
            'down': ['left', 'right'],
            'left': ['up', 'down'],
            'right': ['up', 'down']
        }

    def _generate_all_states(self) -> List[State]:
        """Generates the complete set of all possible states."""
        valid_positions = [
            (x, y)
            for x in range(1, self.grid_size + 1)
            for y in range(1, self.grid_size + 1)
            if (x, y) not in self.obstacles
        ]
        box_status_combinations = list(itertools.product([False, True], repeat=len(self.boxes)))
        return [(pos, status) for pos in valid_positions for status in box_status_combinations]

    def is_terminal(self, state: State) -> bool:
        """Checks if a state is terminal (all boxes collected)."""
        return all(state[1])

    # --- MODIFIED STEP FUNCTION ---
    def step(self, state: State, action: Action, deterministic: bool = False) -> Tuple[State, float]:
        """
        Calculates the next state and the reward.
        Can be run in deterministic mode to turn off stochasticity.
        """
        if self.is_terminal(state):
            return state, 0

        actual_action = action
        # If not in deterministic mode, apply stochasticity
        if not deterministic:
            if random.random() > self.intended_action_prob:
                actual_action = random.choice(self.perpendicular_actions[action])
        
        position, box_statuses = state
        move_map = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
        dx, dy = move_map.get(actual_action, (0, 0)) # Use the potentially altered action
        next_position = (position[0] + dx, position[1] + dy)

        nx, ny = next_position
        # If move is invalid (out of bounds or into obstacle), stay in place
        final_position = next_position if 1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size and next_position not in self.obstacles else position

        new_box_statuses = list(box_statuses)
        if final_position in self.boxes:
            box_index = self.boxes[final_position]
            if not new_box_statuses[box_index]:
                new_box_statuses[box_index] = True
    
        next_state = (final_position, tuple(new_box_statuses))
    
        just_collected_a_box = sum(next_state[1]) > sum(state[1])
        if self.is_terminal(next_state) and just_collected_a_box:
            reward = 100
        elif just_collected_a_box:
            reward = 10
        else:
            reward = -1
            
        return next_state, reward
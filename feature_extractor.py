# feature_extractor.py

from typing import Tuple, List, Dict, Set, Optional

State = Tuple[Tuple[int, int], Tuple[bool, ...]]
Action = str

class FeatureExtractor:
    def __init__(self, grid_size: int, num_boxes: int, boxes_data: Dict[Tuple[int, int], int], obstacles: Set[Tuple[int, int]]):
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.box_locations = {v: k for k, v in boxes_data.items()}
        self.obstacles = obstacles
        self.actions = ['up', 'down', 'left', 'right']

    # --- MODIFIED SIGNATURE: Now accepts last_pos ---
    def get_features(self, state: State, action: Action, last_pos: Optional[Tuple[int, int]]) -> List[float]:
        player_pos, box_statuses = state
        
        move_map = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
        dx, dy = move_map[action]
        next_pos = (player_pos[0] + dx, player_pos[1] + dy)
        
        features = []
        
        # Feature 1: Bias
        features.append(1.0)
        
        # Feature 2: Collision
        nx, ny = next_pos
        is_collision = 1.0 if not (1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size) or next_pos in self.obstacles else 0.0
        features.append(is_collision)

        # Find nearest box
        uncollected_boxes = [self.box_locations[i] for i in range(self.num_boxes) if not box_statuses[i]]
        nearest_box_pos = None
        if uncollected_boxes:
            min_dist = float('inf')
            for box_pos in uncollected_boxes:
                dist = abs(player_pos[0] - box_pos[0]) + abs(player_pos[1] - box_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_box_pos = box_pos
        
        # Feature 3: Gets closer to nearest box
        gets_closer = 0.0
        if nearest_box_pos and not is_collision:
            dist_before = abs(player_pos[0] - nearest_box_pos[0]) + abs(player_pos[1] - nearest_box_pos[1])
            dist_after = abs(next_pos[0] - nearest_box_pos[0]) + abs(next_pos[1] - nearest_box_pos[1])
            if dist_after < dist_before:
                gets_closer = 1.0
        features.append(gets_closer)

        # Feature 4: Collects a box
        is_box_collected = 0.0
        if next_pos in self.box_locations and not is_collision:
            box_id = self.box_locations[next_pos]
            if not box_statuses[box_id]:
                is_box_collected = 1.0
        features.append(is_box_collected)

        # --- NEW FEATURE 5: Discourage immediate reversal ---
        # This feature is 1.0 if the action would return the agent to its previous position.
        is_reversal = 0.0
        if last_pos and next_pos == last_pos:
            is_reversal = 1.0
        features.append(is_reversal)

        return features

    def get_num_features(self) -> int:
        # Now we have 5 features
        return 5
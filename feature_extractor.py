# feature_extractor.py

from collections import deque
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
        # The cache is essential for performance. It will store BFS results.
        self.bfs_cache = {}

    def _bfs(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> Optional[int]:
        """A private helper to run the actual BFS calculation."""
        if start_pos == end_pos:
            return 0
        queue = deque([(start_pos, 0)])
        visited = {start_pos}
        while queue:
            (x, y), dist = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (x + dx, y + dy)
                if next_pos == end_pos:
                    return dist + 1
                if (1 <= next_pos[0] <= self.grid_size and 1 <= next_pos[1] <= self.grid_size and
                        next_pos not in self.obstacles and next_pos not in visited):
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))
        return None

    def get_path_distance(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> Optional[int]:
        """
        Public method to get the true path distance using a cache (memoization).
        This prevents re-calculating the same path over and over.
        """
        # Create a consistent key for the cache (A->B is same as B->A)
        key = tuple(sorted((start_pos, end_pos)))
        if key not in self.bfs_cache:
            # If the distance is not in the cache, calculate it and store it.
            self.bfs_cache[key] = self._bfs(start_pos, end_pos)
        # Return the cached result.
        return self.bfs_cache[key]

    def get_features(self, state: State, action: Action, last_pos: Optional[Tuple[int, int]]) -> List[float]:
        player_pos, box_statuses = state
        
        move_map = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}
        dx, dy = move_map[action]
        next_pos = (player_pos[0] + dx, player_pos[1] + dy)
        
        features = []
        
        # Feature 1: Bias (Unchanged)
        features.append(1.0)
        
        # Feature 2: Collision (Unchanged)
        nx, ny = next_pos
        is_collision = 1.0 if not (1 <= nx <= self.grid_size and 1 <= ny <= self.grid_size) or next_pos in self.obstacles else 0.0
        features.append(is_collision)

        # --- THIS IS THE KEY MODIFICATION ---
        # Find the true path distance to the nearest box from the CURRENT and NEXT positions.
        uncollected_boxes = [self.box_locations[i] for i in range(self.num_boxes) if not box_statuses[i]]
        
        dist_before = float('inf')
        if uncollected_boxes:
            all_dists = [self.get_path_distance(player_pos, box_pos) for box_pos in uncollected_boxes]
            reachable_dists = [d for d in all_dists if d is not None]
            if reachable_dists:
                dist_before = min(reachable_dists)

        dist_after = float('inf')
        if not is_collision and uncollected_boxes:
            all_dists_after = [self.get_path_distance(next_pos, box_pos) for box_pos in uncollected_boxes]
            reachable_dists_after = [d for d in all_dists_after if d is not None]
            if reachable_dists_after:
                dist_after = min(reachable_dists_after)
                
        # Feature 3: Gets closer to nearest box (using TRUE PATH DISTANCE)
        gets_closer = 1.0 if dist_after < dist_before else 0.0
        features.append(gets_closer)
        # --- END MODIFICATION ---

        # Feature 4: Collects a box (Unchanged)
        is_box_collected = 0.0
        if next_pos in self.box_locations and not is_collision:
            box_id = self.box_locations[next_pos]
            if not box_statuses[box_id]:
                is_box_collected = 1.0
        features.append(is_box_collected)

        # Feature 5: Discourage immediate reversal (Unchanged)
        is_reversal = 0.0
        if last_pos and next_pos == last_pos:
            is_reversal = 1.0
        features.append(is_reversal)

        return features

    def get_num_features(self) -> int:
        return 5
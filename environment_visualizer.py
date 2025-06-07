import pygame
from typing import List, Tuple, Dict

# --- Type Aliases for clarity ---
State = Tuple[Tuple[int, int], Tuple[bool, ...]]
Path = List[Tuple[State, str, float]]
Boxes = Dict[Tuple[int, int], int]

# --- Game Configuration ---
GRID_SIZE = 4
GRID_WINDOW_SIZE = 640
INFO_PANEL_HEIGHT = 80
TILE_SIZE = GRID_WINDOW_SIZE // GRID_SIZE
GRID_LINE_WIDTH = 3
WINDOW_SIZE = (GRID_WINDOW_SIZE, GRID_WINDOW_SIZE + INFO_PANEL_HEIGHT)

# --- Colors ---
COLOR_BACKGROUND = "white"; COLOR_GRID = "grey"; COLOR_PLAYER = "blue"
COLOR_OBSTACLE = "red"; COLOR_BOX = "saddlebrown"; COLOR_BOX_COLLECTED = "forestgreen"
COLOR_INFO_PANEL = "gainsboro"; COLOR_TEXT = "black"

# --- Helper Function ---
def grid_to_pixels(grid_x: int, grid_y: int) -> Tuple[int, int]:
    """Converts 1-indexed grid coordinates to top-left pixel coordinates."""
    pixel_x = (grid_x - 1) * TILE_SIZE
    pixel_y = (GRID_SIZE - grid_y) * TILE_SIZE
    return (pixel_x, pixel_y)

# --- Sprite Classes ---
class BaseSprite(pygame.sprite.Sprite):
    """A base class for all game sprites to reduce boilerplate code."""
    def __init__(self, *groups):
        super().__init__(*groups)
        self.image = pygame.surface.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        self.rect = self.image.get_rect()

class Player(BaseSprite):
    """Represents the player/robot sprite and its movement."""
    def __init__(self, grid_pos: Tuple[int, int], *groups):
        super().__init__(*groups)
        self.radius = TILE_SIZE // 4
        pygame.draw.circle(self.image, COLOR_PLAYER, (TILE_SIZE // 2, TILE_SIZE // 2), self.radius)
        self.rect.topleft = grid_to_pixels(*grid_pos)
        
        self.pixel_pos = pygame.math.Vector2(self.rect.center)
        self.target_pixel_pos = pygame.math.Vector2(self.rect.center)
        self.moving = False
        self.speed = 250 * (TILE_SIZE / 128)

    def set_target(self, grid_pos: Tuple[int, int]):
        pixel_x, pixel_y = grid_to_pixels(*grid_pos)
        self.target_pixel_pos.xy = (pixel_x + TILE_SIZE // 2, pixel_y + TILE_SIZE // 2)
        if self.pixel_pos != self.target_pixel_pos:
            self.moving = True

    def update(self, dt: float) -> bool:
        """Moves the player and returns True upon arrival."""
        if not self.moving:
            return False
            
        self.pixel_pos.move_towards_ip(self.target_pixel_pos, self.speed * dt)
        self.rect.center = self.pixel_pos
        
        if self.pixel_pos.distance_to(self.target_pixel_pos) < 1:
            self.pixel_pos.xy = self.target_pixel_pos.xy
            self.rect.center = self.pixel_pos
            self.moving = False
            return True
        return False

class Obstacle(BaseSprite):
    """A static obstacle sprite (X)."""
    def __init__(self, grid_pos: Tuple[int, int], *groups):
        super().__init__(*groups)
        size = TILE_SIZE // 2; offset = TILE_SIZE // 4; width = 5
        tl, tr = (offset, offset), (offset + size, offset)
        bl, br = (offset, offset + size), (offset + size, offset + size)
        pygame.draw.line(self.image, COLOR_OBSTACLE, tl, br, width)
        pygame.draw.line(self.image, COLOR_OBSTACLE, tr, bl, width)
        self.rect.topleft = grid_to_pixels(*grid_pos)

class Box(BaseSprite):
    """A box sprite that can be collected."""
    def __init__(self, grid_pos: Tuple[int, int], box_id: int, *groups):
        super().__init__(*groups)
        self.id = box_id
        size = TILE_SIZE // 2; offset = TILE_SIZE // 4
        self.box_rect = pygame.Rect(offset, offset, size, size)
        self.rect.topleft = grid_to_pixels(*grid_pos)
        self.update_status(False)

    def update_status(self, is_collected: bool):
        self.image.fill((0, 0, 0, 0)) # Clear to transparent
        if is_collected:
            pygame.draw.rect(self.image, COLOR_BOX_COLLECTED, self.box_rect)
        else:
            pygame.draw.rect(self.image, COLOR_BOX, self.box_rect, 5)

# --- World and Renderer Classes ---
class World:
    """Manages all game objects and their states."""
    def __init__(self, path_data: Path, boxes_data: Boxes):
        self.all_sprites = pygame.sprite.Group()
        
        # Create static obstacles
        for pos in {(1, 3), (2, 1), (3, 2)}:
            Obstacle(pos, self.all_sprites)

        # Create boxes based on the provided data
        self.box_objects = []
        for pos, box_id in sorted(boxes_data.items(), key=lambda item: item[1]):
            self.box_objects.append(Box(pos, box_id, self.all_sprites))

        # Create the player at the starting position of the path
        initial_state, _, _ = path_data[0]
        self.player = Player(initial_state[0], self.all_sprites)

    def update_state(self, current_path_step: Tuple[State, str, float]):
        """Updates the world's objects based on a step from the path."""
        state, _, _ = current_path_step
        player_pos, box_statuses = state
        
        self.player.set_target(player_pos)
        for i, box in enumerate(self.box_objects):
            box.update_status(box_statuses[i])

    def update_kinematics(self, dt: float) -> bool:
        """Updates sprite positions and returns True if the player has arrived."""
        self.all_sprites.update(dt)
        return not self.player.moving

    def draw(self, surface: pygame.Surface):
        self.all_sprites.draw(surface)

class Renderer:
    """Handles all drawing operations for the game."""
    def __init__(self, window: pygame.Surface):
        self.window = window
        self.info_font = pygame.font.Font(None, 32)

    def draw(self, world: World, step_info: Tuple[int, str, float]):
        self.window.fill(COLOR_BACKGROUND)
        self.draw_grid()
        world.draw(self.window)
        self.draw_info_panel(step_info)
        pygame.display.update()

    def draw_grid(self):
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.window, COLOR_GRID, (0, i * TILE_SIZE), (GRID_WINDOW_SIZE, i * TILE_SIZE), GRID_LINE_WIDTH)
            pygame.draw.line(self.window, COLOR_GRID, (i * TILE_SIZE, 0), (i * TILE_SIZE, GRID_WINDOW_SIZE), GRID_LINE_WIDTH)

    def draw_info_panel(self, step_info: Tuple[int, str, float]):
        step, action, reward = step_info
        panel_rect = pygame.Rect(0, GRID_WINDOW_SIZE, WINDOW_SIZE[0], INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.window, COLOR_INFO_PANEL, panel_rect)

        texts = [f"Step: {step}", f"Action: {action}", f"Total Reward: {reward}"]
        for i, text in enumerate(texts):
            surf = self.info_font.render(text, True, COLOR_TEXT)
            self.window.blit(surf, (20, GRID_WINDOW_SIZE + 5 + i * 25))

# --- Main Game Orchestrator ---
class Game:
    """Orchestrates the game loop, state updates, and rendering."""
    def __init__(self, path_data: Path, boxes_data: Boxes):
        pygame.init()
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("MDP Path Visualization")
        self.clock = pygame.time.Clock()
        
        self.path_data = path_data
        self.current_step = 0

        self.world = World(path_data, boxes_data)
        self.renderer = Renderer(self.window)

    def _advance_to_next_step(self):
        """Moves to the next step in the path if available."""
        if self.current_step < len(self.path_data) - 1:
            self.current_step += 1
            self.world.update_state(self.path_data[self.current_step])

    def run(self):
        """The main game loop."""
        running = True
        # Set the initial state
        self.world.update_state(self.path_data[0])

        while running:
            dt = self.clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Update sprite kinematics (movement)
            player_has_arrived = self.world.update_kinematics(dt)

            # If player arrived, advance the logical state of the world
            if player_has_arrived:
                self._advance_to_next_step()

            # Render the current state
            step_info = (self.current_step, self.path_data[self.current_step][1], self.path_data[self.current_step][2])
            self.renderer.draw(self.world, step_info)
            
        pygame.quit()

# --- Public Entry Point ---
def visualize_path(path: Path, boxes: Boxes):
    """The main entry point function that the logic script will call."""
    if not path:
        print("Path cannot be empty."); return
    game = Game(path, boxes)
    game.run()

if __name__ == "__main__":
    print("Testing visualizer directly...")
    test_boxes = {(2, 2): 0, (3, 3): 1, (4, 1): 2, (1, 4): 3}
    test_path = [
        (((1, 1), (False, False, False, False)), "Start", 0),
        (((1, 2), (False, False, False, False)), "up", -1),
        (((2, 2), (True, False, False, False)), "right", 9),
    ]
    visualize_path(test_path, test_boxes)
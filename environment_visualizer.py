# Visualization for MDP pathfinding with agent movement, obstacles, and boxes
import pygame
from typing import List, Tuple, Dict, Set, Optional

# Define types for state, path, and boxes
State = Tuple[Tuple[int, int], Tuple[bool, ...]]
Path = List[Tuple[State, str, float]]
Boxes = Dict[Tuple[int, int], int]

# window size and info panel height
GRID_WINDOW_SIZE = 800
INFO_PANEL_HEIGHT = 80

# Define colors used in the visualization
COLOR_BACKGROUND = "white"; COLOR_GRID = "grey"; COLOR_PLAYER = "blue"
COLOR_OBSTACLE = "red"; COLOR_BOX = "saddlebrown"; COLOR_BOX_COLLECTED = "forestgreen"
COLOR_INFO_PANEL = "gainsboro"; COLOR_TEXT = "black"
COLOR_TRAIL = "black"
TRAIL_WIDTH = 4

# Convert grid coordinates to pixel coordinates
def grid_to_pixels(grid_x: int, grid_y: int, grid_size: int, tile_size: float) -> Tuple[int, int]:
    pixel_x = (grid_x - 1) * tile_size
    pixel_y = (grid_size - grid_y) * tile_size
    # return pixel coordinates adjusted for the grid system
    return (pixel_x, pixel_y)

# Base class for all sprites in the game
class BaseSprite(pygame.sprite.Sprite):
    def update(self, dt: float):
        # Base update method does nothing, but needs to exist
        pass
# Player class representing the agent in the grid world
class Player(BaseSprite):
    def __init__(self, grid_pos: Tuple[int, int], grid_size: int, tile_size: int, trail_surface: pygame.Surface, *groups):
        super().__init__(*groups)
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.trail_surface = trail_surface
        # Create a surface for the agent sprite
        self.image = pygame.surface.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        radius = self.tile_size / 4
        # Draw a circle to represent the agent
        pygame.draw.circle(self.image, COLOR_PLAYER, (self.tile_size / 2, self.tile_size / 2), radius)
        
        # Set the initial position of the agent based on grid coordinates
        self.rect.topleft = grid_to_pixels(*grid_pos, self.grid_size, self.tile_size)
        self.pixel_pos = pygame.math.Vector2(self.rect.center)
        self.target_pixel_pos = pygame.math.Vector2(self.rect.center)
        self.last_pixel_pos = pygame.math.Vector2(self.pixel_pos)
        
        self.moving = False

        # Initialize speed based on tile size
        self.speed = 2 * 250 * (self.tile_size / 128)

    # Set the target position for the agent to move towards
    def set_target(self, grid_pos: Tuple[int, int]):
        pixel_x, pixel_y = grid_to_pixels(*grid_pos, self.grid_size, self.tile_size)
        self.target_pixel_pos.xy = (pixel_x + self.tile_size / 2, pixel_y + self.tile_size / 2)
        if self.pixel_pos != self.target_pixel_pos:
            self.moving = True
            self.last_pixel_pos.xy = self.pixel_pos.xy

    # Update the agent's position towards the target position
    def update(self, dt: float):
        if not self.moving:
            return

        # Update the last pixel position before moving    
        self.last_pixel_pos.xy = self.pixel_pos.xy
        
        # Calculate the distance to move based on speed and delta time
        move_speed = max(1, self.speed * dt)
        self.pixel_pos.move_towards_ip(self.target_pixel_pos, move_speed)
        self.rect.center = self.pixel_pos
        
        # Draw the trail on the trail surface
        if self.pixel_pos != self.last_pixel_pos:
             pygame.draw.line(self.trail_surface, COLOR_TRAIL, self.last_pixel_pos, self.pixel_pos, TRAIL_WIDTH)

        if self.pixel_pos.distance_to(self.target_pixel_pos) < 1:
            self.pixel_pos.xy = self.target_pixel_pos.xy
            self.rect.center = self.pixel_pos
            self.moving = False

# Obstacle class representing obstacles in the grid world
class Obstacle(BaseSprite):
    def __init__(self, grid_pos: Tuple[int, int], grid_size: int, tile_size: int, *groups):
        super().__init__(*groups)
        self.image = pygame.surface.Surface((tile_size, tile_size), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        size = tile_size / 2; offset = tile_size / 4
        width = max(1, int(tile_size / 20))
        tl, tr = (offset, offset), (offset + size, offset)
        bl, br = (offset, offset + size), (offset + size, offset + size)
        pygame.draw.line(self.image, COLOR_OBSTACLE, tl, br, width)
        pygame.draw.line(self.image, COLOR_OBSTACLE, tr, bl, width)
        self.rect.topleft = grid_to_pixels(*grid_pos, grid_size, tile_size)

# Box class representing boxes in the grid world
class Box(BaseSprite):
    def __init__(self, grid_pos: Tuple[int, int], box_id: int, grid_size: int, tile_size: int, *groups):
        super().__init__(*groups)
        self.id = box_id
        self.image = pygame.surface.Surface((tile_size, tile_size), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        size = tile_size / 2; offset = tile_size / 4
        self.box_rect = pygame.Rect(offset, offset, size, size)
        self.rect.topleft = grid_to_pixels(*grid_pos, grid_size, tile_size)
        self.update_status(False)

    # Update the box's status (collected or not) and redraw it
    def update_status(self, is_collected: bool):
        self.image.fill((0, 0, 0, 0))
        width = max(1, int(self.image.get_width() / 20))
        if is_collected:
            pygame.draw.rect(self.image, COLOR_BOX_COLLECTED, self.box_rect)
        else:
            pygame.draw.rect(self.image, COLOR_BOX, self.box_rect, width)

# This class represents the world containing all sprites, obstacles, and boxes
class World:
    def __init__(self, path_data: Path, boxes_data: Boxes, obstacles: Set[Tuple[int, int]], grid_size: int, tile_size: int):
        self.all_sprites = pygame.sprite.Group()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.trail_surface = pygame.Surface((GRID_WINDOW_SIZE, GRID_WINDOW_SIZE), pygame.SRCALPHA)
        
        for pos in obstacles:
            Obstacle(pos, grid_size, tile_size, self.all_sprites)
        self.box_objects = []
        for pos, box_id in sorted(boxes_data.items(), key=lambda item: item[1]):
            self.box_objects.append(Box(pos, box_id, grid_size, tile_size, self.all_sprites))
        
        initial_state, _, _ = path_data[0]
        self.player = Player(initial_state[0], grid_size, tile_size, self.trail_surface, self.all_sprites)

    # Update the state of the world based on the current path step
    def update_state(self, current_path_step: Tuple[State, str, float]):
        state, _, _ = current_path_step
        player_pos, box_statuses = state
        self.player.set_target(player_pos)
        for i, box in enumerate(self.box_objects):
            box.update_status(box_statuses[i])

    # Update the kinematics of the agent
    def update_kinematics(self, dt: float) -> bool:
        # Update the agent position based on the time delta
        self.all_sprites.update(dt)
        # Check if the agent has arrived at the target position
        return not self.player.moving
    
    # Draw the world, including the trail and all sprites
    def draw(self, surface: pygame.Surface):
        surface.blit(self.trail_surface, (0, 0))
        self.all_sprites.draw(surface)

# This class handles rendering the world and the information panel
class Renderer:
    def __init__(self, window: pygame.Surface, grid_size: int, tile_size: int, window_size: Tuple[int, int]):
        self.window, self.grid_size, self.tile_size, self.window_size = window, grid_size, tile_size, window_size
        self.info_font = pygame.font.Font(None, 32)

    # Draw the entire window, including the grid, world, and info panel
    def draw(self, world: World, step_info: Tuple[int, str, float]):
        self.window.fill(COLOR_BACKGROUND)
        self.draw_grid()
        world.draw(self.window)
        self.draw_info_panel(step_info)
        pygame.display.update()

    # Draw the grid lines on the window
    def draw_grid(self):
        grid_line_width = max(1, int(self.tile_size / 25))
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.window, COLOR_GRID, (0, i * self.tile_size), (GRID_WINDOW_SIZE, i * self.tile_size), grid_line_width)
            pygame.draw.line(self.window, COLOR_GRID, (i * self.tile_size, 0), (i * self.tile_size, GRID_WINDOW_SIZE), grid_line_width)

    # Draw the information panel at the bottom of the window
    def draw_info_panel(self, step_info: Tuple[int, str, float]):
        step, action, reward = step_info
        panel_rect = pygame.Rect(0, GRID_WINDOW_SIZE, self.window_size[0], INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.window, COLOR_INFO_PANEL, panel_rect)
        texts = [f"Step: {step}", f"Action: {action}", f"Total Reward: {f'{reward:.2f}'}"]
        for i, text in enumerate(texts):
            surf = self.info_font.render(text, True, COLOR_TEXT)
            self.window.blit(surf, (20, GRID_WINDOW_SIZE + 5 + i * 25))

# This class represents the visualization, initializing Pygame and managing the main loop
class Game:
    def __init__(self, path_data: Path, boxes_data: Boxes, obstacles: Set[Tuple[int, int]], grid_size: int):
        pygame.init()
        self.grid_size = grid_size
        self.tile_size = GRID_WINDOW_SIZE / self.grid_size
        self.window_size = (GRID_WINDOW_SIZE, GRID_WINDOW_SIZE + INFO_PANEL_HEIGHT)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Trained agent path visualization")
        self.clock = pygame.time.Clock()
        self.path_data, self.current_step = path_data, 0
        self.world = World(path_data, boxes_data, obstacles, self.grid_size, self.tile_size)
        self.renderer = Renderer(self.window, self.grid_size, self.tile_size, self.window_size)

    # Advance to the next step in the path data and update the world state
    def _advance_to_next_step(self):
        if self.current_step < len(self.path_data) - 1:
            self.current_step += 1
            self.world.update_state(self.path_data[self.current_step])

    # Main loop for running the visualization
    def run(self):
        running = True
        self.world.update_state(self.path_data[0])
        while running:
            dt = self.clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            
            # Handle key events to advance the path
            player_has_arrived = self.world.update_kinematics(dt)

            if player_has_arrived:
                self._advance_to_next_step()
            
            step_info = (self.current_step, self.path_data[self.current_step][1], self.path_data[self.current_step][2])
            self.renderer.draw(self.world, step_info)
        pygame.quit()

# Function to visualize the path in the grid world
def visualize_path(path: Path, boxes: Boxes, obstacles: Set[Tuple[int, int]], grid_size: int):
    if not path:
        print("Path cannot be empty."); return
    game = Game(path, boxes, obstacles, grid_size)
    game.run()
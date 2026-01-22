from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _src = Path(__file__).resolve().parents[3]  # src/
    sys.path.insert(0, str(_src))

import pygame

from simverse.envs.farmtila.env import FarmtilaEnv
from simverse.envs.farmtila.config import FarmtilaConfig

KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_z: 4,
    pygame.K_x: 5,
}

class FarmtilaRender:
    def __init__(self, 
        width: int,
        height: int,
        cell_size: int = 24, 
        fps: int = 30
    ):
        self.width = width
        self.height = height
        pygame.init()
        self.cell_size = cell_size
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.screen = self._init_display()
        self.grid_surface = self._build_grid_surface()
        pygame.display.set_caption("Farmtila")

    def _init_display(self) -> pygame.Surface:
        size = (self.width * self.cell_size, self.height * self.cell_size)
        try:
            return pygame.display.set_mode(size)
        except pygame.error:
            if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                raise
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.quit()
            pygame.display.init()
            return pygame.display.set_mode(size)

    def _build_grid_surface(self) -> pygame.Surface:
        """Pre-render the grid so it can be blitted each frame."""
        w = self.width * self.cell_size
        h = self.height * self.cell_size
        surface = pygame.Surface((w, h))
        surface.fill((255, 255, 255))
        
        grid_color = (220, 220, 220)
        # vertical lines
        for x in range(0, w + 1, self.cell_size):
            pygame.draw.line(surface, grid_color, (x, 0), (x, h - 1))
        # horizontal lines
        for y in range(0, h + 1, self.cell_size):
            pygame.draw.line(surface, grid_color, (0, y), (w - 1, y))
        
        return surface.convert()

    def draw(self, env: FarmtilaEnv):
        self.screen.blit(self.grid_surface, (0, 0))
        
        # draw each agent
        half = self.cell_size // 2
        for agent in env.agents:
            cx = agent.position[0] * self.cell_size + half
            cy = agent.position[1] * self.cell_size + half
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), max(4, half))
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def handle_events(self) -> int | None:
        """Handle pygame events. Returns action if key pressed, None otherwise. Raises SystemExit on quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise SystemExit
                return KEY_TO_ACTION.get(event.key)
        return None
    
    def close(self):
        pygame.quit()

    




if __name__ == "__main__":
    render = FarmtilaRender(width=50, height=50)
    env = FarmtilaEnv(FarmtilaConfig(width=30, height=20))
    env.reset()

    max_frames = int(os.environ.get("FARMTILA_MAX_FRAMES", "0"))
    frames = 0

    try:
        while True:
            render.handle_events()
            render.draw(env)
            frames += 1
            if max_frames and frames >= max_frames:
                break
    except SystemExit:
        pass
    finally:
        render.close()

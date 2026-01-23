from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

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
        self.agent_surface = self._build_agent_surface()
        self.font = pygame.font.SysFont("Arial", max(12, self.cell_size // 2))
        self.agent_label_font = pygame.font.SysFont("Arial", max(10, self.cell_size // 3))
        self.agent_label_cache: dict[int, pygame.Surface] = {}
        button_width = int(self.cell_size * 5)
        button_height = int(self.cell_size)
        self.button_rect = pygame.Rect(10, 10, button_width, button_height)
        self.button_text_surface = self._render_button_text("Run Random Simulation")
        self.running_random = False
        self.env: FarmtilaEnv | None = None
        self.controlled_agent_id = 0
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
        """Pre-render the grid using numpy so it can be blitted each frame."""
        w = self.width * self.cell_size
        h = self.height * self.cell_size
        grid_color = np.array((220, 220, 220), dtype=np.uint8)
        surface_array = np.full((h, w, 3), 255, dtype=np.uint8)
        surface_array[::self.cell_size, :, :] = grid_color
        surface_array[:, ::self.cell_size, :] = grid_color
        surface_array[-1, :, :] = grid_color
        surface_array[:, -1, :] = grid_color
        surface = pygame.surfarray.make_surface(np.swapaxes(surface_array, 0, 1))
        return surface.convert()

    def _build_agent_surface(self) -> pygame.Surface:
        """Create a single-color agent sprite inspired by the provided SVG."""
        base = 400
        color = (74, 144, 226)
        surface = pygame.Surface((base, base), pygame.SRCALPHA)

        def ellipse(cx, cy, rx, ry):
            rect = pygame.Rect(cx - rx, cy - ry, rx * 2, ry * 2)
            pygame.draw.ellipse(surface, color, rect)

        def circle(cx, cy, radius):
            ellipse(cx, cy, radius, radius)

        # Major body parts kept but flattened to a single color
        ellipse(200, 240, 70, 80)   # torso
        ellipse(175, 310, 25, 30)   # left leg
        ellipse(225, 310, 25, 30)   # right leg
        ellipse(145, 230, 22, 45)   # left arm
        ellipse(255, 230, 22, 45)   # right arm
        circle(200, 160, 60)        # head
        circle(160, 130, 25)        # left ear
        circle(240, 130, 25)        # right ear

        scaled_size = self.cell_size
        return pygame.transform.smoothscale(surface, (scaled_size, scaled_size))

    def draw(self, env: FarmtilaEnv):
        self.env = env
        if self.running_random:
            actions = self._seed_harvest_actions(env)
            env.step(actions)
        self.screen.blit(self.grid_surface, (0, 0))

        self._draw_farms(env)
        self._draw_seeds(env)

        # draw each agent using the simplified sprite with index overlay
        for idx, agent in enumerate(env.agents):
            x = agent.position[0] * self.cell_size
            y = agent.position[1] * self.cell_size
            self.screen.blit(self.agent_surface, (x, y))
            self._draw_agent_label(idx, x, y)

        self._draw_button()
        
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
                if self.running_random:
                    self.run_random_simulation()  # toggles off
                return KEY_TO_ACTION.get(event.key)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.button_rect.collidepoint(event.pos):
                    self.run_random_simulation()
        return None
    
    def close(self):
        pygame.quit()

    def run_random_simulation(self):
        self.running_random = not self.running_random
        label = "Stop Random Simulation" if self.running_random else "Run Random Simulation"
        self.button_text_surface = self._render_button_text(label)

    def _draw_button(self):
        pygame.draw.rect(self.screen, (30, 144, 255), self.button_rect, border_radius=6)
        text_rect = self.button_text_surface.get_rect(center=self.button_rect.center)
        self.screen.blit(self.button_text_surface, text_rect)

    def _render_button_text(self, label: str) -> pygame.Surface:
        return self.font.render(label, True, (255, 255, 255))

    def _draw_agent_label(self, idx: int, x: int, y: int):
        label_surface = self._get_agent_label_surface(idx)
        rect = label_surface.get_rect(center=(x + self.cell_size // 2, y + label_surface.get_height() // 2 + 2))
        self.screen.blit(label_surface, rect)

    def _draw_seeds(self, env: FarmtilaEnv):
        seeds = np.argwhere(env.seed_grid > 0)
        if seeds.size == 0:
            return
        radius = max(3, self.cell_size // 6)
        color = (46, 204, 113)
        offset = self.cell_size // 2
        for x, y in seeds:
            cx = int(x) * self.cell_size + offset
            cy = int(y) * self.cell_size + offset
            pygame.draw.circle(self.screen, color, (cx, cy), radius)

    def _draw_farms(self, env: FarmtilaEnv):
        farm_grid = getattr(env, "farm_grid", None)
        if farm_grid is None:
            return
        farms = np.argwhere(farm_grid > 0)
        if farms.size == 0:
            return
        padding = max(2, self.cell_size // 6)
        for x, y in farms:
            owner = int(env.owner_grid[x, y]) if env.owner_grid.size else -1
            color = self._farm_color(owner)
            rect = pygame.Rect(
                x * self.cell_size + padding,
                y * self.cell_size + padding,
                self.cell_size - padding * 2,
                self.cell_size - padding * 2,
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _get_agent_label_surface(self, idx: int) -> pygame.Surface:
        if idx not in self.agent_label_cache:
            surface = self.agent_label_font.render(str(idx), True, (0, 0, 0))
            surface = surface.convert_alpha()
            surface.set_alpha(150)
            self.agent_label_cache[idx] = surface
        return self.agent_label_cache[idx]

    def _seed_harvest_actions(self, env: FarmtilaEnv) -> dict[int, int]:
        actions: dict[int, int] = {}
        for agent in env.agents:
            if agent.inventory > 0 and env.farm_grid[agent.position[0], agent.position[1]] == 0:
                actions[agent.agent_id] = env.HARVEST_ACTION
                continue
            rng = getattr(env, "rng", None)
            if rng is None:
                action = int(np.random.randint(0, 4))
            else:
                action = int(rng.integers(0, 4))
            actions[agent.agent_id] = action
        return actions

    def _nearest_seed(self, position: tuple[int, int], seeds: np.ndarray) -> tuple[int, int] | None:
        if seeds.size == 0:
            return None
        px, py = position
        best_seed: tuple[int, int] | None = None
        best_dist = None
        for sx, sy in seeds:
            dist = abs(int(sx) - px) + abs(int(sy) - py)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_seed = (int(sx), int(sy))
        return best_seed

    def _farm_color(self, owner: int) -> tuple[int, int, int]:
        palette = [
            (205, 133, 63),  # sienna
            (160, 82, 45),   # saddle brown
            (210, 180, 140), # tan
            (222, 184, 135), # burlywood
        ]
        if owner < 0:
            return palette[0]
        return palette[owner % len(palette)]

    




if __name__ == "__main__":
    render = FarmtilaRender(width=50, height=50)
    env = FarmtilaEnv(FarmtilaConfig(width=30, height=20, num_agents=2))
    env.reset()

    render.run_random_simulation()

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

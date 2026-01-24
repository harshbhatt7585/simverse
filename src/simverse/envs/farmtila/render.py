from __future__ import annotations

import math
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

# =============================================================================
# ENHANCED COLOR PALETTE - Warm, natural farming aesthetic
# =============================================================================
COLORS = {
    # Grass tones (layered for depth)
    "grass_base": (86, 125, 70),
    "grass_light": (124, 172, 104),
    "grass_dark": (62, 95, 52),
    "grass_accent": (98, 145, 80),
    
    # Soil/farm tones
    "soil_rich": (101, 67, 33),
    "soil_light": (139, 90, 43),
    "soil_dark": (72, 47, 23),
    "soil_tilled": (121, 85, 51),
    
    # Crop/plant colors
    "crop_green": (76, 153, 0),
    "crop_yellow": (218, 165, 32),
    "crop_mature": (139, 195, 74),
    "seed_glow": (144, 238, 144),
    
    # Agent colors
    "skin_tone": (255, 213, 170),
    "skin_shadow": (220, 180, 140),
    "overalls_blue": (70, 130, 180),
    "overalls_dark": (50, 100, 150),
    "hat_straw": (245, 222, 179),
    "hat_band": (139, 69, 19),
    
    # UI colors
    "ui_bg": (40, 44, 52),
    "ui_accent": (97, 175, 239),
    "ui_text": (248, 248, 242),
    "ui_shadow": (30, 33, 40),
    
    # Effects
    "shadow": (0, 0, 0, 60),
    "highlight": (255, 255, 255, 40),
    "particle_gold": (255, 215, 0),
}


class FarmtilaRender:
    def __init__(self, 
        width: int,
        height: int,
        cell_size: int = 32, 
        fps: int = 30
    ):
        self.width = width
        self.height = height
        pygame.init()
        self.cell_size = cell_size
        self.fps = fps
        self.clock = pygame.time.Clock()
        
        # Right panel dimensions
        self.panel_width = int(cell_size * 7)
        self.panel_padding = 12
        
        self.screen = self._init_display()
        
        # Animation state
        self.frame_count = 0
        self.particles: list[dict] = []
        
        # Pre-render surfaces
        self.grass_surface = self._build_grass_surface()
        self.agent_surfaces = self._build_agent_surfaces()
        self.agent_colors = [
            (70, 130, 180),   # Steel blue
            (180, 90, 70),    # Terracotta
            (130, 100, 160),  # Lavender
            (100, 160, 130),  # Sage
            (180, 140, 80),   # Mustard
            (160, 80, 120),   # Mauve
        ]
        self.seed_surface = self._build_seed_surface()
        self.farm_surfaces = self._build_farm_surfaces()
        
        # Fonts
        pygame.font.init()
        self.font = pygame.font.SysFont("Verdana", max(12, self.cell_size // 2), bold=True)
        self.agent_label_font = pygame.font.SysFont("Verdana", max(10, self.cell_size // 3), bold=True)
        self.small_font = pygame.font.SysFont("Verdana", max(9, self.cell_size // 4))
        self.panel_title_font = pygame.font.SysFont("Verdana", max(13, self.cell_size // 2), bold=True)
        self.panel_font = pygame.font.SysFont("Verdana", max(11, int(self.cell_size * 0.4)))
        self.panel_small_font = pygame.font.SysFont("Verdana", max(9, int(self.cell_size * 0.3)))
        self.agent_label_cache: dict[int, pygame.Surface] = {}
        
        # UI elements
        button_width = int(self.cell_size * 6)
        button_height = int(self.cell_size * 1.2)
        self.button_rect = pygame.Rect(12, 12, button_width, button_height)
        self.button_text_surface = self._render_button_text("Run Random Simulation")
        self.running_random = False
        self.env: FarmtilaEnv | None = None
        self.controlled_agent_id = 0
        pygame.display.set_caption("🌾 Farmtila")

    def _init_display(self) -> pygame.Surface:
        # Add panel width to the right side
        game_width = self.width * self.cell_size
        total_width = game_width + self.panel_width
        size = (total_width, self.height * self.cell_size)
        try:
            return pygame.display.set_mode(size)
        except pygame.error:
            if os.environ.get("SDL_VIDEODRIVER") == "dummy":
                raise
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.quit()
            pygame.display.init()
            return pygame.display.set_mode(size)

    def _build_grass_surface(self) -> pygame.Surface:
        """Create a rich, textured grass background with natural variation."""
        w = self.width * self.cell_size
        h = self.height * self.cell_size
        surface = pygame.Surface((w, h))
        
        # Base gradient from darker at top to lighter at bottom (sun effect)
        for y in range(h):
            blend = y / h
            r = int(COLORS["grass_dark"][0] * (1 - blend * 0.3) + COLORS["grass_light"][0] * blend * 0.3)
            g = int(COLORS["grass_dark"][1] * (1 - blend * 0.3) + COLORS["grass_light"][1] * blend * 0.3)
            b = int(COLORS["grass_dark"][2] * (1 - blend * 0.3) + COLORS["grass_light"][2] * blend * 0.3)
            pygame.draw.line(surface, (r, g, b), (0, y), (w, y))
        
        # Add tile-based variation for depth
        rng = np.random.default_rng(42)  # Fixed seed for consistent look
        for gx in range(self.width):
            for gy in range(self.height):
                x = gx * self.cell_size
                y = gy * self.cell_size
                
                # Subtle per-cell color variation
                variation = rng.integers(-8, 8)
                base_color = COLORS["grass_base"]
                cell_color = (
                    max(0, min(255, base_color[0] + variation)),
                    max(0, min(255, base_color[1] + variation + rng.integers(-3, 5))),
                    max(0, min(255, base_color[2] + variation)),
                )
                
                # Draw cell with subtle border
                cell_rect = pygame.Rect(x + 1, y + 1, self.cell_size - 2, self.cell_size - 2)
                pygame.draw.rect(surface, cell_color, cell_rect)
                
                # Add grass texture detail
                num_blades = rng.integers(2, 5)
                for _ in range(num_blades):
                    bx = x + rng.integers(4, self.cell_size - 4)
                    by = y + rng.integers(int(self.cell_size * 0.5), self.cell_size - 2)
                    blade_height = rng.integers(3, 7)
                    blade_color = COLORS["grass_light"] if rng.random() > 0.5 else COLORS["grass_accent"]
                    pygame.draw.line(surface, blade_color, (bx, by), (bx + rng.integers(-1, 2), by - blade_height), 1)
        
        # Soft grid overlay
        grid_color = (*COLORS["grass_dark"][:3],)
        for gx in range(self.width + 1):
            x = gx * self.cell_size
            pygame.draw.line(surface, grid_color, (x, 0), (x, h), 1)
        for gy in range(self.height + 1):
            y = gy * self.cell_size
            pygame.draw.line(surface, grid_color, (0, y), (w, y), 1)
        
        return surface.convert()

    def _build_agent_surfaces(self) -> list[pygame.Surface]:
        """Create charming farmer character sprites with different colored overalls."""
        colors = [
            (70, 130, 180),   # Steel blue
            (180, 90, 70),    # Terracotta
            (130, 100, 160),  # Lavender
            (100, 160, 130),  # Sage
            (180, 140, 80),   # Mustard
            (160, 80, 120),   # Mauve
        ]
        
        surfaces = []
        for color in colors:
            surface = self._create_farmer_sprite(color)
            surfaces.append(surface)
        return surfaces

    def _create_farmer_sprite(self, overalls_color: tuple[int, int, int]) -> pygame.Surface:
        """Create a single farmer sprite with given overalls color."""
        size = self.cell_size
        padding = max(2, size // 10)
        sprite_size = size - padding * 2
        
        # Work at higher resolution for quality
        scale = 4
        canvas_size = sprite_size * scale
        surface = pygame.Surface((canvas_size, canvas_size), pygame.SRCALPHA)
        
        cx, cy = canvas_size // 2, canvas_size // 2
        
        # Shadow under character
        shadow_y = int(cy + canvas_size * 0.38)
        shadow_rect = pygame.Rect(cx - canvas_size // 4, shadow_y, canvas_size // 2, canvas_size // 8)
        shadow_surface = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0, 50), shadow_surface.get_rect())
        surface.blit(shadow_surface, shadow_rect.topleft)
        
        # Body/overalls - rounded rectangle shape
        body_width = int(canvas_size * 0.45)
        body_height = int(canvas_size * 0.35)
        body_x = cx - body_width // 2
        body_y = int(cy - canvas_size * 0.05)
        
        # Overalls darker shade for depth
        darker_overalls = tuple(max(0, c - 30) for c in overalls_color)
        body_rect = pygame.Rect(body_x, body_y, body_width, body_height)
        pygame.draw.rect(surface, darker_overalls, body_rect, border_radius=canvas_size // 10)
        
        # Overalls main color
        inner_rect = pygame.Rect(body_x + 2, body_y + 2, body_width - 4, body_height - 6)
        pygame.draw.rect(surface, overalls_color, inner_rect, border_radius=canvas_size // 12)
        
        # Legs
        leg_width = int(canvas_size * 0.14)
        leg_height = int(canvas_size * 0.18)
        leg_y = body_y + body_height - 4
        
        left_leg_x = cx - leg_width - 2
        right_leg_x = cx + 2
        
        pygame.draw.rect(surface, darker_overalls, (left_leg_x, leg_y, leg_width, leg_height), border_radius=4)
        pygame.draw.rect(surface, darker_overalls, (right_leg_x, leg_y, leg_width, leg_height), border_radius=4)
        
        # Boots
        boot_color = (60, 45, 30)
        boot_height = leg_height // 3
        pygame.draw.rect(surface, boot_color, (left_leg_x - 1, leg_y + leg_height - boot_height, leg_width + 2, boot_height), border_radius=2)
        pygame.draw.rect(surface, boot_color, (right_leg_x - 1, leg_y + leg_height - boot_height, leg_width + 2, boot_height), border_radius=2)
        
        # Arms
        arm_width = int(canvas_size * 0.1)
        arm_height = int(canvas_size * 0.2)
        arm_y = body_y + 8
        
        # Shirt sleeves
        sleeve_color = COLORS["skin_shadow"]
        pygame.draw.ellipse(surface, sleeve_color, (body_x - arm_width + 4, arm_y, arm_width, arm_height))
        pygame.draw.ellipse(surface, sleeve_color, (body_x + body_width - 4, arm_y, arm_width, arm_height))
        
        # Hands
        hand_size = int(canvas_size * 0.08)
        pygame.draw.circle(surface, COLORS["skin_tone"], (body_x - arm_width // 2 + 4, arm_y + arm_height), hand_size)
        pygame.draw.circle(surface, COLORS["skin_tone"], (body_x + body_width + arm_width // 2 - 4, arm_y + arm_height), hand_size)
        
        # Head
        head_radius = int(canvas_size * 0.18)
        head_y = body_y - head_radius + 4
        pygame.draw.circle(surface, COLORS["skin_shadow"], (cx, head_y), head_radius)
        pygame.draw.circle(surface, COLORS["skin_tone"], (cx, head_y - 1), head_radius - 2)
        
        # Straw hat
        hat_color = COLORS["hat_straw"]
        hat_band_color = COLORS["hat_band"]
        
        # Hat brim
        brim_width = int(canvas_size * 0.5)
        brim_height = int(canvas_size * 0.08)
        brim_y = head_y - head_radius - brim_height // 2 + 4
        brim_rect = pygame.Rect(cx - brim_width // 2, brim_y, brim_width, brim_height)
        pygame.draw.ellipse(surface, hat_band_color, brim_rect)
        pygame.draw.ellipse(surface, hat_color, (brim_rect.x + 1, brim_rect.y + 1, brim_rect.width - 2, brim_rect.height - 3))
        
        # Hat crown
        crown_width = int(canvas_size * 0.28)
        crown_height = int(canvas_size * 0.15)
        crown_x = cx - crown_width // 2
        crown_y = brim_y - crown_height + 4
        pygame.draw.rect(surface, hat_color, (crown_x, crown_y, crown_width, crown_height), border_radius=6)
        
        # Hat band
        band_height = 4
        pygame.draw.rect(surface, hat_band_color, (crown_x, crown_y + crown_height - band_height - 2, crown_width, band_height))
        
        # Face details - simple and charming
        eye_y = head_y - 2
        eye_spacing = head_radius // 2
        eye_size = max(3, canvas_size // 20)
        
        # Eyes (simple dots)
        pygame.draw.circle(surface, (50, 40, 30), (cx - eye_spacing, eye_y), eye_size)
        pygame.draw.circle(surface, (50, 40, 30), (cx + eye_spacing, eye_y), eye_size)
        
        # Eye highlights
        highlight_size = max(1, eye_size // 2)
        pygame.draw.circle(surface, (255, 255, 255), (cx - eye_spacing + 1, eye_y - 1), highlight_size)
        pygame.draw.circle(surface, (255, 255, 255), (cx + eye_spacing + 1, eye_y - 1), highlight_size)
        
        # Rosy cheeks
        cheek_size = max(2, canvas_size // 25)
        cheek_color = (255, 180, 170, 100)
        cheek_surface = pygame.Surface((cheek_size * 2, cheek_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(cheek_surface, cheek_color, (cheek_size, cheek_size), cheek_size)
        surface.blit(cheek_surface, (cx - eye_spacing - cheek_size * 2, eye_y + eye_size))
        surface.blit(cheek_surface, (cx + eye_spacing, eye_y + eye_size))
        
        # Smile
        smile_rect = pygame.Rect(cx - head_radius // 3, eye_y + head_radius // 3, head_radius * 2 // 3, head_radius // 4)
        pygame.draw.arc(surface, (80, 60, 50), smile_rect, 3.14, 2 * 3.14, 2)
        
        # Scale down to final size
        final_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        scaled = pygame.transform.smoothscale(surface, (sprite_size, sprite_size))
        final_surface.blit(scaled, (padding, padding))
        
        return final_surface

    def _build_seed_surface(self) -> pygame.Surface:
        """Create a glowing seed/sprout sprite."""
        size = self.cell_size
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        cx, cy = size // 2, size // 2
        
        # Glow effect
        for r in range(size // 3, 2, -2):
            alpha = int(40 * (1 - r / (size // 3)))
            glow_color = (*COLORS["seed_glow"][:3], alpha)
            glow_surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, glow_color, (r, r), r)
            surface.blit(glow_surface, (cx - r, cy - r))
        
        # Seed body
        seed_radius = max(3, size // 6)
        pygame.draw.circle(surface, (139, 90, 43), (cx, cy + 2), seed_radius)
        pygame.draw.circle(surface, (180, 120, 60), (cx, cy + 1), seed_radius - 1)
        
        # Sprout
        sprout_height = size // 4
        stem_color = (76, 153, 0)
        leaf_color = (100, 180, 50)
        
        # Stem
        pygame.draw.line(surface, stem_color, (cx, cy), (cx, cy - sprout_height), 2)
        
        # Leaves
        leaf_size = max(3, size // 8)
        # Left leaf
        pygame.draw.ellipse(surface, leaf_color, (cx - leaf_size - 1, cy - sprout_height + 2, leaf_size, leaf_size // 2 + 2))
        # Right leaf
        pygame.draw.ellipse(surface, leaf_color, (cx + 1, cy - sprout_height + 2, leaf_size, leaf_size // 2 + 2))
        
        return surface

    def _build_farm_surfaces(self) -> list[pygame.Surface]:
        """Create farm/tilled soil surfaces with crops for different owners."""
        crop_colors = [
            (76, 153, 0),     # Green
            (218, 165, 32),   # Golden wheat
            (255, 140, 0),    # Orange/pumpkin
            (147, 112, 219),  # Purple/eggplant
            (220, 20, 60),    # Red/tomato
            (34, 139, 34),    # Forest green
        ]
        
        surfaces = []
        for color in crop_colors:
            surface = self._create_farm_tile(color)
            surfaces.append(surface)
        return surfaces

    def _create_farm_tile(self, crop_color: tuple[int, int, int]) -> pygame.Surface:
        """Create a single farm tile with tilled soil and crops."""
        size = self.cell_size
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        padding = max(2, size // 8)
        inner_size = size - padding * 2
        
        # Soil base with texture
        soil_rect = pygame.Rect(padding, padding, inner_size, inner_size)
        
        # Darker border for depth
        pygame.draw.rect(surface, COLORS["soil_dark"], soil_rect, border_radius=3)
        
        # Inner soil
        inner_soil = pygame.Rect(padding + 1, padding + 1, inner_size - 2, inner_size - 2)
        pygame.draw.rect(surface, COLORS["soil_rich"], inner_soil, border_radius=2)
        
        # Tilled rows
        row_spacing = max(3, inner_size // 5)
        for i in range(1, 5):
            y = padding + i * row_spacing
            if y < padding + inner_size - 2:
                pygame.draw.line(surface, COLORS["soil_dark"], (padding + 3, y), (padding + inner_size - 3, y), 1)
        
        # Crop plants
        plant_height = inner_size // 2
        cx = size // 2
        
        # Main stem
        stem_bottom = size - padding - 3
        stem_top = stem_bottom - plant_height
        pygame.draw.line(surface, (60, 120, 40), (cx, stem_bottom), (cx, stem_top), 2)
        
        # Leaves/crop top
        leaf_size = max(4, inner_size // 4)
        
        # Draw crop based on color (different shapes for variety)
        if crop_color[1] > 150 and crop_color[0] > 150:  # Golden/wheat-like
            # Wheat head
            for i in range(3):
                offset_y = stem_top + i * 3
                pygame.draw.ellipse(surface, crop_color, (cx - 2, offset_y - 2, 5, 4))
        else:
            # Round fruit/vegetable
            pygame.draw.circle(surface, crop_color, (cx, stem_top + leaf_size // 3), leaf_size // 2 + 1)
            # Highlight
            pygame.draw.circle(surface, (min(255, crop_color[0] + 50), min(255, crop_color[1] + 50), min(255, crop_color[2] + 50)), 
                             (cx - 1, stem_top + leaf_size // 3 - 1), leaf_size // 4)
        
        # Small leaves on stem
        pygame.draw.ellipse(surface, (80, 140, 50), (cx - leaf_size, stem_top + plant_height // 2, leaf_size, leaf_size // 2))
        pygame.draw.ellipse(surface, (80, 140, 50), (cx, stem_top + plant_height // 2 + 2, leaf_size, leaf_size // 2))
        
        return surface

    def draw(self, env: FarmtilaEnv):
        self.env = env
        self.frame_count += 1
        
        if self.running_random:
            actions = self._seed_harvest_actions(env)
            env.step(actions)
        
        # Draw grass background
        self.screen.blit(self.grass_surface, (0, 0))
        
        # Draw farms (under seeds and agents)
        self._draw_farms(env)
        
        # Draw seeds
        self._draw_seeds(env)
        
        # Draw particles
        self._update_and_draw_particles()
        
        # Draw agents
        for idx, agent in enumerate(env.agents):
            x = agent.position[0] * self.cell_size
            y = agent.position[1] * self.cell_size
            
            # Get appropriate surface
            surface = self.agent_surfaces[idx % len(self.agent_surfaces)]
            
            # Subtle bobbing animation
            bob_offset = int(math.sin(self.frame_count * 0.15 + idx) * 2)
            
            self.screen.blit(surface, (x, y + bob_offset))
            self._draw_agent_label(idx, x, y + bob_offset)
        
        # Draw UI
        self._draw_button()
        self._draw_stats(env)
        
        # Draw agent stats panel on the right
        self._draw_agent_stats_panel(env)
        
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
        label = "⏹ Stop Simulation" if self.running_random else "▶ Run Simulation"
        self.button_text_surface = self._render_button_text(label)

    def _draw_button(self):
        # Button shadow
        shadow_rect = self.button_rect.move(2, 2)
        pygame.draw.rect(self.screen, COLORS["ui_shadow"], shadow_rect, border_radius=8)
        
        # Button background with gradient effect
        pygame.draw.rect(self.screen, COLORS["ui_bg"], self.button_rect, border_radius=8)
        
        # Accent border
        pygame.draw.rect(self.screen, COLORS["ui_accent"], self.button_rect, width=2, border_radius=8)
        
        # Button text
        text_rect = self.button_text_surface.get_rect(center=self.button_rect.center)
        self.screen.blit(self.button_text_surface, text_rect)

    def _draw_stats(self, env: FarmtilaEnv):
        """Draw game statistics in the corner."""
        stats_y = self.button_rect.bottom + 8
        
        # Stats background
        stats_width = int(self.cell_size * 4)
        stats_height = int(self.cell_size * 2)
        stats_rect = pygame.Rect(12, stats_y, stats_width, stats_height)
        
        # Semi-transparent background
        stats_surface = pygame.Surface((stats_width, stats_height), pygame.SRCALPHA)
        pygame.draw.rect(stats_surface, (*COLORS["ui_bg"][:3], 200), stats_surface.get_rect(), border_radius=6)
        self.screen.blit(stats_surface, stats_rect.topleft)
        
        # Stats text
        seeds_count = int(np.sum(env.seed_grid > 0))
        farms_count = int(np.sum(env.farm_grid > 0))
        
        seed_text = self.small_font.render(f"🌱 Seeds: {seeds_count}", True, COLORS["ui_text"])
        farm_text = self.small_font.render(f"🌾 Farms: {farms_count}", True, COLORS["ui_text"])
        step_text = self.small_font.render(f"⏱ Step: {env.steps}", True, COLORS["ui_text"])
        
        line_height = max(14, self.cell_size // 2)
        self.screen.blit(seed_text, (stats_rect.x + 8, stats_rect.y + 6))
        self.screen.blit(farm_text, (stats_rect.x + 8, stats_rect.y + 6 + line_height))
        self.screen.blit(step_text, (stats_rect.x + 8, stats_rect.y + 6 + line_height * 2))

    def _draw_agent_stats_panel(self, env: FarmtilaEnv):
        """Draw the right-side panel showing stats for all agents."""
        panel_x = self.width * self.cell_size
        panel_height = self.height * self.cell_size
        
        # Panel background - darker theme
        panel_surface = pygame.Surface((self.panel_width, panel_height))
        panel_surface.fill(COLORS["ui_bg"])
        
        # Subtle gradient overlay for depth
        for y in range(panel_height):
            alpha = int(15 * (1 - y / panel_height))
            overlay_color = (255, 255, 255, alpha)
            overlay_line = pygame.Surface((self.panel_width, 1), pygame.SRCALPHA)
            overlay_line.fill(overlay_color)
            panel_surface.blit(overlay_line, (0, y))
        
        self.screen.blit(panel_surface, (panel_x, 0))
        
        # Left border accent line
        pygame.draw.line(self.screen, COLORS["ui_accent"], (panel_x, 0), (panel_x, panel_height), 2)
        
        # Panel title
        title_text = self.panel_title_font.render("AGENTS", True, COLORS["ui_accent"])
        title_x = panel_x + (self.panel_width - title_text.get_width()) // 2
        self.screen.blit(title_text, (title_x, self.panel_padding))
        
        # Decorative line under title
        line_y = self.panel_padding + title_text.get_height() + 6
        pygame.draw.line(self.screen, COLORS["ui_accent"], 
                        (panel_x + self.panel_padding, line_y), 
                        (panel_x + self.panel_width - self.panel_padding, line_y), 1)
        
        # Calculate farm ownership for each agent
        agent_farms = {}
        for agent in env.agents:
            agent_farms[agent.agent_id] = int(np.sum(env.owner_grid == agent.agent_id))
        
        # Draw each agent's stats
        card_start_y = line_y + 12
        card_height = max(60, int(self.cell_size * 2.2))
        card_spacing = 8
        
        for idx, agent in enumerate(env.agents):
            card_y = card_start_y + idx * (card_height + card_spacing)
            
            # Skip if card would be off-screen
            if card_y + card_height > panel_height - self.panel_padding:
                # Draw overflow indicator
                overflow_text = self.panel_small_font.render(f"+ {len(env.agents) - idx} more...", True, COLORS["ui_text"])
                self.screen.blit(overflow_text, (panel_x + self.panel_padding, card_y))
                break
            
            self._draw_agent_card(env, agent, idx, panel_x, card_y, card_height, agent_farms.get(agent.agent_id, 0))

    def _draw_agent_card(self, env: FarmtilaEnv, agent, idx: int, panel_x: int, card_y: int, card_height: int, farm_count: int):
        """Draw a single agent's stat card."""
        card_width = self.panel_width - self.panel_padding * 2
        card_x = panel_x + self.panel_padding
        
        # Card background
        agent_color = self.agent_colors[idx % len(self.agent_colors)]
        card_surface = pygame.Surface((card_width, card_height), pygame.SRCALPHA)
        
        # Subtle gradient background with agent color tint
        base_bg = (50, 54, 62)
        for y in range(card_height):
            blend = y / card_height
            r = int(base_bg[0] + (agent_color[0] - base_bg[0]) * 0.1 * (1 - blend))
            g = int(base_bg[1] + (agent_color[1] - base_bg[1]) * 0.1 * (1 - blend))
            b = int(base_bg[2] + (agent_color[2] - base_bg[2]) * 0.1 * (1 - blend))
            pygame.draw.line(card_surface, (r, g, b, 220), (0, y), (card_width, y))
        
        # Card border with agent color
        pygame.draw.rect(card_surface, agent_color, card_surface.get_rect(), width=2, border_radius=8)
        self.screen.blit(card_surface, (card_x, card_y))
        
        # Agent color indicator bar on left
        indicator_rect = pygame.Rect(card_x + 4, card_y + 8, 4, card_height - 16)
        pygame.draw.rect(self.screen, agent_color, indicator_rect, border_radius=2)
        
        # Content area
        content_x = card_x + 16
        content_y = card_y + 8
        
        # Agent header with ID
        header_text = self.panel_font.render(f"Farmer {idx}", True, COLORS["ui_text"])
        self.screen.blit(header_text, (content_x, content_y))
        
        # Position text (smaller)
        pos_text = self.panel_small_font.render(f"({agent.position[0]}, {agent.position[1]})", True, (150, 150, 160))
        self.screen.blit(pos_text, (content_x + header_text.get_width() + 6, content_y + 2))
        
        # Stats row
        stats_y = content_y + header_text.get_height() + 6
        
        # Inventory stat with icon
        inv_icon_color = COLORS["seed_glow"] if agent.inventory > 0 else (100, 100, 110)
        pygame.draw.circle(self.screen, inv_icon_color, (content_x + 6, stats_y + 8), 5)
        pygame.draw.circle(self.screen, (60, 120, 40), (content_x + 6, stats_y + 4), 2)  # sprout
        
        inv_label = self.panel_small_font.render("Inventory:", True, (150, 150, 160))
        self.screen.blit(inv_label, (content_x + 16, stats_y))
        
        inv_value_color = (100, 255, 150) if agent.inventory > 0 else COLORS["ui_text"]
        inv_value = self.panel_font.render(str(agent.inventory), True, inv_value_color)
        self.screen.blit(inv_value, (content_x + 16 + inv_label.get_width() + 4, stats_y - 2))
        
        # Farms stat with icon
        farms_y = stats_y + 18
        farm_icon_color = COLORS["soil_light"] if farm_count > 0 else (100, 100, 110)
        farm_icon_rect = pygame.Rect(content_x + 2, farms_y + 3, 8, 8)
        pygame.draw.rect(self.screen, farm_icon_color, farm_icon_rect, border_radius=2)
        # Small crop on farm icon
        if farm_count > 0:
            pygame.draw.line(self.screen, (80, 160, 60), (content_x + 6, farms_y + 6), (content_x + 6, farms_y + 1), 1)
        
        farm_label = self.panel_small_font.render("Farms:", True, (150, 150, 160))
        self.screen.blit(farm_label, (content_x + 16, farms_y))
        
        farm_value_color = (255, 200, 100) if farm_count > 0 else COLORS["ui_text"]
        farm_value = self.panel_font.render(str(farm_count), True, farm_value_color)
        self.screen.blit(farm_value, (content_x + 16 + farm_label.get_width() + 4, farms_y - 2))

    def _render_button_text(self, label: str) -> pygame.Surface:
        return self.font.render(label, True, COLORS["ui_text"])

    def _draw_agent_label(self, idx: int, x: int, y: int):
        label_surface = self._get_agent_label_surface(idx)
        # Position label at bottom of agent
        rect = label_surface.get_rect(center=(x + self.cell_size // 2, y + self.cell_size - 4))
        
        # Small background for readability
        bg_rect = rect.inflate(4, 2)
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(bg_surface, (0, 0, 0, 120), bg_surface.get_rect(), border_radius=3)
        self.screen.blit(bg_surface, bg_rect.topleft)
        
        self.screen.blit(label_surface, rect)

    def _draw_seeds(self, env: FarmtilaEnv):
        seeds = np.argwhere(env.seed_grid > 0)
        if seeds.size == 0:
            return
        
        for x, y in seeds:
            px = int(x) * self.cell_size
            py = int(y) * self.cell_size
            
            # Subtle pulsing glow
            pulse = 0.8 + 0.2 * math.sin(self.frame_count * 0.1 + x + y)
            
            if pulse > 0.9:
                # Draw with slight scale variation for pulse effect
                self.screen.blit(self.seed_surface, (px, py))
            else:
                self.screen.blit(self.seed_surface, (px, py))

    def _draw_farms(self, env: FarmtilaEnv):
        farm_grid = getattr(env, "farm_grid", None)
        if farm_grid is None:
            return
        farms = np.argwhere(farm_grid > 0)
        if farms.size == 0:
            return
        
        for x, y in farms:
            owner = int(env.owner_grid[x, y]) if env.owner_grid.size else 0
            surface = self.farm_surfaces[owner % len(self.farm_surfaces)]
            
            px = int(x) * self.cell_size
            py = int(y) * self.cell_size
            
            self.screen.blit(surface, (px, py))

    def _get_agent_label_surface(self, idx: int) -> pygame.Surface:
        if idx not in self.agent_label_cache:
            surface = self.agent_label_font.render(str(idx), True, (255, 255, 255))
            self.agent_label_cache[idx] = surface
        return self.agent_label_cache[idx]

    def _seed_harvest_actions(self, env: FarmtilaEnv) -> dict[int, int]:
        actions: dict[int, int] = {}
        for agent in env.agents:
            if agent.inventory > 0 and env.farm_grid[agent.position[0], agent.position[1]] == 0:
                actions[agent.agent_id] = env.HARVEST_ACTION
                # Add particles for planting
                self._spawn_particles(agent.position[0] * self.cell_size + self.cell_size // 2,
                                     agent.position[1] * self.cell_size + self.cell_size // 2, count=5)
                continue
            rng = getattr(env, "rng", None)
            if rng is None:
                action = int(np.random.randint(0, 4))
            else:
                action = int(rng.integers(0, 4))
            actions[agent.agent_id] = action
        return actions

    def _spawn_particles(self, x: int, y: int, count: int = 3):
        """Spawn decorative particles at a position."""
        rng = np.random.default_rng()
        for _ in range(count):
            self.particles.append({
                "x": x + rng.integers(-5, 6),
                "y": y,
                "vx": rng.uniform(-1, 1),
                "vy": rng.uniform(-2, -0.5),
                "life": 30,
                "color": COLORS["particle_gold"],
                "size": rng.integers(2, 5),
            })

    def _update_and_draw_particles(self):
        """Update and draw all active particles."""
        alive_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1  # gravity
            p["life"] -= 1
            
            if p["life"] > 0:
                alive_particles.append(p)
                alpha = int(255 * (p["life"] / 30))
                color = (*p["color"][:3], alpha)
                particle_surface = pygame.Surface((p["size"] * 2, p["size"] * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surface, color, (p["size"], p["size"]), p["size"])
                self.screen.blit(particle_surface, (int(p["x"]) - p["size"], int(p["y"]) - p["size"]))
        
        self.particles = alive_particles

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


if __name__ == "__main__":
    render = FarmtilaRender(width=30, height=20, cell_size=32)
    env = FarmtilaEnv(FarmtilaConfig(width=30, height=20, num_agents=4))
    env.reset()

    render.run_random_simulation()

    max_frames = int(os.environ.get("FARMTILA_MAX_FRAMES", "0"))
    frames = 0

    try:
        while True:
            render.handle_events()
            render.draw(env)
            frames += 1
            if env.done:
                break
            if max_frames and frames >= max_frames:
                break
    except SystemExit:
        pass
    finally:
        render.close()

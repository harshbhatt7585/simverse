import pygame

KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_z: 4,
    pygame.K_x: 5,
}

class FarmtilaRender:
    def __init__(self, cell_size: int = 24, fps: int = 30):
        self.cell_size = cell_size
        self.fps = fps

    def render(self):
        pass
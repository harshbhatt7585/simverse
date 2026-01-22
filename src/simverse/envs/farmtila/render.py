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
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        pygame.display.set_caption("Farmtila")

    def draw(self, env: FarmtilaEnv):
        self.screen.fill((255, 255, 255))
        
        # draw each agent
        for agent in env.agents:
            pygame.draw.circle(self.screen, (0, 0, 0), (agent.position[0] * self.cell_size, agent.position[1] * self.cell_size), 10)
        
    
    def close(self):
        pygame.quit()

    




if __name__ == "__main__":
    render = FarmtilaRender(width=50, height=50)
    env = FarmtilaEnv(FarmtilaConfig(width=50, height=50))
    render.draw(env)
    render.close()

        
import pygame
from game.utils.dashboard_color import DashboardColor
from game.utils.common import BLOCK_SIZE


class Dashboard:
    """Responsible for drawing the game's UI: snake, food, and score."""

    def __init__(self, font: pygame.font.Font):
        self.font = font

    def draw(self, display: pygame.Surface, snake_body, food, score: int) -> None:
        display.fill(DashboardColor.BLACK.value)
        for segment in snake_body:
            pygame.draw.rect(
                display,
                DashboardColor.BLUE1.value,
                pygame.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                display,
                DashboardColor.BLUE2.value,
                pygame.Rect(segment.x + 4, segment.y + 4, 12, 12),
            )
        pygame.draw.rect(
            display,
            DashboardColor.RED.value,
            pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        score_text = self.font.render(f"Score: {score}", True, DashboardColor.WHITE.value)
        display.blit(score_text, [0, 0])
        pygame.display.flip()

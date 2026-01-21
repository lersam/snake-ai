import pygame
from typing import Sequence, Any
from game.utils.common import BLOCK_SIZE


class Dashboard:
    """Draw the game's UI: snake, food, and score, with a gray title bar."""

    def __init__(self, font: pygame.font.Font):
        self.font = font
        # make title height align to grid to avoid coordinate misalignment
        self.title_height = BLOCK_SIZE
        # color attributes (lowercase as requested)
        self.white = (255, 255, 255)
        self.red = (200, 0, 0)
        self.blue1 = (0, 0, 255)
        self.blue2 = (0, 100, 255)
        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)

    def draw(self, display: pygame.Surface, snake_body: Sequence[Any], food: Any,
             score: int, screen_width: int) -> None:
        """Render the board with a gray title bar showing the score.

        - screen_width: width in pixels used to size the title bar and center text.
        """
        # title bar (gray)
        title_rect = pygame.Rect(0, 0, screen_width, self.title_height)
        pygame.draw.rect(display, self.gray, title_rect)

        # score text centered vertically in title bar, left aligned with small padding
        score_text = self.font.render(f"Score: {score}", True, self.black)
        display.blit(score_text, (8, (self.title_height - score_text.get_height()) // 2))

        # board area below title (black background)
        board_offset_y = self.title_height
        display.fill(self.black, pygame.Rect(0, board_offset_y, screen_width, display.get_height() - board_offset_y))

        # draw snake and food using absolute coordinates (already below title)
        for seg in snake_body:
            outer = pygame.Rect(seg.x, seg.y, BLOCK_SIZE, BLOCK_SIZE)
            inner = pygame.Rect(seg.x + 4, seg.y + 4, 12, 12)
            pygame.draw.rect(display, self.blue1, outer)
            pygame.draw.rect(display, self.blue2, inner)

        pygame.draw.rect(display, self.red, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        # redraw score on top in white for contrast
        score_text_white = self.font.render(f"Score: {score}", True, self.white)
        display.blit(score_text_white, (8, (self.title_height - score_text_white.get_height()) // 2))

        pygame.display.flip()

import pygame
from typing import Sequence, Any
from utils.common import BLOCK_SIZE


class Dashboard:
    """Draw the game's UI: snake, food, and score, with a gray title bar."""

    def __init__(self, font: pygame.font.Font):
        self.font = font
        # title height computed from font height plus 4px padding top and bottom
        pad = 4
        # Make title height a multiple of BLOCK_SIZE so the play grid lines up
        raw_height = max(BLOCK_SIZE, self.font.get_height() + pad * 2)
        # ceil to nearest BLOCK_SIZE
        self.title_height = ((raw_height + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        # color attributes (lowercase as requested)
        self.white = (255, 255, 255)
        self.red = (200, 0, 0)
        self.blue1 = (0, 0, 255)
        self.blue2 = (0, 100, 255)
        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)

    def _draw_title_bar(self, display: pygame.Surface, score: int, screen_width: int) -> None:
        """Draw the gray title bar and the score label/counter.

        This encapsulates padding, bold label rendering and positioning so the
        main draw() remains focused on board rendering.
        """
        # title bar (gray)
        title_rect = pygame.Rect(0, 0, screen_width, self.title_height)
        pygame.draw.rect(display, self.gray, title_rect)

        # padding and baseline
        pad = 4
        text_y = pad  # top padding
        x = pad  # left padding

        # Render 'Score' label in bold blue, then the counter in black
        prev_bold = self.font.get_bold()
        try:
            self.font.set_bold(True)
            label_surf = self.font.render("Score:", True, self.blue1)
        finally:
            self.font.set_bold(prev_bold)

        counter_surf = self.font.render(str(score), True, self.black)

        # blit label and counter with a small gap
        display.blit(label_surf, (x, text_y))
        display.blit(counter_surf, (x + label_surf.get_width() + 6, text_y))

    def draw(self, display: pygame.Surface, snake_body: Sequence[Any], food: Any,
             score: int, screen_width: int) -> None:
        """Render the board with a gray title bar showing the score.

        - screen_width: width in pixels used to size the title bar and center text.
        """
        # Draw the title bar (delegated)
        self._draw_title_bar(display, score, screen_width)

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

        pygame.display.flip()

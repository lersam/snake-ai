import pygame
import logging
from pathlib import Path
from utils.common import Direction, Point, BLOCK_SIZE, SPEED, ScreenSize
from snake import Snake
from dashboard import Dashboard
from food import Food

pygame.init()
# Load bundled font if available, otherwise fall back to system font
font = pygame.font.Font(Path(Path(__file__).parent, "support/arial.ttf").absolute(), 25)

# configure simple logger for debugging (set to DEBUG to surface collision diagnostics)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class SnakeGame:
    def __init__(self, screen_size: ScreenSize = ScreenSize(640, 480)):
        self.screen_size = screen_size
        self.font = font

        self.display = pygame.display.set_mode((self.screen_size.width, self.screen_size.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT

        self.dashboard = Dashboard(font)

        self.score = 0
        self.frame_iteration = 0
        self.food = Food()
        self._reset_ui()

    def _reset_ui(self) -> None:
        self.score = 0
        self.snake = Snake(self.compute_initial_head_point())

        # place initial food via Food; ensure it is below the title bar
        self.food.place_random(self.screen_size.width, self.screen_size.height, BLOCK_SIZE, self.snake.body, y_min=self.dashboard.title_height)

    def compute_initial_head_point(self) -> Point:
        """Compute and return a grid-aligned Point for the snake's starting head position.

        The point is horizontally centered and vertically centered inside the playable
        area (screen height minus title/header). Both coordinates are rounded to
        multiples of BLOCK_SIZE so the head, movement, and food all align to the same grid.
        """
        grid_center_x = (self.screen_size.width // 2) // BLOCK_SIZE * BLOCK_SIZE

        # center vertically within the playable board (below title) and align to grid
        playable_height = self.screen_size.height - self.dashboard.title_height
        grid_center_y = (playable_height // 2) // BLOCK_SIZE * BLOCK_SIZE + self.dashboard.title_height
        logger.debug("computed grid_center_x=%s grid_center_y=%s playable_height=%s", grid_center_x,
                     grid_center_y, playable_height)

        # debug: log metrics affecting layout using percent-style logging
        logger.debug("title_height=%s font_height=%s BLOCK_SIZE=%s", self.dashboard.title_height,
                     (self.font.get_height() if hasattr(self, 'font') else 'N/A'), BLOCK_SIZE)

        return Point(grid_center_x, grid_center_y)

    def play_step(self) -> tuple[bool, int]:
        self.frame_iteration += 1
        # 1. process events
        if not self._handle_events():
            return True, self.score

        # 2. move
        self.snake.move(self.direction)

        # Debug: log head and food every frame to diagnose collection issues
        logger.debug("FRAME head=(%s,%s) food=(%s,%s) score=%s", self.snake.head.x, self.snake.head.y,
                     self.food.x, self.food.y, self.score)

        # 3. food handling: decide whether the tail will be removed
        head_rect = pygame.Rect(self.snake.head.x, self.snake.head.y, BLOCK_SIZE, BLOCK_SIZE)
        food_rect = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        ate_food = head_rect.colliderect(food_rect)
        if ate_food:
            logger.info("Food eaten at (%s,%s) via rect collision", self.food.x, self.food.y)
            self.score += 1
            # place new food below the title bar only
            self.food.place_random(self.screen_size.width, self.screen_size.height, BLOCK_SIZE, self.snake.body,
                                   y_min=self.dashboard.title_height)
        else:
            # If the snake did not eat, remove the tail now so collision checks ignore
            # the cell that will be vacated this turn (classic snake rule).
            tail = self.snake.body[-1]
            self.snake.remove_tail()
            # If the new head equals the old tail position and we've removed it,
            # that's allowed — log at debug level to help diagnosing false positives.
            if self.snake.head == tail:
                logger.debug("Moved into tail cell which was removed: head=%s tail=%s", self.snake.head, tail)

        # 4. check collision (after possibly removing tail)
        if self.snake.is_collision(self.screen_size.width, self.screen_size.height, board_offset_y=self.dashboard.title_height):
            return True, self.score

        # 5. finish game when none iterations
        if self.frame_iteration > 100 * len(self.snake):
            return True, self.score

        # 6. update UI and tick
        self._update_ui()
        self.clock.tick(SPEED)

        return False, self.score

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            logger.debug("EVENT: %s", event)
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        return True

    def _update_ui(self) -> None:
        # delegate drawing to Dashboard (pass width for title layout)
        self.dashboard.draw(self.display, self.snake.body, self.food, self.score, self.screen_size.width)


if __name__ == '__main__':
    game = SnakeGame()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Final Score:', score)
    pygame.quit()

import pygame
import logging

from game.utils.common import Direction, Point, BLOCK_SIZE, SPEED, ScreenSize
from game.snake import Snake
from game.dashboard import Dashboard
from game.food import Food

pygame.init()
# Load bundled font if available, otherwise fall back to system font
font = pygame.font.Font("support/arial.ttf", 25)

# configure simple logger for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class SnakeGame:
    def __init__(self, screen_size: ScreenSize = ScreenSize(640, 480)):
        self.screen_size = screen_size
        self.width = screen_size.width
        self.height = screen_size.height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT

        # Create dashboard first so we can read its title_height
        self.dashboard = Dashboard(font)
        self.title_height = self.dashboard.title_height
        # debug: log metrics affecting layout using percent-style logging
        logger.debug("title_height=%s font_height=%s BLOCK_SIZE=%s", self.title_height,
                     (self.font.get_height() if hasattr(self, 'font') else 'N/A'), BLOCK_SIZE)

        # initial head and snake; place head below title bar
        # align head to grid so movement and food placement match (multiples of BLOCK_SIZE)
        grid_center_x = (self.width // 2) // BLOCK_SIZE * BLOCK_SIZE
        # center vertically within the playable board (below title) and align to grid
        playable_height = self.height - self.title_height
        grid_center_y = (playable_height // 2) // BLOCK_SIZE * BLOCK_SIZE + self.title_height
        logger.debug("computed grid_center_x=%s grid_center_y=%s playable_height=%s", grid_center_x,
                     grid_center_y, playable_height)
        head = Point(grid_center_x, grid_center_y)
        self.snake = Snake(head)

        self.score = 0
        self.food = Food()
        # place initial food via Food; ensure it is below the title bar
        self.food.place_random(self.width, self.height, BLOCK_SIZE, self.snake.body, y_min=self.title_height)

    # food placement is handled by the Food class (Food.place_random)

    def play_step(self):
        # 1. process events
        if not self._handle_events():
            return True, self.score

        # 2. move
        self.snake.move(self.direction)

        # Debug: log head and food every frame to diagnose collection issues
        logger.debug("FRAME head=(%s,%s) food=(%s,%s) score=%s", self.snake.head.x, self.snake.head.y,
                     self.food.x,self.food.y, self.score)

        # 3. check collision (pass board_offset_y so boundaries consider title)
        board_offset_y = self.title_height
        if self.snake.is_collision(self.width, self.height, board_offset_y=board_offset_y):
            return True, self.score

        # 4. food handling
        # use rect collision so small coordinate mismatches don't block eating
        head_rect = pygame.Rect(self.snake.head.x, self.snake.head.y, BLOCK_SIZE, BLOCK_SIZE)
        food_rect = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        if head_rect.colliderect(food_rect):
            logger.info("Food eaten at (%s,%s) via rect collision", self.food.x, self.food.y)
            self.score += 1
            # place new food below the title bar only
            self.food.place_random(self.width, self.height, BLOCK_SIZE, self.snake.body, y_min=self.title_height)
        else:
            self.snake.remove_tail()

        # 5. update UI and tick
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return status
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
        self.dashboard.draw(self.display, self.snake.body, self.food, self.score, self.width)


if __name__ == '__main__':
    game = SnakeGame()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Final Score:', score)
    pygame.quit()

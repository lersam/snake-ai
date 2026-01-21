import pygame

from game.utils.common import Direction, Point, BLOCK_SIZE, SPEED, ScreenSize
from game.snake import Snake
from game.dashboard import Dashboard
from game.food import Food

pygame.init()
# Load bundled font if available, otherwise fall back to system font
font = pygame.font.Font("support/arial.ttf", 25)


class SnakeGame:
    def __init__(self, screen_size: ScreenSize = ScreenSize(640, 480)):
        self.screen_size = screen_size
        self.width = screen_size.width
        self.height = screen_size.height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT

        # initial head and snake; place head below title bar
        title_height = BLOCK_SIZE
        head = Point(self.width // 2, self.height // 2 + title_height)
        self.snake = Snake(head)

        self.score = 0
        self.food = Food()
        self.dashboard = Dashboard(font)
        # place initial food via Food, ensure y_min prevents placement under title
        self.food.place_random(self.width, self.height, BLOCK_SIZE, self.snake.body, y_min=title_height)

    # food placement is handled by the Food class (Food.place_random)

    def play_step(self):
        # 1. process events
        if not self._handle_events():
            return True, self.score

        # 2. move
        self.snake.move(self.direction)

        # 3. check collision (pass board_offset_y so boundaries consider title)
        board_offset_y = BLOCK_SIZE
        if self.snake.is_collision(self.width, self.height, board_offset_y=board_offset_y):
            return True, self.score

        # 4. food handling
        if self.snake.head == self.food:
            self.score += 1
            self.food.place_random(self.width, self.height, BLOCK_SIZE, self.snake.body, y_min=board_offset_y)
        else:
            self.snake.remove_tail()

        # 5. update UI and tick
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return status
        return False, self.score

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
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

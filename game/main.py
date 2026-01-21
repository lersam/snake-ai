import pygame

from utils.common import Direction, Point, BLOCK_SIZE, SPEED
from game.snake import Snake
from game.dashboard import Dashboard
from game.food import Food

pygame.init()
font = pygame.font.Font("support/arial.ttf", 25)


class SnakeGame:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT

        # initial head and snake
        head = Point(self.width // 2, self.height // 2)
        self.snake = Snake(head)

        self.score = 0
        self.food = Food()
        self.dashboard = Dashboard(font)
        # place initial food via Food
        self.food.place_random(self.width, self.height, BLOCK_SIZE, self.snake.body)

    # food placement is handled by the Food class (Food.place_random)

    def play_step(self):
        # 1. process events
        if not self._handle_events():
            return True, self.score

        # 2. move
        self.snake.move(self.direction)

        # 3. check collision
        if self.snake.is_collision(self.width, self.height):
            return True, self.score

        # 4. food handling
        if self.snake.head == self.food:
            self.score += 1
            self.food.place_random(self.width, self.height, BLOCK_SIZE, self.snake.body)
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
        # delegate drawing to Dashboard
        self.dashboard.draw(self.display, self.snake.body, self.food, self.score)


if __name__ == '__main__':
    game = SnakeGame()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Final Score:', score)
    pygame.quit()

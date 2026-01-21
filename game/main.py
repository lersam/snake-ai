import os
import pygame
import random

from game.common import Direction, Point, BLOCK_SIZE, SPEED
from game.snake import Snake

pygame.init()
font = pygame.font.Font("support/arial.ttf", 25)

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class SnakeGame:
    def __init__(self, width: int = 640, height: int = 480):
        self.width, self.height = width, height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT

        # initial head and snake
        head = Point(self.width // 2, self.height // 2)
        self.snake = Snake(head)
        self.head = self.snake.head

        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self) -> None:
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake.body:
                break

    def play_step(self):
        if not self._handle_events():
            return True, self.score

        # move snake (it inserts the new head)
        self.snake.move(self.direction)
        # keep `head` in sync for compatibility with previous code
        self.head = self.snake.head

        # check collision
        if self._is_collision():
            return True, self.score

        # food handling
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            # remove tail if no food eaten
            self.snake.remove_tail()

        # draw and tick
        self._update_ui()
        self.clock.tick(SPEED)
        return False, self.score

    def _handle_events(self) -> bool:
        key_to_direction = {
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                self.direction = key_to_direction.get(event.key, self.direction)
        return True

    def _is_collision(self) -> bool:
        # boundary check
        if (
                self.head.x < 0 or self.head.x >= self.width or
                self.head.y < 0 or self.head.y >= self.height
        ):
            return True
        # self collision
        return self.snake.collides_self()

    def _update_ui(self) -> None:
        self.display.fill(BLACK)
        for segment in self.snake.body:
            pygame.draw.rect(
                self.display,
                BLUE1,
                pygame.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                BLUE2,
                pygame.Rect(segment.x + 4, segment.y + 4, 12, 12),
            )
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()


if __name__ == '__main__':
    game = SnakeGame()
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Final Score:', score)
    pygame.quit()

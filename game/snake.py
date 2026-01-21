from typing import List

# Support running as a submodule or as a script: try relative import first, then fallback
try:
    from .common import Point, Direction, BLOCK_SIZE
except Exception:
    from common import Point, Direction, BLOCK_SIZE  # type: ignore


class Snake:
    """Encapsulates the snake's position and movement."""

    def __init__(self, head: Point, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.head = head
        self.body: List[Point] = [
            head,
            Point(head.x - block_size, head.y),
            Point(head.x - 2 * block_size, head.y),
        ]

    def move(self, direction: Direction) -> None:
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
        self.head = Point(x, y)
        self.body.insert(0, self.head)

    def remove_tail(self) -> None:
        self.body.pop()

    def collides_self(self) -> bool:
        return self.head in self.body[1:]

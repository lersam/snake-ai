from typing import List
from game.utils.common import Point, Direction, BLOCK_SIZE


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

    def collides_self(self, board_offset_y: int = 0) -> bool:
        """Check self collision considering an optional vertical offset.

        If the drawing uses an offset (title bar), snake segments and head may be
        shifted on-screen. This method allows callers to pass board_offset_y to
        adjust checks where needed.
        """
        if board_offset_y == 0:
            return self.head in self.body[1:]
        # shift head and body for comparison
        shifted_head = Point(self.head.x, self.head.y - board_offset_y)
        shifted_body = [Point(p.x, p.y - board_offset_y) for p in self.body[1:]]
        return shifted_head in shifted_body

    def is_collision(self, width: int, height: int, board_offset_y: int = 0) -> bool:
        """Return True if the snake's head collides with boundary or itself.

        Width and height are the game area dimensions in pixels. When a title bar
        (or other UI) is present above the game board, pass board_offset_y so
        boundaries are computed relative to the board area.
        """
        # effective board height reduces by any top offset
        effective_height = height - board_offset_y
        # head adjusted relative to board origin
        head_y_rel = self.head.y - board_offset_y
        # boundary check
        if self.head.x < 0 or self.head.x >= width or head_y_rel < 0 or head_y_rel >= effective_height:
            return True
        # self collision
        return self.collides_self(board_offset_y)

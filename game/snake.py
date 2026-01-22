from typing import List, Optional
from utils.common import Point, Direction, BLOCK_SIZE
import logging

logger = logging.getLogger(__name__)


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

    def _check_and_log_collision(self, head: Point, segments: List[Point], board_offset_y: Optional[int] = None) -> bool:
        """Helper to check whether head collides with segments and log which segment.

        Returns True if a collision is found. If board_offset_y is provided the
        log message includes offset context.
        """
        collided = head in segments
        if not collided:
            return False

        for idx, seg in enumerate(segments, start=1):
            if seg == head:
                if board_offset_y is None:
                    logger.debug("Self-collision detected: head=%s equals body[%s]=%s", head, idx, seg)
                else:
                    logger.debug(
                        "Self-collision with offset: shifted_head=%s equals shifted_body[%s]=%s (board_offset_y=%s)",
                        head, idx, seg, board_offset_y)
                break
        return True

    def collides_self(self, board_offset_y: int = 0) -> bool:
        """Check self collision considering an optional vertical offset.

        If the drawing uses an offset (title bar), snake segments and head may be
        shifted on-screen. This method allows callers to pass board_offset_y to
        adjust checks where needed.
        """
        if board_offset_y == 0:
            # reuse helper for non-offset case
            return self._check_and_log_collision(self.head, self.body[1:], None)

        # shift head and body for comparison and reuse helper
        shifted_head = Point(self.head.x, self.head.y - board_offset_y)
        shifted_body = [Point(p.x, p.y - board_offset_y) for p in self.body[1:]]
        return self._check_and_log_collision(shifted_head, shifted_body, board_offset_y)

    def is_collision(self, width: int, height: int, board_offset_y: int = 0) -> bool:
        """Return True if the snake's head collides with boundary or itself.

        Width and height are the game area dimensions in pixels. When a title bar
        (or other UI) is present above the game board, pass board_offset_y so
        boundaries are computed relative to the board area.
        """
        effective_height = height - board_offset_y
        # head adjusted relative to board origin
        head_y_rel = self.head.y - board_offset_y

        # boundary check
        out_of_bounds = (
            self.head.x < 0
            or self.head.x >= width
            or head_y_rel < 0
            or head_y_rel >= effective_height
        )
        if out_of_bounds:
            logger.debug(
                "Boundary collision: head=(%s,%s) head_y_rel=%s width=%s effective_height=%s board_offset_y=%s",
                self.head.x,
                self.head.y,
                head_y_rel,
                width,
                effective_height,
                board_offset_y,
            )
            return True

        # self collision
        collided_self = self.collides_self(board_offset_y)
        if collided_self:
            logger.debug("Collision reported by collides_self(head=%s, board_offset_y=%s)", self.head, board_offset_y)
        return collided_self

    def __len__(self):
        return len(self.body)

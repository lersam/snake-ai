import random
from game.utils.common import Point


class Food:
    """Represents food on the board.

    Behaves like a Point for equality and attribute access so existing code that
    expects a Point (checks, drawing) continues to work unchanged.
    """

    def __init__(self, x: int | None = None, y: int | None = None):
        self.x = x
        self.y = y

    @property
    def pos(self) -> Point:
        return Point(self.x, self.y)

    def place_random(self, width: int, height: int, block_size: int, snake_body, y_min: int = 0) -> None:
        """Pick a random location aligned to `block_size` not colliding with snake.

        y_min: minimum y (inclusive) where food may be placed — used to keep food
        below a UI title bar.
        """
        while True:
            x = random.randint(0, (width - block_size) // block_size) * block_size
            # y is chosen between y_min and height - block_size inclusive at block granularity
            min_row = y_min // block_size
            max_row = (height - block_size) // block_size
            y = random.randint(min_row, max_row) * block_size
            candidate = Point(x, y)
            # snake_body contains Points
            if candidate not in snake_body:
                self.x = x
                self.y = y
                return

    def __eq__(self, other) -> bool:  # compare to Point or another Food
        try:
            ox = other.x
            oy = other.y
        except Exception:
            return False
        return self.x == ox and self.y == oy

    def __repr__(self) -> str:
        return f"Food(x={self.x}, y={self.y})"

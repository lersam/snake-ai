# game.utils package
try:
    from . import common, dashboard_color
except Exception:
    # Fallback for static analysis or unusual import contexts
    import importlib

    common = importlib.import_module('game.utils.common')
    dashboard_color = importlib.import_module('game.utils.dashboard_color')

__all__ = ["common", "dashboard_color"]

# GitHub Copilot Instructions

Align AI code suggestions and human contributions for a Python/Pygame project. Enforce consistent architecture, safe game-loop usage, performance practices, and maintainable code.
# Python Coding Style
- Follow PEP 8, the official Python style guide, when generating or editing Python code.
- Use clear naming conventions (snake_case for variables and functions, CapWords for classes).
- Format code with consistent indentation (4 spaces), line length around 119 characters, and meaningful docstrings.
- Prefer readability and maintainability over cleverness.
- Use type hints for function signatures where appropriate.
- Use f-strings for string formatting instead of older methods like `%` or `str.format()`.
- Use logging instead of print statements for debug and runtime information.
- Use list comprehensions and generator expressions for concise and efficient looping.
- Handle exceptions using try/except blocks, and avoid bare except clauses.
- Use context managers (with statements) for resource management (e.g., file handling).
- Avoid global variables; prefer passing parameters and returning values.
- Write modular code with functions and classes that have single responsibilities.
- Write unit tests for new features and bug fixes, and ensure existing tests pass.
- Global variables should be avoided; use function parameters and return values instead.

[//]: # (## Project context)

## Implementation guidance

[//]: # (- Favor small, composable functions and explicit async/await usage in I/O paths.)

[//]: # (- Preserve existing configuration loaders &#40;`get_config_data`, `config`&#41; instead of introducing new env readers unless required.)

[//]: # (- When touching database or models, update Alembic migrations accordingly.)

[//]: # (- Always add or update unit tests in `tests/` &#40;create the directory if missing&#41; when behavior changes.)

## Documentation & comments
- Keep docstrings short but descriptive; prefer Markdown tables for API schema additions in README updates.
- Annotate complex control flow with brief comments that explain the intent rather than the mechanics.

[//]: # (## Security & validation)

[//]: # ()
[//]: # (## Performance & resilience)


# GitHub Copilot Instructions

Align AI code suggestions and human contributions for a Python/Pygame project. Enforce consistent architecture, safe game-loop usage, performance practices, and maintainable code.

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


# GitHub Copilot Instructions

These preferences steer Copilot when proposing completions for this repository.

## Project context
- Primary stack: Python 3.11, FastAPI, Alembic, Kafka integrations.
- Coding style: type hints when practical, Pydantic models for payloads, FastAPI dependency patterns, pytest style tests.


## Implementation guidance
- Favor small, composable functions and explicit async/await usage in I/O paths.
- Preserve existing configuration loaders (`get_config_data`, `config`) instead of introducing new env readers unless required.
- When touching database or models, update Alembic migrations accordingly.
- Always add or update unit tests in `tests/` (create the directory if missing) when behavior changes.

## Documentation & comments
- Keep docstrings short but descriptive; prefer Markdown tables for API schema additions in README updates.
- Annotate complex control flow with brief comments that explain the intent rather than the mechanics.

## Security & validation
- Validate external inputs via Pydantic models; never trust raw request payloads.
- Redact secrets and PII in logs, configs, and responses.

## Performance & resilience
- Avoid blocking calls inside async endpoints; offload to executors when necessary.
- Reuse Kafka and DB connections; do not create per-request unless unavoidable.
- Guard background tasks with retries and timeout handling.


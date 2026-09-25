from __future__ import annotations

PROVIDERS = ("claude", "deepseek")


def get_provider(name: str):
    if name == "claude":
        from app.answer.claude import ClaudeProvider

        return ClaudeProvider()
    if name == "deepseek":
        from app.answer.deepseek import DeepSeekProvider

        return DeepSeekProvider()
    raise ValueError(f"Unknown provider {name!r}; expected one of {PROVIDERS}")

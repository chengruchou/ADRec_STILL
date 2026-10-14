"""Minimal registry for stable public model identifiers."""

MODEL_REGISTRY: dict[str, object] = {}


def register_model(name: str, model: object) -> None:
    if name in MODEL_REGISTRY:
        raise KeyError(f"model already registered: {name}")
    MODEL_REGISTRY[name] = model

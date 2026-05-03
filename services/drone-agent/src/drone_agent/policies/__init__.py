"""Policy registry. KAM-5 ships `noop` (default), `ppo`, and `pronav`.

Each entry is loaded lazily so the `noop` path never imports torch.
"""

from __future__ import annotations

from collections.abc import Callable

from drone_agent.config import DroneConfig
from drone_agent.policy import Policy

_PolicyFactory = Callable[[DroneConfig], Policy]


def _make_noop(cfg: DroneConfig) -> Policy:
    from drone_agent.policies.noop import NoOpPolicy

    return NoOpPolicy()


def _make_ppo(cfg: DroneConfig) -> Policy:
    from drone_agent.policies.ppo import PPOPolicy

    return PPOPolicy(cfg.ppo_checkpoint)


def _make_pronav(cfg: DroneConfig) -> Policy:
    from drone_agent.policies.pronav import ProNavMlpPolicy

    return ProNavMlpPolicy()


REGISTRY: dict[str, _PolicyFactory] = {
    "noop": _make_noop,
    "ppo": _make_ppo,
    "pronav": _make_pronav,
}


def build(cfg: DroneConfig) -> Policy:
    """Build the configured policy, falling back to NoOp on any import or
    checkpoint error so the drone always boots."""
    factory = REGISTRY.get(cfg.policy)
    if factory is None:
        from kamikaze_common.logging import get_logger

        get_logger("policy").warning(
            "unknown POLICY=%s; valid=%s — falling back to noop",
            cfg.policy,
            list(REGISTRY.keys()),
        )
        return _make_noop(cfg)

    try:
        return factory(cfg)
    except Exception as exc:
        from kamikaze_common.logging import get_logger

        get_logger("policy").warning(
            "policy=%s failed to init (%s) — falling back to noop", cfg.policy, exc
        )
        return _make_noop(cfg)

"""Training-time smooth relaxations for discrete simulator gates."""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only when jax is absent
    jax = None
    jnp = None


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise RuntimeError("jax is required for smooth differentiable relaxations")


def smooth_gate(value, *, sharpness: float = 24.0, threshold: float = 0.0):
    """Sigmoid gate centered on a threshold."""

    _require_jax()
    return jax.nn.sigmoid(float(sharpness) * (value - threshold))


def smooth_positive(value, *, sharpness: float = 24.0):
    """Smooth approximation to `max(value, 0)`."""

    _require_jax()
    scale = jnp.maximum(jnp.asarray(sharpness, dtype=jnp.float32), 1e-6)
    return jax.nn.softplus(scale * value) / scale


def smooth_clip(value, lower, upper, *, sharpness: float = 24.0):
    """Differentiable approximation to clipping into `[lower, upper]`."""

    _require_jax()
    return lower + smooth_positive(value - lower, sharpness=sharpness) - smooth_positive(value - upper, sharpness=sharpness)


def smooth_trade_amount(amount, *, minimum: float = 0.0001, sharpness: float = 24.0):
    """Softly suppress tiny trades instead of hard zeroing them out."""

    _require_jax()
    gate = smooth_gate(amount, sharpness=sharpness, threshold=minimum)
    return smooth_positive(amount, sharpness=sharpness) * gate


def smooth_or(left, right):
    """Smooth OR for two gates in `[0, 1]`."""

    _require_jax()
    return left + right - left * right

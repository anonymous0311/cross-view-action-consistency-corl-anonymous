"""FiLM (Feature-wise Linear Modulation) for per-layer camera conditioning.

Design:
- `SharedFiLMGenerator` is a small MLP that maps a conditioning vector
  (D-dim projected camera token) to a (2*D*num_layers) tensor of (gamma, beta)
  per layer. Sharing one generator across layers keeps parameter count O(D^2)
  instead of O(L * D^2).
- `FiLMLayer` applies `h <- (1 + gamma) * h + beta` at the call site. We use
  `1 + gamma` rather than raw `gamma` so a zero-initialized head starts the
  model as identity (standard FiLM-residual trick).
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class SharedFiLMGenerator(nnx.Module):
    """Single MLP that produces (gamma, beta) for every layer of a decoder.

    Output layout: (..., num_layers, 2, d). Call `per_layer(out, i)` to pull
    (gamma_i, beta_i) for layer i.
    """

    def __init__(
        self,
        cond_dim: int,
        d: int,
        num_layers: int,
        *,
        hidden_dim: int | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        self.d = d
        self.num_layers = num_layers
        hidden = hidden_dim if hidden_dim is not None else cond_dim
        self.fc1 = nnx.Linear(cond_dim, hidden, rngs=rngs)
        # Zero-init the final projection so gamma=0, beta=0 at start => identity.
        self.fc2 = nnx.Linear(
            hidden,
            num_layers * 2 * d,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, cond: jax.Array) -> jax.Array:
        """cond: (..., cond_dim) -> params: (..., num_layers, 2, d)."""
        h = nnx.gelu(self.fc1(cond))
        h = self.fc2(h)
        return h.reshape(*cond.shape[:-1], self.num_layers, 2, self.d)

    @staticmethod
    def per_layer(params: jax.Array, layer_idx: int) -> tuple[jax.Array, jax.Array]:
        """Extract (gamma, beta) for a specific layer from the generator output."""
        gamma = params[..., layer_idx, 0, :]
        beta = params[..., layer_idx, 1, :]
        return gamma, beta


class FiLMLayer(nnx.Module):
    """Applies `h <- (1 + gamma) * h + beta`.

    Stateless in the sense that it carries no learnable parameters on its own;
    (gamma, beta) come from `SharedFiLMGenerator`. We keep it as a Module for
    symmetry with the rest of the stack, which makes it easy to slot into a
    layer list.
    """

    def __call__(self, h: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
        # gamma/beta typically have shape (..., d) and broadcast over sequence dim.
        if gamma.ndim == h.ndim - 1:
            gamma = gamma[..., None, :]
            beta = beta[..., None, :]
        return (1.0 + gamma) * h + beta

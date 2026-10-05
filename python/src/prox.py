"""Right negative-entropy and left Burg Bregman proximal operators."""

from math import sqrt

import torch
from deepinv.models import TVDenoiser


##################
# Negative Entropy
##################


@torch.no_grad()
def rprox_negentropy_l1(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    sensitivity: torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """Right Bregman L1 prox for weighted negative entropy."""
    return (sensitivity * x) / (sensitivity + weight).clamp_min(filter_epsilon)


@torch.no_grad()
def rprox_negentropy_l2(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    sensitivity: torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """Right Bregman L2 prox for weighted negative entropy."""
    # Solve weight*u**2 + sensitivity*u = sensitivity*x. Rationalizing
    # avoids cancellation for small weights and recovers x when weight is zero.
    scaled_x = sensitivity * x
    return (
        2
        * scaled_x
        / (
            sensitivity + torch.sqrt(sensitivity.square() + 4 * weight * scaled_x)
        ).clamp_min(filter_epsilon)
    )


@torch.no_grad()
def rprox_negentropy_TV(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    sensitivity: torch.Tensor,
    filter_epsilon: float = 1e-20,
    n_iter: int = 50,
) -> torch.Tensor:
    r"""Approximate the right Bregman TV prox with standard PDHG.

    This minimizes

    .. math::
        \mathrm{weight}\,\|\nabla u\|_{2,1}
        + \langle s, u - x\log(u)\rangle

    over positive ``u``. The data-term prox is evaluated in closed form.
    Spatial differences and their adjoint use DeepInverse's ``TVDenoiser``
    operators in 2D and 3D.
    """

    weight_tensor = torch.as_tensor(weight, device=x.device, dtype=x.dtype)
    if weight_tensor.numel() != 1:
        raise ValueError("weight must be a scalar.")
    weight_value = float(weight_tensor.item())
    if not torch.isfinite(weight_tensor) or weight_value < 0:
        raise ValueError("weight must be finite and nonnegative.")

    x = x.clamp_min(filter_epsilon)
    if n_iter == 0 or weight_value == 0:
        return x.clone()

    sensitivity = sensitivity.to(device=x.device, dtype=x.dtype).clamp_min(
        filter_epsilon
    )
    spatial_dims = x.dim() - 2

    # ||gradient||^2 <= 4 * spatial_dims. These symmetric step sizes therefore
    # satisfy tau * sigma * ||gradient||^2 < 1.
    primal_stepsize = 0.99 / sqrt(4 * spatial_dims)
    dual_stepsize = primal_stepsize

    primal = x.clone()
    primal_bar = primal.clone()
    dual = torch.zeros(
        (*x.shape, spatial_dims),
        device=x.device,
        dtype=x.dtype,
    )

    for _ in range(n_iter):
        dual = dual + dual_stepsize * TVDenoiser.nabla(primal_bar)
        dual_norm = torch.linalg.vector_norm(dual, dim=-1, keepdim=True)
        dual_scale = torch.maximum(
            torch.ones_like(dual_norm),
            dual_norm / weight_tensor,
        )
        dual = dual / dual_scale

        previous_primal = primal
        value = primal - primal_stepsize * TVDenoiser.nabla_adjoint(dual)
        linear_term = value - primal_stepsize * sensitivity
        product = primal_stepsize * sensitivity * x
        root = torch.sqrt(linear_term.square() + 4 * product)

        # Both expressions solve the data-term prox quadratic. The second
        # avoids cancellation when linear_term is negative.
        direct = 0.5 * (linear_term + root)
        stable = 2 * product / (root - linear_term).clamp_min(filter_epsilon)
        primal = torch.where(linear_term >= 0, direct, stable).clamp_min(filter_epsilon)
        primal_bar = 2 * primal - previous_primal

    return primal


##################
# Burg's Entropy
##################


@torch.no_grad()
def lprox_burg_l1(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """Left Bregman L1 prox for Burg's entropy."""
    return x / (1 + weight * x).clamp_min(filter_epsilon)


@torch.no_grad()
def lprox_burg_l2(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """Left Bregman L2 prox for Burg's entropy."""
    # The optimality condition is weight*u + 1/x - 1/u = 0.
    return (
        2 * x / (1 + torch.sqrt(1 + 4 * weight * x.square())).clamp_min(filter_epsilon)
    )

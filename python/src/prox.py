from math import sqrt
from typing import Literal

import torch


TVProxMethod = Literal["dual", "pdhg"]

##################
# Negative Entropy
##################


@torch.no_grad()
def right_prox_L1_negative_entropy(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    sensitivity: torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Proximal operator for the L1 regularizer with respect to the negative entropy.

    """
    prox = (sensitivity * x) / (sensitivity + weight).clamp(min=filter_epsilon)

    return prox


@torch.no_grad()
def right_prox_L2_negative_entropy(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    sensitivity: torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Proximal operator for the L2 regularizer with respect to the negative entropy.

    """
    # Solve weight*u**2 + sensitivity*u = sensitivity*x. Rationalizing
    # avoids cancellation for small weights and recovers x when weight is zero.
    scaled_x = sensitivity * x
    prox = 2 * scaled_x / (
        sensitivity + torch.sqrt(sensitivity.square() + 4 * weight * scaled_x)
    ).clamp_min(filter_epsilon)

    return prox


@torch.no_grad()
def right_prox_TV_negative_entropy(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    sensitivity: torch.Tensor,
    filter_epsilon: float = 1e-20,
    n_iter: int = 50,
    fista_tv: bool = False,
    method: TVProxMethod = "dual",
) -> torch.Tensor:
    """Approximate the right Bregman TV prox for weighted negative entropy.

    ``method="dual"`` uses the specialized dual-TV algorithm from Maxim et al.
    ``method="pdhg"`` uses the standard primal-dual hybrid gradient algorithm.
    """
    if method == "dual":
        prox = TV_dual_denoising(
            x=x,
            sensitivity=sensitivity,
            alpha=weight,
            n_iter=n_iter,
            epsilon=filter_epsilon,
            fista=fista_tv,
        )
    elif method == "pdhg":
        if fista_tv:
            raise ValueError("fista_tv is only available with method='dual'.")
        prox = TV_pdhg_denoising(
            x=x,
            sensitivity=sensitivity,
            alpha=weight,
            n_iter=n_iter,
            epsilon=filter_epsilon,
        )
    else:
        raise ValueError(
            f"Unknown TV prox method {method!r}; expected either 'dual' or 'pdhg'."
        )

    return prox


##################
# Burg's Entropy
##################


@torch.no_grad()
def left_prox_L1_burg_entropy(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Proximal operator for the L1 regularizer with respect to Burg's entropy.

    """
    prox = x / (torch.ones_like(x) + weight * x).clamp(min=filter_epsilon)

    return prox


@torch.no_grad()
def left_prox_L2_burg_entropy(
    x: torch.Tensor,
    weight: float | torch.Tensor,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Proximal operator for the L2 regularizer with respect to Burg's entropy.

    """
    # The optimality condition is weight*u + 1/x - 1/u = 0.
    prox = 2 * x / (1 + torch.sqrt(1 + 4 * weight * x.square())).clamp_min(
        filter_epsilon
    )

    return prox


###########################
# TV regularization
###########################


def pos(x_pos: torch.Tensor) -> torch.Tensor:
    """
    Compute pos(x) = max(0, x)

    Parameters
    ----------
    x_pos : torch.Tensor
        input tensor

    Return
    ------
    torch.Tensor
        ReLU(x_pos)
    """
    return torch.clamp(x_pos, min=0)


def torch_gradient(a: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of a using finite differences

    Parameters
    ----------
    a : torch.Tensor
        input tensor

    Returns
    -------
    torch.Tensor
        Spatial gradient(a) with shape (spatial_dims, *a.shape)
    """
    if a.dim() < 2:
        raise ValueError("torch_gradient expects at least 2D spatial data")

    # By convention in this repo, tensors are shaped (B, C, *spatial_dims).
    # We compute TV over spatial dimensions only.
    spatial_dims = a.dim() - 2 if a.dim() >= 3 else a.dim()
    spatial_axes = tuple(range(a.dim() - spatial_dims, a.dim()))

    # torch.gradient requires each differentiated axis size >= edge_order+1.
    # If a spatial axis is degenerate (size 1), its gradient is identically 0.
    grads = []
    for axis in spatial_axes:
        if a.shape[axis] < 2:
            grads.append(torch.zeros_like(a))
        else:
            grads.append(torch.gradient(a, dim=axis, edge_order=1)[0])
    return torch.stack(grads, dim=0)


def torch_gradient_div(a: torch.Tensor) -> torch.Tensor:
    """
    Compute backward gradient of a using finite differences

    Parameters
    ----------
    a : torch.Tensor
        input tensor

    Returns
    -------
    torch.Tensor
        Spatial backward gradient(a) with shape (spatial_dims, *a.shape)
    """
    if a.dim() < 2:
        raise ValueError("torch_gradient_div expects at least 2D spatial data")

    spatial_dims = a.dim() - 2 if a.dim() >= 3 else a.dim()
    spatial_axes = tuple(range(a.dim() - spatial_dims, a.dim()))

    grads = []
    for axis in spatial_axes:
        # edge_order=2 needs size >= 3; otherwise fall back to edge_order=1 (or 0 if degenerate)
        if a.shape[axis] < 2:
            grads.append(torch.zeros_like(a))
        elif a.shape[axis] < 3:
            grads.append(torch.gradient(a, dim=axis, edge_order=1)[0])
        else:
            grads.append(torch.gradient(a, dim=axis, edge_order=2)[0])
    return torch.stack(grads, dim=0)


def torch_divergence(u: torch.Tensor) -> torch.Tensor:
    """
    Compute divergence of u

    Parameters
    ----------
    u : torch.Tensor
        vector field with shape (dim, *spatial_dims)

    Returns
    -------
    torch.Tensor
        divergence(u)
    """
    if u.dim() < 3:
        raise ValueError(
            "torch_divergence expects a vector field shaped (spatial_dims, B, C, *spatial)"
        )

    spatial_dims = u.shape[0]
    a = u[0]
    if a.dim() < 2:
        raise ValueError("torch_divergence expects at least 2D spatial data")

    spatial_axes = tuple(range(a.dim() - spatial_dims, a.dim()))
    if len(spatial_axes) != spatial_dims:
        raise ValueError(
            f"Inconsistent shapes: u has {spatial_dims} components but inferred {len(spatial_axes)} spatial axes"
        )

    div = torch.zeros_like(a)
    for d, axis in enumerate(spatial_axes):
        if a.shape[axis] < 2:
            continue
        div = div + torch.gradient(u[d], dim=axis, edge_order=1)[0]
    return div


def torch_module(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norm along the first dimension

    Parameters
    ----------
    q : torch.Tensor
        tensor with shape (dim, *spatial_dims)

    Return
    ------
    torch.Tensor
        L2 norm along dim 0
    """
    return torch.linalg.norm(q, dim=0)


def torch_TV(x: torch.Tensor) -> float:
    """
    Compute TV norm of x

    Parameters
    ----------
    x : torch.array

    Returns
    -------
    res : float()
          TV norm of x
    """
    # x = crop(x,10)
    grad = torch_gradient(x)
    res = torch_module(grad)
    return float(torch.sum(res))


def div_zer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "Both matrix should have the same size"
    new = torch.zeros_like(a)
    mask = b > 0
    new[mask] = a[mask] / b[mask]
    return new


def _torch_gradient_adjoint(q: torch.Tensor) -> torch.Tensor:
    """Apply the exact adjoint of :func:`torch_gradient`."""
    if q.dim() < 4:
        raise ValueError("q must be shaped (spatial_dims, B, C, *spatial_dims).")

    spatial_dims = q.shape[0]
    a = q[0]
    inferred_spatial_dims = a.dim() - 2
    if spatial_dims != inferred_spatial_dims:
        raise ValueError(
            f"Inconsistent shapes: q has {spatial_dims} components but inferred "
            f"{inferred_spatial_dims} spatial axes."
        )
    spatial_axes = tuple(range(2, a.dim()))

    adjoint = torch.zeros_like(a)
    for component, axis in enumerate(spatial_axes):
        field = q[component]
        axis_size = field.shape[axis]
        if axis_size < 2:
            continue

        if axis_size == 2:
            field_sum = field.select(axis, 0) + field.select(axis, 1)
            adjoint.select(axis, 0).sub_(field_sum)
            adjoint.select(axis, 1).add_(field_sum)
            continue

        # torch.gradient uses first-order one-sided differences at the
        # boundaries and centered differences in the interior.
        adjoint.select(axis, 0).sub_(field.select(axis, 0))
        adjoint.select(axis, 1).add_(field.select(axis, 0))
        interior = field.narrow(axis, 1, axis_size - 2)
        adjoint.narrow(axis, 0, axis_size - 2).sub_(0.5 * interior)
        adjoint.narrow(axis, 2, axis_size - 2).add_(0.5 * interior)
        adjoint.select(axis, axis_size - 2).sub_(field.select(axis, axis_size - 1))
        adjoint.select(axis, axis_size - 1).add_(field.select(axis, axis_size - 1))

    return adjoint


def _negative_entropy_data_prox(
    value: torch.Tensor,
    x: torch.Tensor,
    sensitivity: torch.Tensor,
    stepsize: float,
    epsilon: float,
) -> torch.Tensor:
    r"""Prox of ``<s, u - x log(u)>`` evaluated at ``value``."""
    linear_term = value - stepsize * sensitivity
    product = stepsize * sensitivity * x
    root = torch.sqrt(linear_term.square() + 4 * product)

    # Both expressions solve the same quadratic. The second avoids
    # cancellation when linear_term is negative.
    direct = 0.5 * (linear_term + root)
    stable = 2 * product / (root - linear_term).clamp_min(epsilon)
    return torch.where(linear_term >= 0, direct, stable).clamp_min(epsilon)


def TV_pdhg_denoising(
    x: torch.Tensor,
    sensitivity: torch.Tensor,
    alpha: float | torch.Tensor,
    n_iter: int = 50,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    r"""Approximate the right Bregman TV prox with standard PDHG.

    This minimizes

    .. math::
        \alpha \|\nabla u\|_{2,1}
        + \langle s, u - x\log(u)\rangle

    over positive ``u``. The primal data-term prox is evaluated in closed form,
    and the dual TV prox is the pointwise projection onto L2 balls of radius
    ``alpha``.
    """
    if x.dim() < 3:
        raise ValueError("x must be shaped (B, C, *spatial_dims).")
    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype.")

    alpha_tensor = torch.as_tensor(alpha, device=x.device, dtype=x.dtype)
    if alpha_tensor.numel() != 1:
        raise ValueError("alpha must be a scalar.")
    alpha_value = float(alpha_tensor.item())
    if not torch.isfinite(alpha_tensor) or alpha_value < 0:
        raise ValueError("alpha must be finite and nonnegative.")

    x = x.clamp_min(epsilon)
    if n_iter == 0 or alpha_value == 0:
        return x.clone()

    sensitivity = sensitivity.to(device=x.device, dtype=x.dtype).clamp_min(epsilon)
    spatial_dims = x.dim() - 2

    # ||gradient||^2 <= 4 * spatial_dims. These symmetric step sizes therefore
    # satisfy tau * sigma * ||gradient||^2 < 1.
    primal_stepsize = 0.99 / sqrt(4 * spatial_dims)
    dual_stepsize = primal_stepsize

    primal = x.clone()
    primal_bar = primal.clone()
    dual = torch.zeros(
        (spatial_dims,) + x.shape,
        device=x.device,
        dtype=x.dtype,
    )

    for _ in range(n_iter):
        dual = dual + dual_stepsize * torch_gradient(primal_bar)
        dual_norm = torch_module(dual)
        dual_scale = torch.maximum(
            torch.ones_like(dual_norm),
            dual_norm / alpha_tensor,
        )
        dual = dual / dual_scale.unsqueeze(0)

        previous_primal = primal
        primal = _negative_entropy_data_prox(
            value=primal - primal_stepsize * _torch_gradient_adjoint(dual),
            x=x,
            sensitivity=sensitivity,
            stepsize=primal_stepsize,
            epsilon=epsilon,
        )
        primal_bar = 2 * primal - previous_primal

    return primal


def TV_dual_denoising(
    x: torch.Tensor,
    sensitivity: torch.Tensor,
    alpha: float,
    n_iter: int = 50,
    epsilon: float = 1e-8,
    fista: bool = False,
):
    """
    Compute dual denoising, from Maxim2018. Used for Poisson noise

    Parameters
    ----------
    x : np.array
        image to denoise
    sensitivity : np.array
        sensitivity
    alpha : float
            TV weight parameter
    n_iter : int
             Number of iteration
    epsilon : float
              Minimum value of the returned image
    fista : bool
            Whether to use heuristic FISTA acceleration or not
    Returns
    -------
    f : np.array
        denoised image
    """

    # Determine spatial dimensions (exclude batch and channel dimensions)
    spatial_dims = x.dim() - 2

    # crop sensitivity to get rid of zero value
    sensitivity[sensitivity < 0.1] = 0.1
    if spatial_dims == 2:
        den = sensitivity - 4 * alpha
        den[den < 0] = torch.min(sensitivity)
        # Lh = 8 * alpha**2 * sensitivity * x / den**2
        Lh = 8 * alpha**2 * x / den**2

    elif spatial_dims == 3:
        den = sensitivity - 6 * alpha
        den[den < 0] = torch.min(sensitivity)
        Lh = 12 * alpha**2 * sensitivity * x / den**2

    tau = 0.5 * alpha * div_zer(torch.ones_like(Lh), Lh)

    # Dual variable phi is a vector field with one component per spatial dimension.
    phi = torch.zeros((spatial_dims,) + x.shape, device=x.device, dtype=torch.float32)
    phi_ = torch.zeros_like(phi)
    tTV = 1

    for k in range(n_iter):
        z = pos(div_zer(sensitivity * x, sensitivity + alpha * torch_divergence(phi)))
        phi__ = phi - tau * torch_gradient(z)
        norm_phi = torch_module(phi__)
        denom = torch.maximum(norm_phi, torch.ones_like(norm_phi))
        phi__ = phi__ / denom.unsqueeze(0)

        # FISTA
        if fista:
            tTV_ = (1 + sqrt(1 + 4 * tTV**2)) * 0.5
            phi = phi__ + (tTV - 1) / tTV_ * (phi__ - phi_)
            tTV = tTV_
            phi_ = phi__
        else:
            phi = phi__

    x = div_zer(x, 1 + alpha * torch_divergence(phi) / sensitivity)
    # Ensure we don't return negative values
    x[x < epsilon] = epsilon

    return x

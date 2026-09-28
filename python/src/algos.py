from collections.abc import Callable

import deepinv
import torch
from tqdm import tqdm
from .prox import (
    TVProxMethod,
    right_prox_L1_negative_entropy,
    right_prox_L2_negative_entropy,
    right_prox_TV_negative_entropy,
    torch_divergence,
    torch_gradient,
    torch_module,
)

##########################
# Unregularized algorithms
##########################


@torch.no_grad()
def mlem(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    iterate_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """
    MLEM (Maximum Likelihood Expectation Maximization) reconstruction algorithm.

    This function implements the MLEM algorithm for Poisson inverse problems.

    :param torch.Tensor y: The observed/measured image.
    :param torch.Tensor x_init: The initial estimate for reconstruction.
    :param float stepsize: Step size for the iterative update, typically in range [0, 1].
    :param deepinv.physics.LinearPhysics physics: The linear physics operator describing the forward model.
    :param int max_steps: Maximum number of iterations to perform.
    :param bool verbose: Whether to display a progress bar during execution. Default: True.
    :param bool keep_inter: Whether to keep and return intermediate reconstruction results. Default: False.
    :param float filter_epsilon: Small epsilon value for numerical stability to avoid division by zero. Default: 1e-20.

    :return: If keep_inter is False, returns the final reconstructed image. If keep_inter is True, returns a tuple containing the final reconstruction and a list of intermediate results.
    :rtype: torch.Tensor or tuple(torch.Tensor, list[torch.Tensor])
    """
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)
    # Safely get background attribute if present
    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="MU", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        if iterate_callback is not None:
            iterate_callback(step, recon, prediction)

        mu_update = (recon / s) * physics.A_adjoint(y / prediction)
        recon = (1 - stepsize) * recon + stepsize * mu_update

        if keep_inter:
            xs.append(recon.cpu().clone())

    if iterate_callback is not None:
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        iterate_callback(max_steps, recon, prediction)

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def nolips(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    iterate_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """
    NoLips reconstruction algorithm.

    This function implements the NoLips algorithm for Poisson inverse problems.

    :param torch.Tensor y: The observed/measured image.
    :param torch.Tensor x_init: The initial estimate for reconstruction.
    :param float stepsize: Step size for the iterative update, typically in range [0, 1].
    :param deepinv.physics.LinearPhysics physics: The linear physics operator describing the forward model.
    :param int max_steps: Maximum number of iterations to perform.
    :param bool verbose: Whether to display a progress bar during execution. Default: True.
    :param bool keep_inter: Whether to keep and return intermediate reconstruction results. Default: False.
    :param float filter_epsilon: Small epsilon value for numerical stability to avoid division by zero. Default: 1e-20.
    """
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)
    # Safely get background attribute if present
    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="NoLips", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        if iterate_callback is not None:
            iterate_callback(step, recon, prediction)

        grad = s - physics.A_adjoint(y / prediction)
        recon = recon / (1 + stepsize * recon * grad)

        if keep_inter:
            xs.append(recon.cpu().clone())

    if iterate_callback is not None:
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        iterate_callback(max_steps, recon, prediction)

    return (recon, xs) if keep_inter else recon


##########################
# Regularized algorithms
##########################


@torch.no_grad()
def mlem_L1(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="MU + L1", disable=not verbose):
        mlem = (recon / s) * physics.A_adjoint(
            y / (physics.A(recon).clamp(min=filter_epsilon) + b)
        )
        recon = (1 - stepsize) * recon + stepsize * mlem
        recon = right_prox_L1_negative_entropy(
            x=recon,
            weight=stepsize * reg_weight,
            sensitivity=s,
            filter_epsilon=filter_epsilon,
        )

        if keep_inter:
            xs.append(recon.cpu().clone())

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def mlem_L1_osl(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    r"""MLEM with L1 regularization using one-step-late (OSL) approximation.

    For nonnegative reconstructions (enforced via clamping), the L1 subgradient
    is 1, so the OSL denominator becomes $A^T\mathbf{1} + \lambda$.

    With `stepsize=1`, this reduces to the classical multiplicative OSL update.
    """
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone().clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="OSL + L1", disable=not verbose):
        backproj = physics.A_adjoint(
            y / (physics.A(recon).clamp(min=filter_epsilon) + b)
        )
        denom = (s + reg_weight).clamp(min=filter_epsilon)
        update = recon * backproj / denom

        recon = (1 - stepsize) * recon + stepsize * update
        recon = recon.clamp(min=filter_epsilon)

        if keep_inter:
            xs.append(recon.cpu().clone())

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def mlem_L2(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    iterate_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)
    for step in tqdm(range(max_steps), desc="MU + L2", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        if iterate_callback is not None:
            iterate_callback(step, recon, prediction)

        mlem = (recon / s) * physics.A_adjoint(y / prediction)
        recon = (1 - stepsize) * recon + stepsize * mlem
        recon = right_prox_L2_negative_entropy(
            x=recon,
            weight=stepsize * reg_weight,
            sensitivity=s,
            filter_epsilon=filter_epsilon,
        )

        if keep_inter:
            xs.append(recon.cpu().clone())

    if iterate_callback is not None:
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        iterate_callback(max_steps, recon, prediction)

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def mlem_L2_osl(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    iterate_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    r"""MLEM with L2 regularization using one-step-late (OSL) approximation.

    Uses the standard OSL denominator for $R(x)=\tfrac12\|x\|_2^2$:
    $A^T\mathbf{1} + \lambda\,x_k$.

    With `stepsize=1`, this reduces to the classical multiplicative OSL update.
    """
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone().clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="OSL + L2", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        if iterate_callback is not None:
            iterate_callback(step, recon, prediction)

        backproj = physics.A_adjoint(y / prediction)
        denom = (s + reg_weight * recon).clamp(min=filter_epsilon)
        update = recon * backproj / denom

        recon = (1 - stepsize) * recon + stepsize * update
        recon = recon.clamp(min=filter_epsilon)

        if keep_inter:
            xs.append(recon.cpu().clone())

    if iterate_callback is not None:
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        iterate_callback(max_steps, recon, prediction)

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def mlem_tv(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    niter_tv: int = 50,
    fista_tv: bool = False,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    iterate_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    tv_prox: TVProxMethod = "dual",
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """MLEM with a right Bregman TV proximal step.

    Set ``tv_prox="dual"`` for the specialized dual-TV solver or
    ``tv_prox="pdhg"`` for standard primal-dual hybrid gradient.
    """

    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="MU + TV", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        if iterate_callback is not None:
            iterate_callback(step, recon, prediction)

        mu_update = (recon / s) * physics.A_adjoint(y / prediction)
        recon = (1 - stepsize) * recon + stepsize * mu_update
        recon = right_prox_TV_negative_entropy(
            x=recon,
            weight=stepsize * reg_weight,
            sensitivity=s,
            filter_epsilon=filter_epsilon,
            n_iter=niter_tv,
            fista_tv=fista_tv,
            method=tv_prox,
        )

        if keep_inter:
            xs.append(recon.cpu().clone())

    if iterate_callback is not None:
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        iterate_callback(max_steps, recon, prediction)

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def mlem_tv_osl(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    iterate_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    r"""
    MLEM with TV regularization using one-step-late (OSL) approximation.

    Implements the classical RL/MLEM-TV-OSL update (for convolutional PSF with
    unit-sum, this matches the common formula with denominator
    $1 - \lambda\,\mathrm{div}(\nabla x / \|\nabla x\|)$).

    For general `physics`, the sensitivity image `s = A^T 1` appears in the
    denominator, yielding: $s - \lambda\,\mathrm{div}(\nabla x / \|\nabla x\|)$.
    """

    def tv_curvature(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute div(grad(x) / ||grad(x)||) over spatial dimensions."""
        g = torch_gradient(x)
        g_norm = torch_module(g).clamp(min=eps)
        g_unit = g / g_norm.unsqueeze(0)
        return torch_divergence(g_unit)

    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="OSL + TV", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        if iterate_callback is not None:
            iterate_callback(step, recon, prediction)

        tv_div = tv_curvature(recon, eps=1e-8)

        backproj = physics.A_adjoint(y / prediction)
        denom = (s - reg_weight * tv_div).clamp(min=filter_epsilon)
        update = recon * backproj / denom

        # Optional relaxation (stepsize=1 recovers the classical multiplicative update)
        recon = (1 - stepsize) * recon + stepsize * update
        recon = recon.clamp(min=filter_epsilon)

        if keep_inter:
            xs.append(recon.cpu().clone())

    if iterate_callback is not None:
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        iterate_callback(max_steps, recon, prediction)

    return (recon, xs) if keep_inter else recon


def nolips_L1(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="NoLips + L1", disable=not verbose):
        grad = s - physics.A_adjoint(
            y / (physics.A(recon).clamp(min=filter_epsilon) + b)
        )
        recon = recon / (1 + stepsize * (reg_weight * recon + recon * grad)).clamp(
            min=filter_epsilon
        )

        if keep_inter:
            xs.append(recon.cpu().clone())

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def nolips_L2(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    iterate_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="NoLips + L2", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        if iterate_callback is not None:
            iterate_callback(step, recon, prediction)

        grad = s - physics.A_adjoint(y / prediction)
        recon = (
            torch.sqrt(
                (1 + stepsize * recon * grad) ** 2
                + 4 * stepsize * reg_weight * recon**2
            )
            - (1 + stepsize * recon * grad)
        ) / (2 * stepsize * reg_weight * recon).clamp(min=filter_epsilon)

        if keep_inter:
            xs.append(recon.cpu().clone())

    if iterate_callback is not None:
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        iterate_callback(max_steps, recon, prediction)

    return (recon, xs) if keep_inter else recon

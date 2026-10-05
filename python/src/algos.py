"""Poisson reconstruction methods.

MU, mirror descent, and the L2/TV methods return ``(reconstruction, metrics)``.
The metrics dictionary contains objective, NRMSE (percent), and relative
progress at initialization and after every update. NRMSE is NaN without
``x_ref``. With ``keep_inter=True``, it also contains CPU ``iterates``.
"""

import deepinv
import torch
from tqdm import tqdm
from .prox import (
    rprox_negentropy_l1,
    rprox_negentropy_l2,
    rprox_negentropy_TV,
)


def poisson_objective(prediction: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Poisson negative log-likelihood with double-precision reduction."""
    prediction_double = prediction.to(torch.float64)
    return (prediction_double - y.to(torch.float64) * prediction_double.log()).sum()


@torch.no_grad()
def record_metrics(
    metrics, recon, previous, prediction, y, x_ref=None, l2_weight=0.0, tv_weight=0.0
):
    """Append objective, NRMSE (%), and relative progress to ``metrics``.

    ``previous=None`` denotes the initial iterate. Without ``x_ref``, NRMSE
    is NaN. Return a detached copy to use as ``previous`` at the next call.
    """
    objective = poisson_objective(prediction, y)
    if l2_weight:
        objective += (l2_weight / 2) * recon.to(torch.float64).square().sum()
    if tv_weight:
        objective += tv_weight * deepinv.optim.TVPrior().fn(recon).sum()
    metrics["objective"].append(objective.item())
    metrics["nrmse"].append(
        100 * deepinv.metric.NMSE(reduction="mean")(recon, x_ref).sqrt().item()
        if x_ref is not None
        else float("nan")
    )
    progress = (
        float("nan")
        if previous is None
        else max(
            (
                torch.linalg.norm(recon - previous)
                / torch.linalg.norm(previous).clamp_min(1e-16)
            ).item(),
            1e-16,
        )
    )
    metrics["relative_progress"].append(progress)
    return recon.detach().clone()


##########################
# Unregularized algorithms
##########################


@torch.no_grad()
def mu(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    x_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, list[float] | list[torch.Tensor]]]:
    """
    MLEM (Maximum Likelihood Expectation Maximization) reconstruction algorithm.

    This function implements the MLEM algorithm for Poisson inverse problems.

    :param torch.Tensor y: The observed/measured image.
    :param torch.Tensor x_init: The initial estimate for reconstruction.
    :param float stepsize: Step size for the iterative update, typically in range [0, 1].
    :param deepinv.physics.LinearPhysics physics: The linear physics operator describing the forward model.
    :param int max_steps: Maximum number of iterations to perform.
    :param bool verbose: Whether to display a progress bar during execution. Default: True.
    :param bool keep_inter: Include CPU intermediate images in metrics["iterates"].
    :param float filter_epsilon: Small epsilon value for numerical stability to avoid division by zero. Default: 1e-20.

    :return: The reconstruction and a dictionary of metric histories.
    """
    metrics = {"objective": [], "nrmse": [], "relative_progress": []}
    previous = None
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
        previous = record_metrics(metrics, recon, previous, prediction, y, x_ref)

        mu_update = (recon / s) * physics.A_adjoint(y / prediction)
        recon = (1 - stepsize) * recon + stepsize * mu_update

        if keep_inter:
            xs.append(recon.cpu().clone())

    prediction = physics.A(recon).clamp(min=filter_epsilon) + b
    record_metrics(metrics, recon, previous, prediction, y, x_ref)
    if keep_inter:
        metrics["iterates"] = xs
    return recon, metrics


@torch.no_grad()
def mirror_descent(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    x_ref: torch.Tensor | None = None,
    *,
    armijo_constant: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 50,
) -> tuple[torch.Tensor, dict[str, list[float] | list[torch.Tensor]]]:
    """
    Mirror descent reconstruction with Burg's entropy.

    Each iteration starts from ``stepsize`` and shrinks it until the Burg
    update is positive and the Poisson objective satisfies Armijo decrease:
    f(trial) <= f(recon) + armijo_constant * <grad f(recon), trial - recon>.
    The objective includes any additive background from the physics operator.

    :param torch.Tensor y: The observed/measured image.
    :param torch.Tensor x_init: The initial estimate for reconstruction.
    :param float stepsize: Initial trial step size for each backtracking search; must be positive.
    :param deepinv.physics.LinearPhysics physics: The linear physics operator describing the forward model.
    :param int max_steps: Maximum number of iterations to perform.
    :param bool verbose: Whether to display a progress bar during execution. Default: True.
    :param bool keep_inter: Include CPU intermediate images in metrics["iterates"].
    :param float filter_epsilon: Small epsilon value for numerical stability to avoid division by zero. Default: 1e-20.
    :param float armijo_constant: Sufficient-decrease coefficient in (0, 1).
    :param float backtracking_factor: Step reduction factor in (0, 1).
    :param int max_backtracks: Maximum number of trial steps per iteration.
    :raises RuntimeError: If no admissible decreasing trial step is found.
    """
    if not 0 < stepsize < float("inf"):
        raise ValueError("stepsize must be finite and positive.")
    if not 0 < armijo_constant < 1 or not 0 < backtracking_factor < 1:
        raise ValueError("Armijo constant and backtracking factor must lie in (0, 1).")
    if max_backtracks < 1:
        raise ValueError("max_backtracks must be positive.")

    metrics = {"objective": [], "nrmse": [], "relative_progress": []}
    previous = None
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)
    # Safely get background attribute if present
    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="Mirror descent", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        previous = record_metrics(metrics, recon, previous, prediction, y, x_ref)

        grad = s - physics.A_adjoint(y / prediction)
        current_objective = poisson_objective(prediction, y)
        trial_stepsize = stepsize
        for _ in range(max_backtracks):
            denominator = 1 + trial_stepsize * recon * grad
            if torch.isfinite(denominator).all() and (denominator > 0).all():
                trial = recon / denominator
                if torch.isfinite(trial).all() and (trial > 0).all():
                    trial_prediction = physics.A(trial).clamp(min=filter_epsilon) + b
                    trial_objective = poisson_objective(trial_prediction, y)
                    directional_change = (
                        grad.to(torch.float64) * (trial - recon).to(torch.float64)
                    ).sum()
                    if torch.isfinite(trial_objective) and trial_objective <= (
                        current_objective + armijo_constant * directional_change
                    ):
                        recon = trial
                        break
            trial_stepsize *= backtracking_factor
        else:
            raise RuntimeError(
                f"Mirror descent backtracking failed at iteration {step}."
            )

        if keep_inter:
            xs.append(recon.cpu().clone())

    prediction = physics.A(recon).clamp(min=filter_epsilon) + b
    record_metrics(metrics, recon, previous, prediction, y, x_ref)
    if keep_inter:
        metrics["iterates"] = xs
    return recon, metrics


##########################
# Regularized algorithms
##########################


@torch.no_grad()
def rbpg_l1(
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

    for step in tqdm(range(max_steps), desc="R-BPG + L1", disable=not verbose):
        mu_update = (recon / s) * physics.A_adjoint(
            y / (physics.A(recon).clamp(min=filter_epsilon) + b)
        )
        recon = (1 - stepsize) * recon + stepsize * mu_update
        recon = rprox_negentropy_l1(
            x=recon,
            weight=stepsize * reg_weight,
            sensitivity=s,
            filter_epsilon=filter_epsilon,
        )

        if keep_inter:
            xs.append(recon.cpu().clone())

    return (recon, xs) if keep_inter else recon


@torch.no_grad()
def osl_l1(
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
def rbpg_l2(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    x_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, list[float] | list[torch.Tensor]]]:
    metrics = {"objective": [], "nrmse": [], "relative_progress": []}
    previous = None
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)
    for step in tqdm(range(max_steps), desc="R-BPG + L2", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        previous = record_metrics(
            metrics, recon, previous, prediction, y, x_ref, l2_weight=reg_weight
        )

        mu_update = (recon / s) * physics.A_adjoint(y / prediction)
        recon = (1 - stepsize) * recon + stepsize * mu_update
        recon = rprox_negentropy_l2(
            x=recon,
            weight=stepsize * reg_weight,
            sensitivity=s,
            filter_epsilon=filter_epsilon,
        )

        if keep_inter:
            xs.append(recon.cpu().clone())

    prediction = physics.A(recon).clamp(min=filter_epsilon) + b
    record_metrics(metrics, recon, previous, prediction, y, x_ref, l2_weight=reg_weight)
    if keep_inter:
        metrics["iterates"] = xs
    return recon, metrics


@torch.no_grad()
def osl_l2(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    x_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, list[float] | list[torch.Tensor]]]:
    r"""MLEM with L2 regularization using one-step-late (OSL) approximation.

    Uses the standard OSL denominator for $R(x)=\tfrac12\|x\|_2^2$:
    $A^T\mathbf{1} + \lambda\,x_k$.

    With `stepsize=1`, this reduces to the classical multiplicative OSL update.
    """
    metrics = {"objective": [], "nrmse": [], "relative_progress": []}
    previous = None
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone().clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="OSL + L2", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        previous = record_metrics(
            metrics, recon, previous, prediction, y, x_ref, l2_weight=reg_weight
        )

        backproj = physics.A_adjoint(y / prediction)
        denom = (s + reg_weight * recon).clamp(min=filter_epsilon)
        update = recon * backproj / denom

        recon = (1 - stepsize) * recon + stepsize * update
        recon = recon.clamp(min=filter_epsilon)

        if keep_inter:
            xs.append(recon.cpu().clone())

    prediction = physics.A(recon).clamp(min=filter_epsilon) + b
    record_metrics(metrics, recon, previous, prediction, y, x_ref, l2_weight=reg_weight)
    if keep_inter:
        metrics["iterates"] = xs
    return recon, metrics


@torch.no_grad()
def rbpg_tv(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    niter_tv: int = 50,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    x_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, list[float] | list[torch.Tensor]]]:
    """MLEM with a right Bregman TV proximal step computed with PDHG."""

    metrics = {"objective": [], "nrmse": [], "relative_progress": []}
    previous = None
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="R-BPG + TV", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        previous = record_metrics(
            metrics, recon, previous, prediction, y, x_ref, tv_weight=reg_weight
        )

        mu_update = (recon / s) * physics.A_adjoint(y / prediction)
        recon = (1 - stepsize) * recon + stepsize * mu_update
        recon = rprox_negentropy_TV(
            x=recon,
            weight=stepsize * reg_weight,
            sensitivity=s,
            filter_epsilon=filter_epsilon,
            n_iter=niter_tv,
        )

        if keep_inter:
            xs.append(recon.cpu().clone())

    prediction = physics.A(recon).clamp(min=filter_epsilon) + b
    record_metrics(metrics, recon, previous, prediction, y, x_ref, tv_weight=reg_weight)
    if keep_inter:
        metrics["iterates"] = xs
    return recon, metrics


@torch.no_grad()
def osl_tv(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    x_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, list[float] | list[torch.Tensor]]]:
    r"""
    MLEM with TV regularization using one-step-late (OSL) approximation.

    Implements the classical RL/MLEM-TV-OSL update (for convolutional PSF with
    unit-sum, this matches the common formula with denominator
    $1 - \lambda\,\mathrm{div}(\nabla x / \|\nabla x\|)$).

    For general `physics`, the sensitivity image `s = A^T 1` appears in the
    denominator, yielding: $s - \lambda\,\mathrm{div}(\nabla x / \|\nabla x\|)$.
    """

    tv_prior = deepinv.optim.TVPrior()
    metrics = {"objective": [], "nrmse": [], "relative_progress": []}
    previous = None
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
        previous = record_metrics(
            metrics, recon, previous, prediction, y, x_ref, tv_weight=reg_weight
        )

        tv_subgradient = tv_prior.grad(recon)

        backproj = physics.A_adjoint(y / prediction)
        denom = (s + reg_weight * tv_subgradient).clamp(min=filter_epsilon)
        update = recon * backproj / denom

        # Optional relaxation (stepsize=1 recovers the classical multiplicative update)
        recon = (1 - stepsize) * recon + stepsize * update
        recon = recon.clamp(min=filter_epsilon)

        if keep_inter:
            xs.append(recon.cpu().clone())

    prediction = physics.A(recon).clamp(min=filter_epsilon) + b
    record_metrics(metrics, recon, previous, prediction, y, x_ref, tv_weight=reg_weight)
    if keep_inter:
        metrics["iterates"] = xs
    return recon, metrics


@torch.no_grad()
def bpg_l2(
    y: torch.Tensor,
    x_init: torch.Tensor,
    stepsize: float,
    physics: deepinv.physics.LinearPhysics,
    reg_weight: float,
    max_steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    x_ref: torch.Tensor | None = None,
    *,
    armijo_constant: float = 1e-4,
    backtracking_factor: float = 0.5,
    max_backtracks: int = 50,
) -> tuple[torch.Tensor, dict[str, list[float] | list[torch.Tensor]]]:
    """Burg Bregman proximal gradient with L2 composite Armijo backtracking.

    The objective is F(x) = f(x) + g(x), where f is the Poisson negative
    log-likelihood including any additive background and
    g(x) = reg_weight / 2 * ||x||^2. Each iteration starts from ``stepsize``
    and recomputes the exact Bregman proximal candidate after every shrink.
    A finite, strictly positive candidate is accepted when

        F(trial) <= F(recon) + armijo_constant * (
            <grad f(recon), trial - recon> + g(trial) - g(recon)
        ).

    ``stepsize`` must be finite and positive, ``reg_weight`` finite and
    nonnegative, and both line-search factors must lie in (0, 1).
    ``max_backtracks`` bounds the number of trial steps per iteration.
    With zero regularization this reduces to Burg mirror descent.
    Raises RuntimeError if no admissible decreasing trial step is found.
    """

    def regularizer(x):
        return (reg_weight / 2) * x.to(torch.float64).square().sum()

    metrics = {"objective": [], "nrmse": [], "relative_progress": []}
    previous = None
    xs = [x_init.cpu().clone()] if keep_inter else None

    recon = x_init.clone()
    recon = recon.clamp(min=filter_epsilon)
    s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

    if hasattr(physics, "background"):
        b = physics.background.clamp(min=filter_epsilon)
    else:
        b = torch.zeros_like(y)

    for step in tqdm(range(max_steps), desc="BPG + L2", disable=not verbose):
        prediction = physics.A(recon).clamp(min=filter_epsilon) + b
        previous = record_metrics(
            metrics, recon, previous, prediction, y, x_ref, l2_weight=reg_weight
        )

        grad = s - physics.A_adjoint(y / prediction)
        current_regularizer = regularizer(recon)
        current_objective = poisson_objective(prediction, y) + current_regularizer
        trial_stepsize = stepsize
        for _ in range(max_backtracks):
            denominator = 1 + trial_stepsize * recon * grad
            if not torch.isfinite(denominator).all():
                trial_stepsize *= backtracking_factor
                continue

            if reg_weight == 0:
                if not (denominator > 0).all():
                    trial_stepsize *= backtracking_factor
                    continue
                trial = recon / denominator
            else:
                root = torch.hypot(
                    denominator, 2 * (trial_stepsize * reg_weight) ** 0.5 * recon
                )
                trial = torch.where(
                    denominator >= 0,
                    2 * recon / (denominator + root),
                    (root - denominator) / (2 * trial_stepsize * reg_weight * recon),
                )

            if torch.isfinite(trial).all() and (trial > 0).all():
                trial_prediction = physics.A(trial).clamp(min=filter_epsilon) + b
                trial_objective = poisson_objective(trial_prediction, y) + regularizer(trial)
                displacement = trial.to(torch.float64) - recon.to(torch.float64)
                directional_change = (grad.to(torch.float64) * displacement).sum()
                regularizer_change = (reg_weight / 2) * (
                    displacement * (trial.to(torch.float64) + recon.to(torch.float64))
                ).sum()
                model_change = directional_change + regularizer_change
                if torch.isfinite(trial_objective) and trial_objective <= (
                    current_objective + armijo_constant * model_change
                ):
                    recon = trial
                    break
            trial_stepsize *= backtracking_factor
        else:
            raise RuntimeError(f"BPG + L2 backtracking failed at iteration {step}.")

        if keep_inter:
            xs.append(recon.cpu().clone())

    prediction = physics.A(recon).clamp(min=filter_epsilon) + b
    record_metrics(metrics, recon, previous, prediction, y, x_ref, l2_weight=reg_weight)
    if keep_inter:
        metrics["iterates"] = xs
    return recon, metrics

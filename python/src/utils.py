import numpy as np
import torch
import deepinv
import matplotlib.pyplot as plt

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import deepinv as dinv


def plot_residual(
    references: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    iterates_by_method: Mapping[str, Sequence[torch.Tensor]],
    device: torch.device | str | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "Relative residual",
    yscale: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    label_fontsize: int = 28,
    legend_fontsize: int = 20,
    linestyles: Mapping[str, str] | Sequence[str] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    show: bool = True,
    return_fig_ax: bool = False,
) -> tuple[Figure, Axes]:
    """Compute and plot the relative L2 residual vs iteration for multiple methods.

    The residual is computed as ||x_k - x_ref||_2 / ||x_ref||_2 for each iterate.

    Args:
        references: Reference tensor for each method, either as a mapping matching
            `iterates_by_method` labels or as a sequence in the same order.
        iterates_by_method: Mapping `label -> [x_0, x_1, ...]`.
        device: If provided, moves references and iterates to this device before
            computing residuals.

    Returns:
        The Matplotlib `(fig, ax)` objects for the created plot.
    """

    def to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    residual_values: dict[str, list[float]] = {}
    for idx, (label, xs) in enumerate(iterates_by_method.items()):
        if isinstance(references, Mapping):
            reference = references[label]
        else:
            reference = references[idx]

        reference_ = reference.to(device) if device is not None else reference
        reference_norm = torch.linalg.norm(reference_)

        values: list[float] = []
        for x in xs:
            x_ = x.to(device) if device is not None else x
            residual = torch.linalg.norm(x_ - reference_)
            residual = (
                residual if float(reference_norm) == 0.0 else residual / reference_norm
            )
            values.append(to_float(residual))
        residual_values[label] = values

    fig, ax = plt.subplots(figsize=figsize)

    default_styles = ["-", "--", "-.", ":"]
    for idx, (label, values) in enumerate(residual_values.items()):
        if isinstance(linestyles, Mapping):
            linestyle = linestyles.get(label, default_styles[idx % len(default_styles)])
        elif isinstance(linestyles, Sequence) and not isinstance(
            linestyles, (str, bytes)
        ):
            linestyle = (
                linestyles[idx % len(linestyles)]
                if len(linestyles) > 0
                else default_styles[idx % len(default_styles)]
            )
        else:
            linestyle = default_styles[idx % len(default_styles)]

        kwargs = dict(plot_kwargs or {})
        kwargs["linestyle"] = linestyle
        ax.plot(values, label=label, **kwargs)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend(fontsize=legend_fontsize)

    # Prevent clipping of axis labels (common with large fonts / seaborn themes).
    fig.tight_layout()
    if show:
        plt.show()

    if return_fig_ax:
        return fig, ax


def plot_metric(
    reference: torch.Tensor | None,
    iterates_by_method: Mapping[str, Sequence[torch.Tensor]],
    metric: deepinv.metric.Metric,
    *,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    device: torch.device | str | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "",
    yscale: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    label_fontsize: int = 28,
    legend_fontsize: int = 20,
    linestyles: Mapping[str, str] | Sequence[str] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    show: bool = True,
    return_fig_ax: bool = False,
) -> tuple[Figure, Axes]:
    """Compute and plot a metric value vs iteration for multiple methods.

    This is meant for things like PSNR/SSIM vs iteration, or data-fidelity terms
    vs iteration, even when each method has a different number of iterates.

    Metric calling convention:
    - If `reference` is not None, we first try `metric(reference, x)`.
    - If that raises TypeError (e.g., metric only accepts one arg), we fall back to
      `metric(x)`.
    - If `reference` is None, we call `metric(x)`.

    Args:
            reference: Reference tensor (e.g., `x_true` for PSNR). Can be None.
            iterates_by_method: Mapping `label -> [x_0, x_1, ...]`.
            metric: Callable metric (deepinv metrics work here).
            transform: Optional transform applied to each iterate before metric evaluation
                    (e.g., `lambda x: physics.A(x)` for KL on measurements).
            device: If provided, moves `reference` and iterates to this device before
                    applying `transform` and `metric`.

        Returns:
            The Matplotlib `(fig, ax)` objects for the created plot.
    """

    def to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    reference_ = reference
    if reference_ is not None and device is not None:
        reference_ = reference_.to(device)

    metric_values: dict[str, list[float]] = {}
    for label, xs in iterates_by_method.items():
        values: list[float] = []
        for x in xs:
            x_ = x.to(device) if device is not None else x
            x_eval = transform(x_) if transform is not None else x_

            if reference_ is None:
                metric_value = metric(x_eval)
            else:
                try:
                    metric_value = metric(reference_, x_eval)
                except TypeError:
                    metric_value = metric(x_eval)

            values.append(to_float(metric_value))
        metric_values[label] = values

    fig, ax = plt.subplots(figsize=figsize)

    default_styles = ["-", "--", "-.", ":"]
    for idx, (label, values) in enumerate(metric_values.items()):
        if isinstance(linestyles, Mapping):
            linestyle = linestyles.get(label, default_styles[idx % len(default_styles)])
        elif isinstance(linestyles, Sequence) and not isinstance(
            linestyles, (str, bytes)
        ):
            linestyle = (
                linestyles[idx % len(linestyles)]
                if len(linestyles) > 0
                else default_styles[idx % len(default_styles)]
            )
        else:
            linestyle = default_styles[idx % len(default_styles)]

        kwargs = dict(plot_kwargs or {})
        kwargs["linestyle"] = linestyle
        ax.plot(values, label=label, **kwargs)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend(fontsize=legend_fontsize)

    # Prevent clipping of axis labels (common with large fonts / seaborn themes).
    fig.tight_layout()
    if show:
        plt.show()

    if return_fig_ax:
        return fig, ax


def plot_data_fidelity(
    y: torch.Tensor,
    iterates_by_method: Mapping[str, Sequence[torch.Tensor]],
    data_fidelity: Callable[[torch.Tensor, torch.Tensor], Any],
    *,
    physics: deepinv.physics.LinearPhysics,
    device: torch.device | str | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "",
    yscale: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    label_fontsize: int = 28,
    legend_fontsize: int = 20,
    linestyles: Mapping[str, str] | Sequence[str] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    show: bool = True,
    return_fig_ax: bool = False,
) -> tuple[Figure, Axes]:
    """Compute and plot a data-fidelity value vs iteration for multiple methods.

    Typical use: plot KL (or NLL) between measured data `y` and predicted data
    `forward(x_k)` along the iterates `x_k`.

    Note: `data_fidelity` is called as `data_fidelity(y_pred, y_ref)` where
    `y_pred = forward(x_k)` and `y_ref = y`.

    Args:
        y: Reference measurement tensor.
        iterates_by_method: Mapping `label -> [x_0, x_1, ...]`.
        data_fidelity: Callable with signature `(y_pred, y_ref) -> scalar`.
            If your function expects `(y_ref, y_pred)`, pass a lambda that swaps args.
        forward: Forward operator mapping an iterate `x` to predicted measurements.
        device: If provided, moves `y` and iterates to this device before evaluation.

    Returns:
        The Matplotlib `(fig, ax)` objects for the created plot.
    """

    def to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    y_ref = y.to(device) if device is not None else y

    values_by_method: dict[str, list[float]] = {}
    for label, xs in iterates_by_method.items():
        values: list[float] = []
        for x in xs:
            x_ = x.to(device) if device is not None else x
            values.append(to_float(data_fidelity(x_, y_ref, physics)))
        values_by_method[label] = values

    fig, ax = plt.subplots(figsize=figsize)

    default_styles = ["-", "--", "-.", ":"]
    for idx, (label, values) in enumerate(values_by_method.items()):
        if isinstance(linestyles, Mapping):
            linestyle = linestyles.get(label, default_styles[idx % len(default_styles)])
        elif isinstance(linestyles, Sequence) and not isinstance(
            linestyles, (str, bytes)
        ):
            linestyle = (
                linestyles[idx % len(linestyles)]
                if len(linestyles) > 0
                else default_styles[idx % len(default_styles)]
            )
        else:
            linestyle = default_styles[idx % len(default_styles)]

        kwargs = dict(plot_kwargs or {})
        kwargs["linestyle"] = linestyle
        ax.plot(values, label=label, **kwargs)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend(fontsize=legend_fontsize)

    # Prevent clipping of axis labels (common with large fonts / seaborn themes).
    fig.tight_layout()
    if show:
        plt.show()

    if return_fig_ax:
        return fig, ax


class KL_L2(dinv.optim.DataFidelity):
    def __init__(self, gain: float, reg_weight: float):
        super().__init__()
        self.gain = gain
        self.reg_weight = reg_weight
        self.kl = dinv.optim.data_fidelity.PoissonLikelihood(
            gain=gain, bkg=1e-20, denormalize=False
        )

    def fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: dinv.physics.Physics,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{x}{y}`.
        """
        x = x.to(physics.device)
        return (
            self.kl(x, y, physics) + (self.reg_weight / 2) * torch.linalg.norm(x) ** 2
        )


class KL_TV(dinv.optim.DataFidelity):
    def __init__(self, gain: float, reg_weight: float):
        super().__init__()
        self.gain = gain
        self.reg_weight = reg_weight
        self.kl = dinv.optim.data_fidelity.PoissonLikelihood(
            gain=gain, bkg=1e-20, denormalize=False
        )

    def fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: dinv.physics.Physics,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{x}{y}`.
        """
        x = x.to(physics.device)
        return self.kl(x, y, physics) + self.reg_weight * dinv.optim.prior.TVPrior().fn(
            x
        )

# %%
# ruff: noqa: E402
import sys
from pathlib import Path

cwd = Path.cwd().resolve()
project_root = next(
    path for path in (cwd, *cwd.parents) if (path / "latex" / "main.tex").is_file()
)
sys.path.insert(0, str(project_root / "python"))
article_figure_directory = project_root / "latex" / "figures"
article_figure_directory.mkdir(parents=True, exist_ok=True)

import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from deepinv.physics.functional import gaussian_blur
from tqdm.auto import tqdm

from src.algos import mlem_tv
from src.prox import torch_divergence, torch_gradient, torch_module
from src.utils import KL_TV

sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
torch.manual_seed(0)
device = "cuda"


# %%
image_size = 256
gain = 1 / 10
max_iter = 100
niter_tv = 100
eps = 1e-10
reg_weights = np.geomspace(1e-4, 1, 50)

x = dinv.utils.load_example(
    "butterfly.png",
    img_size=image_size,
    grayscale=True,
    resize_mode="resize",
    device=device,
)
physics = dinv.physics.Blur(
    filter=gaussian_blur(sigma=(1.6, 1.6), device=device),
    padding="circular",
    noise_model=dinv.physics.PoissonNoise(gain=gain),
).to(device)
y = physics(x)
x_init = torch.ones_like(y)
s = physics.A_adjoint(torch.ones_like(y))

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
for ax, image, title in zip(axes, [x, y], ["Image", "Observation"], strict=True):
    ax.imshow(image.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()


# %%
@torch.no_grad()
def run_osl(reg_weight, keep_inter=False):
    """Run TV-OSL without clamping its regularization denominator."""
    recon = x_init.clone()
    iterates = [recon.cpu().clone()] if keep_inter else None

    for _ in range(max_iter):
        gradient = torch_gradient(recon)
        curvature = torch_divergence(
            gradient / torch_module(gradient).clamp_min(eps).unsqueeze(0)
        )
        denominator = s - reg_weight * curvature

        if (denominator <= 0).any():
            return (None, iterates) if keep_inter else None

        prediction = physics.A(recon).clamp_min(eps)
        recon = recon * physics.A_adjoint(y / prediction) / denominator

        if keep_inter:
            iterates.append(recon.cpu().clone())

    reconstruction = recon if torch.isfinite(recon).all() else None
    return (reconstruction, iterates) if keep_inter else reconstruction


@torch.no_grad()
def run_proposed(reg_weight):
    recon = mlem_tv(
        y=y,
        x_init=x_init,
        stepsize=1,
        physics=physics,
        reg_weight=reg_weight,
        max_steps=max_iter,
        niter_tv=niter_tv,
        tv_prox="pdhg",
        verbose=False,
        filter_epsilon=eps,
    )
    return recon if torch.isfinite(recon).all() and (recon >= 0).all() else None


# %%
# Check convergence once for both methods before running the weight sweep.
convergence_reg_weight = 0.25
mu_reconstruction, mu_iterates = mlem_tv(
    y=y,
    x_init=x_init,
    stepsize=1,
    physics=physics,
    reg_weight=convergence_reg_weight,
    max_steps=max_iter,
    niter_tv=niter_tv,
    tv_prox="pdhg",
    verbose=True,
    keep_inter=True,
    filter_epsilon=eps,
)
osl_reconstruction, osl_iterates = run_osl(
    convergence_reg_weight,
    keep_inter=True,
)
if osl_reconstruction is None:
    raise RuntimeError("OSL-TV became unstable during the convergence check.")

convergence_method_names = ["R-BPG + TV", "OSL + TV"]
convergence_reconstructions = [mu_reconstruction, osl_reconstruction]
convergence_iterates = [mu_iterates, osl_iterates]
convergence_objective = KL_TV(
    gain=gain,
    reg_weight=convergence_reg_weight,
)
convergence_psnr = dinv.metric.PSNR()
with torch.no_grad():
    convergence_objective_values = [
        [
            convergence_objective(iterate.to(device), y, physics).item()
            for iterate in method_iterates
        ]
        for method_iterates in convergence_iterates
    ]
    convergence_psnr_values = [
        [convergence_psnr(iterate.to(device), x).item() for iterate in method_iterates]
        for method_iterates in convergence_iterates
    ]

convergence_reconstruction_figure, axes = plt.subplots(1, 2, figsize=(6, 3))
for ax, name, reconstruction, values in zip(
    axes,
    convergence_method_names,
    convergence_reconstructions,
    convergence_psnr_values,
    strict=True,
):
    ax.imshow(reconstruction.squeeze().cpu(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"{name}\n"
        rf"$\lambda={convergence_reg_weight:g}$, "
        f"PSNR={values[-1]:.2f} dB"
    )
    ax.axis("off")
convergence_reconstruction_figure.tight_layout()

convergence_metrics_figure, (objective_ax, psnr_history_ax) = plt.subplots(
    1, 2, figsize=(12, 4)
)
for name, color, objective_values, psnr_history in zip(
    convergence_method_names,
    sns.color_palette("colorblind")[:2],
    convergence_objective_values,
    convergence_psnr_values,
    strict=True,
):
    objective_ax.plot(
        range(10, len(objective_values)),
        objective_values[10:],
        color=color,
        label=name,
    )
    psnr_history_ax.plot(psnr_history, color=color, label=name)

objective_ax.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\lambda\,\mathrm{TV}(x)$",
    title=rf"TV objective ($\lambda={convergence_reg_weight:g}$)",
)
objective_ax.set_yscale("log")
objective_ax.legend()
psnr_history_ax.set(
    xlabel="Iteration",
    ylabel="PSNR (dB)",
    title="Reconstruction quality",
)
psnr_history_ax.legend()
convergence_metrics_figure.tight_layout()


# %%
reconstructions = [
    [run_proposed(weight) for weight in tqdm(reg_weights, desc="R-BPG + TV sweep")],
    [run_osl(weight) for weight in tqdm(reg_weights, desc="OSL + TV sweep")],
]
psnr = dinv.metric.PSNR()
psnr_values = np.array(
    [
        [
            psnr(reconstruction, x).item() if reconstruction is not None else np.nan
            for reconstruction in method_reconstructions
        ]
        for method_reconstructions in reconstructions
    ]
)

method_names = ["R-BPG + TV", "OSL + TV"]

# %%
reconstruction_reg_weights = [0.18, 0.07]
selected_reconstructions = [
    run_proposed(reconstruction_reg_weights[0]),
    run_osl(reconstruction_reg_weights[1]),
]
if any(reconstruction is None for reconstruction in selected_reconstructions):
    raise RuntimeError(
        "A reconstruction failed at its predefined regularization weight."
    )
selected_psnr_values = [
    psnr(reconstruction, x).item() for reconstruction in selected_reconstructions
]

sweep_reconstruction_figure, axes = plt.subplots(1, 2, figsize=(6, 3))
for ax, name, reconstruction, reg_weight, psnr_value in zip(
    axes,
    method_names,
    selected_reconstructions,
    reconstruction_reg_weights,
    selected_psnr_values,
    strict=True,
):
    ax.imshow(
        reconstruction.squeeze().cpu(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        f"{name}\n" rf"$\lambda={reg_weight:.2f}$, " f"PSNR={psnr_value:.2f} dB"
    )
    ax.axis("off")
sweep_reconstruction_figure.tight_layout()
sweep_reconstruction_figure.savefig(
    article_figure_directory / "deconvolution_tv_stability_reconstructions.pdf",
    dpi=300,
    pad_inches=0,
    bbox_inches="tight",
    transparent=True,
)


# %%
sweep_psnr_figure, psnr_ax = plt.subplots(figsize=(8, 3.5))
for name, color, values in zip(
    method_names,
    sns.color_palette("colorblind")[:2],
    psnr_values,
    strict=True,
):
    psnr_ax.semilogx(
        reg_weights,
        values,
        marker="o",
        markersize=4,
        color=color,
        label=name,
    )
psnr_ax.set_xlabel(r"Regularization weight $\lambda$")
psnr_ax.set_ylabel("PSNR (dB)")
psnr_ax.legend()
sweep_psnr_figure.tight_layout()
sweep_psnr_figure.savefig(
    article_figure_directory / "deconvolution_tv_stability_metrics.pdf",
    dpi=300,
    pad_inches=0,
    bbox_inches="tight",
    transparent=True,
)

# %%

# %%
import json
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

from src.algos import rbpg_tv

sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    filter=gaussian_blur(sigma=(2.0, 2.0), device=device),
    padding="circular",
    noise_model=dinv.physics.PoissonNoise(gain=gain),
).to(device)
y = physics(x)
x_init = torch.ones_like(y)
s = physics.A_adjoint(torch.ones_like(y))
tv_prior = dinv.optim.TVPrior()


@torch.no_grad()
def run_osl(reg_weight):
    """Return a reconstruction or the reason and iteration of an invalid update."""
    recon = x_init.clone()
    for iteration in range(max_iter):
        denominator = s + reg_weight * tv_prior.grad(recon)
        if not torch.isfinite(denominator).all():
            return None, "nonfinite denominator", iteration + 1
        if (denominator <= 0).any():
            return None, "nonpositive denominator", iteration + 1
        prediction = physics.A(recon).clamp_min(eps)
        recon = (recon * physics.A_adjoint(y / prediction) / denominator).clamp_min(eps)
        if not torch.isfinite(recon).all():
            return None, "nonfinite reconstruction", iteration + 1
    return recon, None, None


@torch.no_grad()
def run_proposed(reg_weight):
    recon, _ = rbpg_tv(
        y=y,
        x_init=x_init,
        stepsize=1,
        physics=physics,
        reg_weight=reg_weight,
        max_steps=max_iter,
        niter_tv=niter_tv,
        verbose=False,
        filter_epsilon=eps,
    )
    return recon if torch.isfinite(recon).all() and (recon >= 0).all() else None


# %%
proposed_reconstructions = [
    run_proposed(weight) for weight in tqdm(reg_weights, desc="R-BPG + TV sweep")
]
osl_runs = [run_osl(weight) for weight in tqdm(reg_weights, desc="OSL + TV sweep")]
reconstructions = [proposed_reconstructions, [run[0] for run in osl_runs]]
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
if any(not np.isfinite(values).any() for values in psnr_values):
    raise RuntimeError("A method has no successful reconstruction in the sweep.")
best_indices = [int(np.nanargmax(values)) for values in psnr_values]
reconstruction_reg_weights = [float(reg_weights[index]) for index in best_indices]
selected_reconstructions = [
    method_reconstructions[index]
    for method_reconstructions, index in zip(reconstructions, best_indices, strict=True)
]
selected_psnr_values = [
    float(values[index])
    for values, index in zip(psnr_values, best_indices, strict=True)
]
print(
    json.dumps(
        {
            "blur_sigma": 2.0,
            "gain": gain,
            "best_weights": dict(
                zip(method_names, reconstruction_reg_weights, strict=True)
            ),
            "best_psnr": dict(zip(method_names, selected_psnr_values, strict=True)),
            "proposed_failed_weights": [
                float(weight)
                for weight, reconstruction in zip(
                    reg_weights, proposed_reconstructions, strict=True
                )
                if reconstruction is None
            ],
            "osl_failures": [
                {"weight": float(weight), "reason": run[1], "iteration": run[2]}
                for weight, run in zip(reg_weights, osl_runs, strict=True)
                if run[0] is None
            ],
        },
        indent=2,
    )
)

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
        f"{name}\n" rf"$\lambda={reg_weight:.3g}$, " f"PSNR={psnr_value:.2f} dB"
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
failed_weights = [
    weight for weight, run in zip(reg_weights, osl_runs, strict=True) if run[0] is None
]
if failed_weights:
    psnr_ax.plot(
        failed_weights,
        [0.04] * len(failed_weights),
        "x",
        transform=psnr_ax.get_xaxis_transform(),
        color=sns.color_palette("colorblind")[1],
        label="OSL: invalid update",
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

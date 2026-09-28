# %%
# ruff: noqa: E402
import sys
from pathlib import Path

cwd = Path.cwd().resolve()
project_root = next(
    path
    for path in (cwd, *cwd.parents)
    if (path / "python" / "src" / "algos.py").is_file()
)
sys.path.insert(0, str(project_root / "python"))

import deepinv as dinv
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.algos import (
    mlem,
    mlem_L2,
    mlem_L2_osl,
    mlem_tv_osl,
    mlem_tv,
    nolips,
    nolips_L2,
)

from src.utils import KL_L2, KL_TV

sns.set_theme(
    context="paper", style="whitegrid", font="serif", font_scale=2, palette="colorblind"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
article_figure_directory = project_root / "latex" / "figures"
article_figure_directory.mkdir(parents=True, exist_ok=True)

img_size = 256

x = dinv.utils.load_example(
    "butterfly.png",
    img_size=img_size,
    grayscale=True,
    resize_mode="resize",
    device=device,
)
gain = 1 / 20
noise_model = dinv.physics.PoissonNoise(gain=gain)

psnr = dinv.metric.PSNR()
# Blur operator
sigma_blur = 2.0
psf = dinv.physics.blur.gaussian_blur(sigma=sigma_blur)
physics = dinv.physics.Blur(
    filter=psf,
    padding="circular",
    device=device,
    noise_model=noise_model,
)

y = physics(x)

# %%
stepsize = 1
max_iter = 150
reg_weight_l2 = 0.1
reg_weight_tv_osl = 0.1
reg_weight_tv = 0.1
x_mlem, xs_mlem = mlem(
    y=y,
    x_init=torch.ones_like(y),
    stepsize=stepsize,
    physics=physics,
    max_steps=max_iter,
    verbose=True,
    keep_inter=True,
)

x_nolips, xs_nolips = nolips(
    y=y,
    x_init=torch.ones_like(y),
    stepsize=stepsize,
    physics=physics,
    max_steps=max_iter,
    verbose=True,
    keep_inter=True,
)
x_mlem_L2, xs_mlem_L2 = mlem_L2(
    y=y,
    x_init=torch.ones_like(y),
    stepsize=stepsize,
    physics=physics,
    reg_weight=reg_weight_l2,
    max_steps=max_iter,
    verbose=True,
    keep_inter=True,
    filter_epsilon=1e-10,
)

x_nolips_L2, xs_nolips_L2 = nolips_L2(
    y=y,
    x_init=torch.ones_like(y),
    stepsize=stepsize,
    physics=physics,
    reg_weight=reg_weight_l2,
    max_steps=max_iter,
    verbose=True,
    keep_inter=True,
    filter_epsilon=1e-10,
)

x_mlem_L2_osl, xs_mlem_L2_osl = mlem_L2_osl(
    y=y,
    x_init=torch.ones_like(y),
    stepsize=stepsize,
    physics=physics,
    reg_weight=reg_weight_l2,
    max_steps=max_iter,
    verbose=True,
    keep_inter=True,
    filter_epsilon=1e-10,
)

x_mlem_tv_osl, xs_mlem_tv_osl = mlem_tv_osl(
    y=y,
    x_init=torch.ones_like(y),
    stepsize=stepsize,
    physics=physics,
    reg_weight=reg_weight_tv_osl,
    max_steps=max_iter,
    verbose=True,
    keep_inter=True,
    filter_epsilon=1e-10,
)

x_mlem_tv, xs_mlem_tv = mlem_tv(
    y=y,
    x_init=torch.ones_like(y),
    stepsize=stepsize,
    physics=physics,
    reg_weight=reg_weight_tv,
    max_steps=max_iter,
    niter_tv=100,
    tv_prox="pdhg",
    verbose=True,
    keep_inter=True,
    filter_epsilon=1e-10,
)

# %%
method_names = [
    "MU",
    "R-BPG + L2",
    "R-BPG + TV",
    "NoLips",
    "NoLips + L2",
    "OSL + L2",
    "OSL + TV",
]
method_reconstructions = {
    "MU": x_mlem,
    "R-BPG + L2": x_mlem_L2,
    "R-BPG + TV": x_mlem_tv,
    "NoLips": x_nolips,
    "NoLips + L2": x_nolips_L2,
    "OSL + L2": x_mlem_L2_osl,
    "OSL + TV": x_mlem_tv_osl,
}
method_iterates = {
    "MU": xs_mlem,
    "R-BPG + L2": xs_mlem_L2,
    "R-BPG + TV": xs_mlem_tv,
    "NoLips": xs_nolips,
    "NoLips + L2": xs_nolips_L2,
    "OSL + L2": xs_mlem_L2_osl,
    "OSL + TV": xs_mlem_tv_osl,
}
method_linestyles = {
    "MU": "-",
    "R-BPG + L2": "--",
    "R-BPG + TV": ":",
    "NoLips": "-.",
    "NoLips + L2": (0, (7, 2)),
    "OSL + L2": (0, (5, 2, 1, 2)),
    "OSL + TV": (0, (1, 2, 1, 2, 5, 2)),
}
method_linewidths = {name: 4 if name == "R-BPG + TV" else 2.5 for name in method_names}
method_colors = {
    "MU": "#4D4D4D",
    "R-BPG + L2": "#E69F00",
    "R-BPG + TV": "#009E73",
    "NoLips": "#56B4E9",
    "NoLips + L2": "#0072B2",
    "OSL + L2": "#CC79A7",
    "OSL + TV": "#D55E00",
}


# %%
# Arrange methods by regularizer, with the ground truth in the bottom-right panel.
reconstruction_titles = [
    "Measurement",
    "MU",
    "NoLips",
    "OSL + L2",
    "R-BPG + L2",
    "NoLips + L2",
    "OSL + TV",
    "R-BPG + TV",
    "Ground truth",
]
reconstruction_images = [
    y,
    *[method_reconstructions[name] for name in reconstruction_titles[1:-1]],
    x,
]
reconstruction_subtitles = [
    f"PSNR: {psnr(x, y).item():.2f} dB",
    *[
        f"PSNR: {psnr(x, method_reconstructions[name]).item():.2f} dB"
        for name in reconstruction_titles[1:-1]
    ],
    "",
]

reconstruction_figure, reconstruction_axes = plt.subplots(3, 3, figsize=(15, 15))
for axis, image, title, subtitle in zip(
    reconstruction_axes.flat,
    reconstruction_images,
    reconstruction_titles,
    reconstruction_subtitles,
    strict=True,
):
    axis.imshow(
        image.squeeze().detach().cpu().clamp(0, 1),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axis.set_title(title)
    axis.set_xlabel(subtitle)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)

reconstruction_figure.tight_layout()
reconstruction_figure.savefig(
    article_figure_directory / "deconvolution_2.pdf",
    dpi=300,
    pad_inches=0.0,
    bbox_inches="tight",
    transparent=True,
)


# %%
def evaluate_objective(iterates, objective):
    values = []
    with torch.no_grad():
        for iterate in iterates:
            value = objective(iterate.to(device), y, physics)
            values.append(float(value.detach().cpu().item()))
    return values


def evaluate_psnr(iterates):
    values = []
    with torch.no_grad():
        for iterate in iterates:
            values.append(float(psnr(x, iterate.to(device)).detach().cpu().item()))
    return values


def relative_iterate_progress(iterates, epsilon=1e-16):
    """Return ||x_k - x_(k-1)||_2 / ||x_(k-1)||_2 for each stored iterate."""
    values = [float("nan")]
    for previous, current in zip(iterates[:-1], iterates[1:], strict=True):
        denominator = torch.linalg.norm(previous).clamp_min(epsilon)
        progress = torch.linalg.norm(current - previous) / denominator
        values.append(max(float(progress.item()), epsilon))
    return values


def plot_method(axis, name, values, start_iteration=0):
    axis.plot(
        range(start_iteration, len(values)),
        values[start_iteration:],
        color=method_colors[name],
        linestyle=method_linestyles[name],
        linewidth=method_linewidths[name],
        zorder=3 if name == "R-BPG + TV" else 2,
        label=name,
    )


l2_objective = KL_L2(gain=gain, reg_weight=reg_weight_l2)
tv_objective = KL_TV(gain=gain, reg_weight=reg_weight_tv)
l2_objective_names = ["R-BPG + L2", "NoLips + L2", "OSL + L2"]
tv_objective_names = ["R-BPG + TV", "OSL + TV"]

l2_objective_histories = {
    name: evaluate_objective(method_iterates[name], l2_objective)
    for name in l2_objective_names
}
tv_objective_histories = {
    name: evaluate_objective(method_iterates[name], tv_objective)
    for name in tv_objective_names
}
psnr_histories = {name: evaluate_psnr(method_iterates[name]) for name in method_names}
progress_histories = {
    name: relative_iterate_progress(method_iterates[name])
    for name in [*l2_objective_names, *tv_objective_names]
}


# %%
metrics_figure, metrics_axes = plt.subplots(2, 3, figsize=(22, 12))
(
    l2_objective_axis,
    psnr_axis,
    l2_progress_axis,
    tv_objective_axis,
    unused_axis,
    tv_progress_axis,
) = metrics_axes.flat

for name in l2_objective_names:
    plot_method(
        l2_objective_axis, name, l2_objective_histories[name], start_iteration=10
    )
    plot_method(l2_progress_axis, name, progress_histories[name])
for name in tv_objective_names:
    plot_method(
        tv_objective_axis, name, tv_objective_histories[name], start_iteration=10
    )
    plot_method(tv_progress_axis, name, progress_histories[name])
for name in method_names:
    plot_method(psnr_axis, name, psnr_histories[name])

l2_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\frac{\lambda}{2}\|x\|_2^2$",
    title=rf"L2 objective ($\lambda={reg_weight_l2:g}$)",
)
tv_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\lambda\,\mathrm{TV}(x)$",
    title=rf"TV objective ($\lambda={reg_weight_tv:g}$)",
)
psnr_axis.set(
    xlabel="Iteration",
    ylabel="PSNR (dB)",
    title="Reconstruction quality",
)
l2_progress_axis.set(
    xlabel="Iteration",
    ylabel=r"$\|x^{(k)}-x^{(k-1)}\|_2/\|x^{(k-1)}\|_2$",
    title="L2 iterate relative progress",
)
tv_progress_axis.set(
    xlabel="Iteration",
    ylabel=r"$\|x^{(k)}-x^{(k-1)}\|_2/\|x^{(k-1)}\|_2$",
    title="TV iterate relative progress",
)

l2_objective_axis.set_yscale("log")
tv_objective_axis.set_yscale("log")
l2_progress_axis.set_yscale("log")
tv_progress_axis.set_yscale("log")
unused_axis.axis("off")
for axis in (
    l2_objective_axis,
    tv_objective_axis,
    psnr_axis,
    l2_progress_axis,
    tv_progress_axis,
):
    axis.legend(fontsize="small", ncol=2)

metrics_figure.tight_layout()
metrics_figure.savefig(
    article_figure_directory / "deconvolution_metrics_2.pdf",
    dpi=300,
    pad_inches=0.0,
    bbox_inches="tight",
    transparent=True,
)

# %%

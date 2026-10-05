# %%
"""Compare unregularized, L2, and TV methods on a Poisson deconvolution example.

Use the PET examples' algorithm interface and metric histories.
"""

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
from src.utils import method_plot_styles

from src.algos import (
    mu,
    rbpg_l2,
    osl_l2,
    osl_tv,
    rbpg_tv,
    mirror_descent,
    bpg_l2,
)

sns.set_theme(
    context="paper", style="whitegrid", font="serif", font_scale=2, palette="colorblind"
)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
article_figure_directory = project_root / "latex" / "figures"
article_figure_directory.mkdir(parents=True, exist_ok=True)

# %%
img_size = 256

x = dinv.utils.load_example(
    "butterfly.png",
    img_size=img_size,
    grayscale=True,
    resize_mode="resize",
    device=device,
)
gain = 1 / 10
noise_model = dinv.physics.PoissonNoise(gain=gain)

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
print(
    f"gain={gain:g}: measurement SNR={dinv.metric.SNR()(y, physics.A(x)).item():.2f} dB"
)

# %%
x_init = torch.ones_like(x)
l2_weight = 0.1
tv_weight = 0.1
mu_steps = 150
max_steps = 150
stepsize = 1.0
niter_tv = 100
filter_epsilon = 1e-10

# Each method returns its reconstruction and the plotted metric histories.
x_mu, metrics_mu = mu(
    y,
    x_init,
    stepsize,
    physics,
    max_steps=mu_steps,
    x_ref=x,
    filter_epsilon=filter_epsilon,
)
x_rbpg_l2, metrics_rbpg_l2 = rbpg_l2(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=l2_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=filter_epsilon,
)
x_rbpg_tv, metrics_rbpg_tv = rbpg_tv(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=tv_weight,
    max_steps=max_steps,
    niter_tv=niter_tv,
    x_ref=x,
    filter_epsilon=filter_epsilon,
)
x_md, metrics_md = mirror_descent(
    y,
    x_init,
    stepsize,
    physics,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=filter_epsilon,
)
x_bpg_l2, metrics_bpg_l2 = bpg_l2(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=l2_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=filter_epsilon,
)
x_osl_l2, metrics_osl_l2 = osl_l2(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=l2_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=filter_epsilon,
)
x_osl_tv, metrics_osl_tv = osl_tv(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=tv_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=filter_epsilon,
)

# %%
reconstructions = {
    "MU": x_mu,
    "R-BPG + L2": x_rbpg_l2,
    "R-BPG + TV": x_rbpg_tv,
    "Mirror descent": x_md,
    "BPG + L2": x_bpg_l2,
    "OSL + L2": x_osl_l2,
    "OSL + TV": x_osl_tv,
}
metrics = {
    "MU": metrics_mu,
    "R-BPG + L2": metrics_rbpg_l2,
    "R-BPG + TV": metrics_rbpg_tv,
    "Mirror descent": metrics_md,
    "BPG + L2": metrics_bpg_l2,
    "OSL + L2": metrics_osl_l2,
    "OSL + TV": metrics_osl_tv,
}
plot_styles = method_plot_styles()

# %%
# Arrange methods by regularizer, with the ground truth in the bottom-right panel.
reconstruction_titles = [
    "Measurement",
    "MU",
    "Mirror descent",
    "OSL + L2",
    "R-BPG + L2",
    "BPG + L2",
    "OSL + TV",
    "R-BPG + TV",
    "Ground truth",
]
reconstruction_images = [
    y,
    *[reconstructions[name] for name in reconstruction_titles[1:-1]],
    x,
]
nrmse = dinv.metric.NMSE(reduction="mean")
reconstruction_subtitles = [
    f"NRMSE: {100 * nrmse(y, x).sqrt().item():.2f}%",
    *[
        f"NRMSE: {metrics[name]['nrmse'][-1]:.2f}%"
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
l2_methods = ("R-BPG + L2", "BPG + L2", "OSL + L2")
tv_methods = ("OSL + TV", "R-BPG + TV")
plot_order = (*[name for name in metrics if name != "R-BPG + TV"], "R-BPG + TV")

metrics_figure, metrics_axes = plt.subplots(2, 3, figsize=(22, 12))
(
    l2_objective_axis,
    nrmse_axis,
    l2_progress_axis,
    tv_objective_axis,
    unused_axis,
    tv_progress_axis,
) = metrics_axes.flat

for names, objective_axis, progress_axis in (
    (l2_methods, l2_objective_axis, l2_progress_axis),
    (tv_methods, tv_objective_axis, tv_progress_axis),
):
    for name in names:
        method_metrics = metrics[name]
        iterations = range(len(method_metrics["objective"]))
        objective_axis.plot(
            iterations[5:], method_metrics["objective"][5:], **plot_styles[name]
        )
        progress_axis.plot(
            iterations, method_metrics["relative_progress"], **plot_styles[name]
        )

for name in plot_order:
    method_metrics = metrics[name]
    nrmse_axis.plot(
        range(5, len(method_metrics["nrmse"])),
        method_metrics["nrmse"][5:],
        **plot_styles[name],
    )

l2_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\frac{\lambda}{2}\|x\|_2^2$",
    title=rf"L2 objective ($\lambda={l2_weight:g}$)",
)
tv_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\lambda\|\nabla x\|_{1,2}$",
    title=rf"TV objective ($\lambda={tv_weight:g}$)",
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
nrmse_axis.set(
    xlabel="Iteration",
    ylabel="NRMSE (%)",
    title="Reconstruction error",
)
unused_axis.axis("off")
l2_objective_axis.set_yscale("log")
tv_objective_axis.set_yscale("log")
l2_progress_axis.set_yscale("log")
tv_progress_axis.set_yscale("log")
for axis in (
    l2_objective_axis,
    tv_objective_axis,
    nrmse_axis,
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

# %%
# ruff: noqa: E402
"""Compare unregularized, L2, and 3D-TV methods for BrainWeb PET."""

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
    mlem_tv,
    mlem_tv_osl,
    nolips,
    nolips_L2,
)
from src.brainweb import load_brainweb_pet_sample, recovery_coefficients
from src.prox import torch_gradient, torch_module

sns.set_theme(
    context="paper", style="whitegrid", font="serif", font_scale=2, palette="colorblind"
)
torch.manual_seed(0)
device = "cuda"


# %%
sample = load_brainweb_pet_sample(
    root=project_root / ".pixi" / "deepinv-cache" / "datasets" / "BrainWeb",
    lesion_diameters_mm=(6, 8, 10, 12, 16),
    lesion_activity=192,
    target_voxel_size_mm=1,
    seed=0,
)
x = sample.emission.to(device)
attenuation = sample.attenuation.to(device)
lesion_mask = sample.lesion_mask.to(device)
img_size = tuple(x.shape[-3:])


# %%
gain = 1 / 100
physics = dinv.physics.PET(
    img_size=img_size,
    voxel_size=sample.voxel_size,
    fwhm_data_mm=4,
    gain=gain,
    normalize=True,
    normalize_counts=True,
    device=device,
    attenuation=attenuation,
)
y = physics(x)
print(y.shape)

# %%
l2_reg_weight = 0.01
tv_reg_weight = 0.01
niter_tv = 20
x_init = torch.ones_like(physics.A_adjoint(y))
reconstruction_parameters = {
    "y": y,
    "x_init": x_init,
    "stepsize": 1,
    "physics": physics,
    "max_steps": 50,
    "filter_epsilon": 1e-5,
}
l2_reconstruction_parameters = {
    **reconstruction_parameters,
    "reg_weight": l2_reg_weight,
}
tv_mu_reconstruction_parameters = {
    **reconstruction_parameters,
    "reg_weight": tv_reg_weight,
    "niter_tv": niter_tv,
    "tv_prox": "pdhg",
}
tv_osl_reconstruction_parameters = {
    **reconstruction_parameters,
    "reg_weight": tv_reg_weight,
}

reconstruction_names = [
    "MU",
    "MU + L2",
    "MU + TV",
    "NoLips",
    "NoLips + L2",
    "OSL + L2",
    "OSL + TV",
]
method_linestyles = {
    "MU": "-",
    "MU + L2": "--",
    "MU + TV": ":",
    "NoLips": "-.",
    "NoLips + L2": (0, (7, 2)),
    "OSL + L2": (0, (5, 2, 1, 2)),
    "OSL + TV": (0, (1, 2, 1, 2, 5, 2)),
}
method_linewidths = {
    name: 4 if name == "MU + TV" else 2.5 for name in reconstruction_names
}
method_colors = {
    "MU": "#4D4D4D",
    "MU + L2": "#E69F00",
    "MU + TV": "#009E73",
    "NoLips": "#56B4E9",
    "NoLips + L2": "#0072B2",
    "OSL + L2": "#CC79A7",
    "OSL + TV": "#D55E00",
}
nrmse = dinv.metric.NMSE(reduction="mean")
histories = {}


def reconstruct_with_history(
    name,
    algorithm,
    parameters,
    tracked_objectives,
    snapshot_iteration=None,
):
    history = {
        "iteration": [],
        "nrmse": [],
        "relative_progress": [],
        "snapshot": None,
        **{objective: [] for objective in tracked_objectives},
    }
    previous_reconstruction = None

    def record_iterate(iteration, reconstruction, prediction):
        nonlocal previous_reconstruction
        poisson_nll = (prediction - y * prediction.log()).sum()
        history["iteration"].append(iteration)
        history["nrmse"].append(100 * nrmse(reconstruction, x).sqrt().item())
        if iteration == snapshot_iteration:
            history["snapshot"] = reconstruction.detach().clone()
        if previous_reconstruction is None:
            history["relative_progress"].append(float("nan"))
        else:
            denominator = torch.linalg.norm(previous_reconstruction).clamp_min(1e-16)
            relative_progress = (
                torch.linalg.norm(reconstruction - previous_reconstruction)
                / denominator
            )
            history["relative_progress"].append(max(relative_progress.item(), 1e-16))
        previous_reconstruction = reconstruction.detach().clone()
        if "l2_objective" in tracked_objectives:
            l2_penalty = (l2_reg_weight / 2) * reconstruction.square().sum()
            history["l2_objective"].append((poisson_nll + l2_penalty).item())
        if "tv_objective" in tracked_objectives:
            tv_penalty = (
                tv_reg_weight * torch_module(torch_gradient(reconstruction)).sum()
            )
            history["tv_objective"].append((poisson_nll + tv_penalty).item())

    reconstruction = algorithm(
        **parameters,
        iterate_callback=record_iterate,
    )
    if snapshot_iteration is not None and history["snapshot"] is None:
        raise ValueError(
            f"{name} did not reach requested snapshot iteration {snapshot_iteration}."
        )
    histories[name] = history
    return reconstruction


x_mu = reconstruct_with_history(
    "MU",
    mlem,
    reconstruction_parameters,
    tracked_objectives=("l2_objective", "tv_objective"),
    snapshot_iteration=50,
)
x_mlem = reconstruct_with_history(
    "MU + L2",
    mlem_L2,
    l2_reconstruction_parameters,
    tracked_objectives=("l2_objective",),
)
x_mu_tv = reconstruct_with_history(
    "MU + TV",
    mlem_tv,
    tv_mu_reconstruction_parameters,
    tracked_objectives=("tv_objective",),
)
x_nolips_unregularized = reconstruct_with_history(
    "NoLips",
    nolips,
    reconstruction_parameters,
    tracked_objectives=("l2_objective", "tv_objective"),
)
x_nolips = reconstruct_with_history(
    "NoLips + L2",
    nolips_L2,
    l2_reconstruction_parameters,
    tracked_objectives=("l2_objective",),
)
# With zero regularization, the OSL update is identical to MU, so only the
# regularized OSL variant needs a separate run.
x_osl = reconstruct_with_history(
    "OSL + L2",
    mlem_L2_osl,
    l2_reconstruction_parameters,
    tracked_objectives=("l2_objective",),
)
x_osl_tv = reconstruct_with_history(
    "OSL + TV",
    mlem_tv_osl,
    tv_osl_reconstruction_parameters,
    tracked_objectives=("tv_objective",),
)


# %%
reconstructions = [
    histories["MU"]["snapshot"],
    x_mlem,
    x_mu_tv,
    x_nolips_unregularized,
    x_nolips,
    x_osl,
    x_osl_tv,
]
# Show the transverse plane with the most emission instead of the first lesion,
# which is close to the base of the brain for this seeded sample.
transverse_slice = int(x[0, 0].sum(dim=(0, 1)).argmax().item())
display_images = [
    # Put the anterior side at the top of the displayed image.
    torch.rot90(volume[..., transverse_slice], k=1, dims=(-2, -1))
    for volume in [x, *reconstructions]
]
rc_by_method = {
    name: recovery_coefficients(reconstruction, x, lesion_mask)
    for name, reconstruction in zip(reconstruction_names, reconstructions, strict=True)
}

reconstruction_titles = ["Ground truth", "MU (iteration 50)", *reconstruction_names[1:]]
reconstruction_subtitles = [
    "",
    *[
        f"NRMSE: {100 * nrmse(x_hat, x).sqrt().item():.2f}%"
        for name, x_hat in zip(reconstruction_names, reconstructions, strict=True)
    ],
]

reconstruction_figure, reconstruction_axes = plt.subplots(
    2,
    4,
    figsize=(16, 11),
)
for axis, image, title, subtitle in zip(
    reconstruction_axes.flat,
    display_images,
    reconstruction_titles,
    reconstruction_subtitles,
    strict=True,
):
    axis.imshow(
        image.squeeze().detach().cpu().clamp(0, 1),
        cmap="gray_r",
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

# %%
l2_objective_names = [
    "MU + L2",
    "NoLips + L2",
    "OSL + L2",
]
tv_objective_names = ["MU + TV", "OSL + TV"]

metrics_figure, metrics_axes = plt.subplots(2, 3, figsize=(22, 12))
(
    l2_objective_axis,
    nrmse_axis,
    l2_progress_axis,
    tv_objective_axis,
    recovery_axis,
    tv_progress_axis,
) = metrics_axes.flat

for name in l2_objective_names:
    history = histories[name]
    l2_objective_axis.plot(
        history["iteration"],
        history["l2_objective"],
        color=method_colors[name],
        linestyle=method_linestyles[name],
        linewidth=method_linewidths[name],
        zorder=3 if name == "MU + TV" else 2,
        label=name,
    )

for name in tv_objective_names:
    history = histories[name]
    tv_objective_axis.plot(
        history["iteration"],
        history["tv_objective"],
        color=method_colors[name],
        linestyle=method_linestyles[name],
        linewidth=method_linewidths[name],
        zorder=3 if name == "MU + TV" else 2,
        label=name,
    )

for name in l2_objective_names:
    history = histories[name]
    l2_progress_axis.plot(
        history["iteration"],
        history["relative_progress"],
        color=method_colors[name],
        linestyle=method_linestyles[name],
        linewidth=method_linewidths[name],
        zorder=3 if name == "MU + TV" else 2,
        label=name,
    )

for name in tv_objective_names:
    history = histories[name]
    tv_progress_axis.plot(
        history["iteration"],
        history["relative_progress"],
        color=method_colors[name],
        linestyle=method_linestyles[name],
        linewidth=method_linewidths[name],
        zorder=3 if name == "MU + TV" else 2,
        label=name,
    )

for name in reconstruction_names:
    history = histories[name]
    nrmse_axis.plot(
        history["iteration"],
        history["nrmse"],
        color=method_colors[name],
        linestyle=method_linestyles[name],
        linewidth=method_linewidths[name],
        zorder=3 if name == "MU + TV" else 2,
        label=name,
    )
    recovery_axis.plot(
        sample.lesion_diameters_mm,
        rc_by_method[name].mean.detach().cpu(),
        color=method_colors[name],
        linestyle=method_linestyles[name],
        linewidth=method_linewidths[name],
        zorder=3 if name == "MU + TV" else 2,
        label=name,
    )

l2_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\frac{\lambda}{2}\|x\|_2^2$",
    title=rf"L2 objective ($\lambda={l2_reg_weight:g}$)",
)
tv_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\lambda\|\nabla x\|_{1,2}$",
    title=rf"3D TV objective ($\lambda={tv_reg_weight:g}$)",
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
recovery_axis.axhline(
    1,
    color="black",
    linewidth=1,
    linestyle=":",
    label="Ideal",
)
recovery_axis.set(
    xlabel="Sphere diameter (mm)",
    ylabel="Mean recovery coefficient",
    title="BrainWeb hot-sphere recovery",
)
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
    recovery_axis,
):
    axis.legend(fontsize="small", ncol=2)
metrics_figure.tight_layout()

# %%

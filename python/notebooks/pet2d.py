# %%
"""Compare algorithms on a 2D PET reconstruction example.

This corresponds to the second figure of the article.
"""

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
import parallelproj
import seaborn as sns
import torch
from src.utils import method_plot_styles
from array_api_compat import torch as torch_compat

from src.algos import (
    mu,
    rbpg_l2,
    osl_l2,
    rbpg_tv,
    osl_tv,
    mirror_descent,
    bpg_l2,
)

sns.set_theme(
    context="paper", style="whitegrid", font="serif", font_scale=2, palette="colorblind"
)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    # Small 2D tensors run efficiently without a large CPU thread pool.
    torch.set_num_threads(min(4, torch.get_num_threads()))
output_directory = project_root / "output" / "pet2d"
output_directory.mkdir(parents=True, exist_ok=True)


# %%
# Setup the physics
img_size = (120, 120)
voxel_size = (2.0, 2.0)
lesion_diameters_mm = (6, 8, 10, 12, 16)


def center_slice(volume):
    crop = tuple(
        slice((size - length) // 2, (size + length) // 2)
        for size, length in zip(volume.shape[-2:], img_size, strict=True)
    )
    return volume[(..., volume.shape[-3] // 2, *crop)].clone()


anatomy_cache = output_directory / f"brainweb_subject04_slice_{img_size[0]}x{img_size[1]}.pt"
if anatomy_cache.is_file():
    anatomy = torch.load(anatomy_cache, map_location="cpu", weights_only=True)
    emission, attenuation = anatomy["emission"], anatomy["attenuation"]
else:
    dataset = dinv.datasets.BrainWebPET(
        root=project_root / ".pixi" / "deepinv-cache" / "datasets" / "BrainWebPET",
        subject_ids=4,
        transform=center_slice,
        seed=0,
    )
    emission, params = dataset[0]
    attenuation = params["attenuation"]
    torch.save({"emission": emission, "attenuation": attenuation}, anatomy_cache)
    del dataset, params

# Add hot disks directly in the plane.
# Choose seeded tissue centers, with enough room for each disk and a 4 mm gap
emission = emission.clone()
lesion_mask = torch.zeros_like(emission, dtype=torch.uint8)
rows, columns = torch.meshgrid(
    torch.arange(img_size[0]) * voxel_size[0],
    torch.arange(img_size[1]) * voxel_size[1],
    indexing="ij",
)
candidates = torch.nonzero(emission[0] > 30)
generator = torch.Generator().manual_seed(0)
candidates = candidates[torch.randperm(len(candidates), generator=generator)]
centers = []
for label, diameter in reversed(list(enumerate(lesion_diameters_mm, start=1))):
    radius = diameter / 2
    for row, column in candidates.tolist():
        center = (row * voxel_size[0], column * voxel_size[1])
        if not all(
            radius <= coordinate <= (length - 1) * spacing - radius
            for coordinate, length, spacing in zip(center, img_size, voxel_size)
        ):
            continue
        if any(
            sum((a - b) ** 2 for a, b in zip(center, previous))
            <= (radius + previous_radius + 4) ** 2
            for previous, previous_radius in centers
        ):
            continue
        disk = (rows - center[0]).square() + (columns - center[1]).square() <= radius**2
        emission[0, disk] = 192.0
        lesion_mask[0, disk] = label
        centers.append((center, radius))
        break
    else:
        raise RuntimeError(f"No tissue center found for the {diameter} mm lesion.")

x = emission.unsqueeze(0).to(device)
x /= x.max()
attenuation = attenuation.unsqueeze(0).to(device)
lesion_mask = lesion_mask.unsqueeze(0).to(device)


# %%
scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_rings=1,
    num_sides=34,
    num_lor_endpoints_per_side=8,
    lor_spacing=8,
)
physics = dinv.physics.PET(
    img_size=img_size,
    voxel_size=voxel_size,
    scanner=scanner,
    fwhm_data_mm=3.0,
    gain=1,
    normalize=True,
    normalize_counts=True,
    device=device,
    attenuation=attenuation,
)

expected_signal = physics.A(x)
target_counts = 500_000
background_fraction = 0.2
background = torch.full_like(
    expected_signal,
    expected_signal.mean() * background_fraction / (1 - background_fraction),
)
gain = (expected_signal.sum() + background.sum()).item() / target_counts
physics.noise_model.update_parameters(gain=gain)
physics.update(background=background)
y = physics(x)
print(f"Image: {img_size}, voxel size: {voxel_size} mm")
print(f"Sinogram: {tuple(y.shape)}, prompt counts: {(y / gain).sum().item():,.0f}")
del expected_signal

# %%
# Run the algorithms
x_init = torch.ones_like(x)
l2_weight = 0.01
tv_weight = 0.0015
mu_steps = 20
max_steps = 50
stepsize = 1.0
niter_tv = 100

# Each method returns its reconstruction and the plotted metric histories.
x_mu, metrics_mu = mu(
    y,
    x_init,
    stepsize,
    physics,
    max_steps=mu_steps,
    x_ref=x,
    filter_epsilon=1e-5,
)
x_rbpg_l2, metrics_rbpg_l2 = rbpg_l2(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=l2_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=1e-5,
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
    filter_epsilon=1e-5,
)
x_md, metrics_md = mirror_descent(
    y,
    x_init,
    stepsize,
    physics,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=1e-5,
)
x_bpg_l2, metrics_bpg_l2 = bpg_l2(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=l2_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=1e-5,
)
x_osl_l2, metrics_osl_l2 = osl_l2(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=l2_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=1e-5,
)
x_osl_tv, metrics_osl_tv = osl_tv(
    y,
    x_init,
    stepsize,
    physics,
    reg_weight=tv_weight,
    max_steps=max_steps,
    x_ref=x,
    filter_epsilon=1e-5,
)

# %%
# Plot reconstructions
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
recovery = dinv.metric.RecoveryCoefficient(reduction="mean")
plot_styles = method_plot_styles()

display_images = [x, *reconstructions.values()]
lesion_labels = torch.unique(lesion_mask)
lesion_labels = lesion_labels[lesion_labels > 0]
displayed_lesion_diameters = [
    lesion_diameters_mm[int(label) - 1] for label in lesion_labels
]

reconstruction_titles = [
    "Ground truth",
    f"MU (iteration {mu_steps})",
    *list(reconstructions)[1:],
]
reconstruction_subtitles = [
    "",
    *[
        f"2D NRMSE: {method_metrics['nrmse'][-1]:.2f}%"
        for method_metrics in metrics.values()
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
reconstruction_figure.savefig(
    output_directory / "pet2d_reconstructions.pdf", bbox_inches="tight"
)
reconstruction_figure.savefig(
    output_directory / "pet2d_reconstructions.png", dpi=160, bbox_inches="tight"
)

# %%
# Plot energies and metrics
l2_methods = ("R-BPG + L2", "BPG + L2", "OSL + L2")
tv_methods = ("OSL + TV", "R-BPG + TV")
plot_order = (*[name for name in reconstructions if name != "R-BPG + TV"], "R-BPG + TV")

metrics_figure, metrics_axes = plt.subplots(2, 3, figsize=(22, 12))
(
    l2_objective_axis,
    nrmse_axis,
    l2_progress_axis,
    tv_objective_axis,
    recovery_axis,
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
    reconstruction = reconstructions[name]
    method_metrics = metrics[name]
    nrmse_axis.plot(
        range(5, len(method_metrics["nrmse"])),
        method_metrics["nrmse"][5:],
        **plot_styles[name],
    )
    coefficients = [
        recovery(reconstruction, x, mask=lesion_mask == label).item()
        for label in lesion_labels
    ]
    recovery_axis.plot(
        displayed_lesion_diameters,
        coefficients,
        **{**plot_styles[name], "markevery": 1},
    )

l2_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\frac{\lambda}{2}\|x\|_2^2$",
    title="L2 objective",
)
tv_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\lambda\|\nabla x\|_{1,2}$",
    title="2D TV objective",
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
    xlabel="Disk diameter (mm)",
    ylabel="Mean recovery coefficient",
    title="BrainWeb hot-disk recovery",
)
recovery_axis.set_ylim(top=1.4)
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
metrics_figure.savefig(output_directory / "pet2d_metrics.pdf", bbox_inches="tight")
metrics_figure.savefig(
    output_directory / "pet2d_metrics.png", dpi=160, bbox_inches="tight"
)

torch.save(
    {
        "reference": x.detach().cpu(),
        "attenuation": attenuation.detach().cpu(),
        "lesion_mask": lesion_mask.detach().cpu(),
        "reconstructions": {
            name: image.detach().cpu() for name, image in reconstructions.items()
        },
        "histories": metrics,
        "voxel_size_mm": voxel_size,
        "gain": gain,
        "background_fraction": background_fraction,
        "target_prompt_counts": target_counts,
        "l2_weight": l2_weight,
        "tv_weight": tv_weight,
        "mu_steps": mu_steps,
        "max_steps": max_steps,
        "stepsize": stepsize,
        "niter_tv": niter_tv,
        "deepinv_commit": "251238b0f082fc42c83ff72708fdb3fa44de268c",
    },
    output_directory / "pet2d_reconstructions.pt",
)
print(
    f"Saved reconstructed images and metrics in {output_directory}"
)

# %%

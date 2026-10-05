# %%
"""Compare unregularized, L2, and 3D-TV methods on a BrainWeb PET example."""

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
output_directory = project_root / "output" / "pet3d"
output_directory.mkdir(parents=True, exist_ok=True)


# %%
volume_size = (120, 120, 120)
voxel_size = (2.0, 2.0, 2.0)  # (D, H, W), mm.
lesion_diameters_mm = (6, 8, 10, 12, 16)


def center_crop(volume):
    crop = tuple(
        slice((size - length) // 2, (size + length) // 2)
        for size, length in zip(volume.shape[-3:], volume_size, strict=True)
    )
    return volume[(..., *crop)]


dataset = dinv.datasets.BrainWebPET(
    root=project_root / ".pixi" / "deepinv-cache" / "datasets" / "BrainWebPET",
    subject_ids=4,
    lesion_diameters=list(lesion_diameters_mm),
    lesion_kwargs={
        "intensity": [192.0] * len(lesion_diameters_mm),
        "blur": [0.0] * len(lesion_diameters_mm),
        "thresh": 30,
    },
    transform=center_crop,
    seed=0,
)
emission, params = dataset[0]
x = emission.unsqueeze(0).to(device)
x /= x.max()
attenuation = params["attenuation"].unsqueeze(0).to(device)
lesion_mask = params["lesion_mask"].unsqueeze(0).to(device)
img_size = tuple(x.shape[-3:])


# %%
scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
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
target_prompt_counts = 50_000_000
background_fraction = 0.2
background = torch.full_like(
    expected_signal,
    expected_signal.mean() * background_fraction / (1 - background_fraction),
)
gain = (expected_signal.sum() + background.sum()).item() / target_prompt_counts
physics.noise_model.update_parameters(gain=gain)
physics.update(background=background)
y = physics(x)
print(f"Volume: {img_size}, voxel size: {voxel_size} mm")
print(f"Sinogram: {tuple(y.shape)}, prompt counts: {(y / gain).sum().item():,.0f}")
del expected_signal

# %%
# Initialize only where the scanner observes the volume. Burg updates cannot
# remove an initial value of one from voxels with zero data gradient.
sensitivity = physics.A_adjoint(torch.ones_like(y))
x_init = torch.ones_like(x)
l2_weight = 0.01
tv_weight = 0.005
mu_steps = 20
max_steps = 50
stepsize = 1.0
niter_tv = 100

# Each method returns its reconstruction and the three plotted metric metrics.
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
# All algorithms reconstruct the entire volume. Extract one common transverse
# plane only for display, NRMSE and lesion recovery use the full 3D volumes.
# Show the plane with the largest hot-sphere cross-section.
transverse_slice = int((lesion_mask[0, 0] > 0).sum(dim=(1, 2)).argmax().item())
display_images = [
    volume[:, :, transverse_slice] for volume in [x, *reconstructions.values()]
]
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
        f"3D NRMSE: {method_metrics['nrmse'][-1]:.2f}%"
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
reconstruction_figure.suptitle(
    f"3D PET reconstruction — transverse slice {transverse_slice}"
)
reconstruction_figure.tight_layout()
reconstruction_figure.savefig(
    output_directory / "pet3d_reconstructions.pdf", bbox_inches="tight"
)
reconstruction_figure.savefig(
    output_directory / "pet3d_reconstructions.png", dpi=160, bbox_inches="tight"
)

# %%
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
    recovery_axis.plot(displayed_lesion_diameters, coefficients, **plot_styles[name])

l2_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\frac{\lambda}{2}\|x\|_2^2$",
    title="L2 objective",
)
tv_objective_axis.set(
    xlabel="Iteration",
    ylabel=r"Poisson NLL $+\,\lambda\|\nabla x\|_{1,2}$",
    title="3D TV objective",
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
metrics_figure.savefig(output_directory / "pet3d_metrics.pdf", bbox_inches="tight")
metrics_figure.savefig(
    output_directory / "pet3d_metrics.png", dpi=160, bbox_inches="tight"
)

torch.save(
    {
        "reference": x.detach().cpu(),
        "attenuation": attenuation.detach().cpu(),
        "lesion_mask": lesion_mask.detach().cpu(),
        "reconstructions": {
            name: volume.detach().cpu() for name, volume in reconstructions.items()
        },
        "histories": metrics,
        "transverse_slice": transverse_slice,
        "voxel_size_mm": voxel_size,
        "gain": gain,
        "background_fraction": background_fraction,
        "target_prompt_counts": target_prompt_counts,
        "l2_weight": l2_weight,
        "tv_weight": tv_weight,
        "mu_steps": mu_steps,
        "max_steps": max_steps,
        "stepsize": stepsize,
        "niter_tv": niter_tv,
        "deepinv_commit": "251238b0f082fc42c83ff72708fdb3fa44de268c",
    },
    output_directory / "pet3d_reconstructions.pt",
)
print(
    f"Saved reconstructed volumes, transverse slice, and metrics in {output_directory}"
)

# %%

"""BrainWeb helpers shared by the PET experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import deepinv as dinv
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BrainWebPETSample:
    """A BrainWeb sample arranged in DeepInv PET's ``(x, y, z)`` convention."""

    emission: torch.Tensor
    attenuation: torch.Tensor
    lesion_mask: torch.Tensor
    voxel_size: tuple[float, float, float]
    lesion_diameters_mm: tuple[float, ...]


@dataclass(frozen=True)
class BrainWebPETSlice:
    """A BrainWeb transverse slice arranged in DeepInv PET's ``(x, y)`` convention."""

    emission: torch.Tensor
    attenuation: torch.Tensor
    lesion_mask: torch.Tensor
    voxel_size: tuple[float, float]
    lesion_diameters_mm: tuple[float, ...]


@dataclass(frozen=True)
class RecoveryCoefficients:
    """Mean activity recovery for each labelled lesion."""

    mean: torch.Tensor


def load_brainweb_pet_sample(
    *,
    root: str | Path | None = None,
    subject_id: int = 4,
    lesion_diameters_mm: Sequence[float] = (6.0, 8.0, 10.0, 12.0, 16.0),
    lesion_activity: float = 192.0,
    activity_levels: str | Mapping[str, float] = "fdg",
    target_voxel_size_mm: float = 1.0,
    seed: int = 0,
    download: bool = True,
) -> BrainWebPETSample:
    """Load BrainWeb, insert reproducible hot spheres, and prepare it for PET.

    BrainWeb is provided at 0.5 mm in ``(z, y, x)`` order. This helper resamples
    it to an isotropic target spacing, converts it to the ``(x, y, z)`` order
    expected by :class:`deepinv.physics.PET`, and converts attenuation from
    ``cm^-1`` to ``mm^-1``.
    """

    diameters = tuple(float(diameter) for diameter in lesion_diameters_mm)
    if not diameters or any(diameter <= 0 for diameter in diameters):
        raise ValueError("lesion_diameters_mm must contain positive values.")
    if target_voxel_size_mm <= 0:
        raise ValueError("target_voxel_size_mm must be positive.")

    dataset = dinv.datasets.BrainWebDataset(
        root=root,
        subject_ids=subject_id,
        download=download,
        activity_levels=activity_levels,
        lesions=[
            dinv.datasets.BrainWebLesion(
                diameter_mm=diameter,
                activity=lesion_activity,
            )
            for diameter in diameters
        ],
        seed=seed,
    )
    emission_zyx, params = dataset[0]

    target_shape_zyx = tuple(
        max(1, round(size * spacing / target_voxel_size_mm))
        for size, spacing in zip(
            emission_zyx.shape[-3:], dataset.voxel_size, strict=True
        )
    )

    def resample(volume: torch.Tensor, mode: str) -> torch.Tensor:
        interpolation_kwargs = {"align_corners": False} if mode == "trilinear" else {}
        volume = F.interpolate(
            volume.unsqueeze(0).float(),
            size=target_shape_zyx,
            mode=mode,
            **interpolation_kwargs,
        )
        return volume.permute(0, 1, 4, 3, 2).contiguous()

    emission = resample(emission_zyx, "trilinear")
    emission /= emission.max().clamp_min(torch.finfo(emission.dtype).eps)
    # BrainWeb stores linear attenuation coefficients in cm^-1, while the PET
    # projector traces rays through a geometry expressed in mm.
    attenuation = resample(params["attenuation"], "trilinear") / 10.0
    lesion_mask = resample(params["lesion_mask"], "nearest").to(torch.uint8)

    return BrainWebPETSample(
        emission=emission,
        attenuation=attenuation,
        lesion_mask=lesion_mask,
        voxel_size=(target_voxel_size_mm,) * 3,
        lesion_diameters_mm=diameters,
    )


def load_brainweb_pet_slice(
    *,
    root: str | Path | None = None,
    subject_id: int = 4,
    slice_index: int = 181,
    lesion_diameters_mm: Sequence[float] = (6.0, 8.0, 10.0, 12.0, 16.0),
    lesion_activity: float = 192.0,
    activity_levels: str | Mapping[str, float] = "fdg",
    target_voxel_size_mm: float = 1.0,
    seed: int = 0,
    download: bool = True,
) -> BrainWebPETSlice:
    """Load one BrainWeb slice and insert reproducible hot circular lesions.

    BrainWeb slices are provided at 0.5 mm in ``(y, x)`` order. This helper
    resamples one slice to an isotropic target spacing, converts it to the
    ``(x, y)`` order expected by :class:`deepinv.physics.PET`, and converts
    attenuation from ``cm^-1`` to ``mm^-1``.
    """

    diameters = tuple(float(diameter) for diameter in lesion_diameters_mm)
    if not diameters or any(diameter <= 0 for diameter in diameters):
        raise ValueError("lesion_diameters_mm must contain positive values.")
    if target_voxel_size_mm <= 0:
        raise ValueError("target_voxel_size_mm must be positive.")

    dataset = dinv.datasets.BrainWebDataset(
        root=root,
        subject_ids=subject_id,
        download=download,
        activity_levels=activity_levels,
        slice_index=slice_index,
        lesions=[
            dinv.datasets.BrainWebLesion(
                diameter_mm=diameter,
                activity=lesion_activity,
            )
            for diameter in diameters
        ],
        seed=seed,
    )
    emission_yx, params = dataset[0]
    target_shape_yx = tuple(
        max(1, round(size * spacing / target_voxel_size_mm))
        for size, spacing in zip(
            emission_yx.shape[-2:], dataset.voxel_size, strict=True
        )
    )

    def resample(volume: torch.Tensor, mode: str) -> torch.Tensor:
        interpolation_kwargs = {"align_corners": False} if mode == "bilinear" else {}
        volume = F.interpolate(
            volume.unsqueeze(0).float(),
            size=target_shape_yx,
            mode=mode,
            **interpolation_kwargs,
        )
        return volume.transpose(-2, -1).contiguous()

    emission = resample(emission_yx, "bilinear")
    emission /= emission.max().clamp_min(torch.finfo(emission.dtype).eps)
    attenuation = resample(params["attenuation"], "bilinear") / 10.0
    lesion_mask = resample(params["lesion_mask"], "nearest").to(torch.uint8)

    return BrainWebPETSlice(
        emission=emission,
        attenuation=attenuation,
        lesion_mask=lesion_mask,
        voxel_size=(target_voxel_size_mm,) * 2,
        lesion_diameters_mm=diameters,
    )


def recovery_coefficients(
    reconstruction: torch.Tensor,
    reference: torch.Tensor,
    lesion_mask: torch.Tensor,
) -> RecoveryCoefficients:
    """Compute the mean recovery coefficient for each labelled lesion.

    A value of one means that the reconstructed statistic matches the reference
    statistic inside that sphere.
    """

    if reconstruction.shape != reference.shape or reference.shape != lesion_mask.shape:
        raise ValueError(
            "reconstruction, reference, and lesion_mask must have identical shapes."
        )

    labels = torch.unique(lesion_mask)
    labels = labels[labels > 0]
    if labels.numel() == 0:
        raise ValueError("lesion_mask does not contain any labelled lesion.")

    mean_values = []
    for label in labels:
        region = lesion_mask == label
        reference_values = reference[region]
        reconstruction_values = reconstruction[region]
        reference_mean = reference_values.mean()
        if reference_mean <= 0:
            raise ValueError(f"Lesion {int(label)} has zero reference activity.")
        mean_values.append(reconstruction_values.mean() / reference_mean)

    return RecoveryCoefficients(mean=torch.stack(mean_values))

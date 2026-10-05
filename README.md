# MU-Bregman experiments

Code for the Poisson deconvolution and PET reconstruction experiments. The
experiment scripts are in `python/notebooks/`; the reconstruction algorithms
and proximal operators are in `python/src/`.

## Environment

The reproducible environment is defined by `pixi.toml` and `pixi.lock`. It
targets Linux with CUDA 12.9 and Python 3.12. Install it with
[`pixi`](https://pixi.sh/):

```sh
pixi install --locked -e full
pixi run --locked -e full test
```

The PET experiments use DeepInv's BrainWebPET dataset and download it on first
use. The 3D experiment also needs substantial GPU memory and runtime.

## Experiments

Run from the repository root, for example:

```sh
pixi run --locked -e full deconvolution-1
pixi run --locked -e full deconvolution-2
pixi run --locked -e full deconvolution-tv-stability
pixi run --locked -e full pet-2d
pixi run --locked -e full pet-3d
pixi run --locked -e full non-convexity
```

The deconvolution scripts save figures in `latex/figures/`. The PET scripts save
figures and reconstruction data in `output/pet2d/` and `output/pet3d/`.
`non-convexity` writes `poisson_nll_surfaces.pdf` in the current directory.

# MU-Bregman experiments

Code for the numerical experiments accompanying the MU-Bregman article. The
`python/src` package contains the reconstruction algorithms, proximal operators,
plotting helpers, and BrainWeb data preparation. The scripts in
`python/notebooks` generate the experiment figures.

## Setup

The included `pixi.toml` and `pixi.lock` reproduce the Linux CUDA 12.9
environment used for the experiments. Install [Pixi](https://pixi.sh/), then run:

```sh
pixi install
pixi run test
```

The environment pins DeepInv to a specific Git commit. The deconvolution
scripts use DeepInv's example butterfly image. The PET scripts download the
BrainWeb data through DeepInv on first use and cache it under `.pixi`.

## Experiments

Run commands from the repository root:

| Command | Experiment | Output |
| --- | --- | --- |
| `pixi run deconvolution-tv-stability` | TV stability sweep | `latex/figures/deconvolution_tv_stability_*.pdf` |
| `pixi run deconvolution-1` | Deconvolution comparison, setting 1 | `latex/figures/deconvolution_1.pdf`, `deconvolution_metrics.pdf` |
| `pixi run deconvolution-2` | Deconvolution comparison, setting 2 | `latex/figures/deconvolution_2.pdf`, `deconvolution_metrics_2.pdf` |
| `pixi run pet-2d` | BrainWeb 2D PET comparison and parameter sweep | `latex/figures/pet_1.pdf`, `pet_metrics.pdf`, `pet_regularization_sweep.pdf` |
| `pixi run pet-3d` | Additional BrainWeb 3D PET comparison | Interactive plots |
| `pixi run non-convexity` | Poisson objective surface | `poisson_nll_surfaces.pdf` |

The scripts run full sweeps and can take substantial time, especially for PET.
Generated figures and downloaded data are excluded from Git.

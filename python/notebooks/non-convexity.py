# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")


# -----------------------------
# Poisson negative log-likelihood
# -----------------------------
# Forward model: y ~ Poisson(A x), with x in R_+^2
# NLL (up to additive const): f(x) = sum_i [ (Ax)_i - y_i log((Ax)_i) ]
def poisson_nll_x(x1, x2, A, y, eps=1e-12):
    X = np.stack([x1, x2], axis=-1)  # (..., 2)
    lam = X @ A.T  # (..., m)
    return np.sum(lam - y * np.log(lam), axis=-1)  # (...)


# Change of variables: x = ∇φ*(η) = exp(η) (elementwise), η in R^2
def poisson_nll_eta(eta1, eta2, A, y, eps=1e-12):
    x1 = np.exp(eta1)
    x2 = np.exp(eta2)
    return poisson_nll_x(x1, x2, A, y, eps=eps)


# # -----------------------------
# # Non-separable mixing A, synthetic y
# # -----------------------------
# A = np.array(
#     [
#         [1.00, 0.70],
#         [0.60, 1.30],
#         [1.20, 0.40],
#     ],
#     dtype=float,
# )  # (m=3, n=2), mixes coordinates and is not separable

# y = np.array([8.0, 3.0, 10.0])  # counts (can be non-integers for visualization)
# -----------------------------
# 2D -> 2D (m=2, n=2) non-separable mixing A, synthetic y
# -----------------------------
A = np.array(
    [
        [1, 1],
    ],
    dtype=float,
)  # (m=2, n=2), mixes coordinates (not separable)

y = np.array([10])  # counts (can be non-integers for visualization)


# -----------------------------
# Helper: finite-diff Hessian for g(η) to detect slight non-convexity
# -----------------------------
def hessian_fd(g, eta1, eta2, h=1e-2):
    """
    Finite-difference Hessian of g at (eta1, eta2).
    g should accept (eta1, eta2) arrays/scalars and return same-shaped array.
    """
    e1p, e1m = eta1 + h, eta1 - h
    e2p, e2m = eta2 + h, eta2 - h

    g00 = g(eta1, eta2)
    gpp = g(e1p, e2p)
    gpm = g(e1p, e2m)
    gmp = g(e1m, e2p)
    gmm = g(e1m, e2m)

    g10p = g(e1p, eta2)
    g10m = g(e1m, eta2)
    g01p = g(eta1, e2p)
    g01m = g(eta1, e2m)

    d11 = (g10p - 2 * g00 + g10m) / (h**2)
    d22 = (g01p - 2 * g00 + g01m) / (h**2)
    d12 = (gpp - gpm - gmp + gmm) / (4 * h**2)

    return d11, d12, d22  # Hessian = [[d11, d12],[d12, d22]]


# -----------------------------
# Generate data for both plots
# -----------------------------
x1_min, x1_max = 1e-8, 100
x2_min, x2_max = 1e-8, 100
nx = 700

x1 = np.linspace(x1_min, x1_max, nx)
x2 = np.linspace(x2_min, x2_max, nx)
X1, X2 = np.meshgrid(x1, x2, indexing="xy")
F = poisson_nll_x(X1, X2, A, y)

eta1_min, eta1_max = np.log(x1_min), np.log(x1_max)
eta2_min, eta2_max = np.log(x2_min), np.log(x2_max)
ne = 700

eta1 = np.linspace(eta1_min, eta1_max, ne)
eta2 = np.linspace(eta2_min, eta2_max, ne)
E1, E2 = np.meshgrid(eta1, eta2, indexing="xy")
G = poisson_nll_eta(E1, E2, A, y)


# Finite-diff Hessian eigenvalues on the grid (vectorized computation)
def G_func(a, b):
    return poisson_nll_eta(a, b, A, y)


d11, d12, d22 = hessian_fd(G_func, E1, E2, h=2e-2)

# Eigenvalues of 2x2 symmetric matrix [[d11, d12],[d12, d22]]
trace = d11 + d22
det = d11 * d22 - d12**2
disc = np.maximum(trace**2 - 4 * det, 0.0)
lam_min = 0.5 * (trace - np.sqrt(disc))

# Mask of "non-convex" regions (lam_min < 0)
nonconvex = lam_min < -1e-4  # small tolerance to ignore numerical noise

# -----------------------------
# Side-by-side 3D surface plots
# -----------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

pane_color = (0.95, 0.95, 0.95, 0.5)  # light grey
# Set matplotlib to use LaTeX font

fig = plt.figure(figsize=(20, 8), facecolor="white")

# Left plot: f(x) surface
ax1 = fig.add_subplot(121, projection="3d")
step = 2
surf1 = ax1.plot_surface(
    X1[::step, ::step],
    X2[::step, ::step],
    F[::step, ::step],
    cmap="plasma",
    linewidth=0,
    antialiased=True,
)
ax1.set_xlabel(r"$x_1$", fontsize=20)
ax1.set_ylabel(r"$x_2$", fontsize=20)
ax1.zaxis.set_label_position("upper")
ax1.view_init(elev=30, azim=-20)
ax1.set_title(r"$f(x)$", fontsize=30)

# Right plot: g(η) surface
ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(
    E1[::step, ::step],
    E2[::step, ::step],
    G[::step, ::step],
    cmap="plasma",
    linewidth=0,
    antialiased=True,
)
ax2.set_xlabel(r"$\eta_1$", fontsize=20)
ax2.set_ylabel(r"$\eta_2$", fontsize=20)
ax2.zaxis.set_label_position("upper")
ax2.view_init(elev=30, azim=-20)
ax2.set_title(
    r"$f \circ \nabla \varphi^* (\eta)$",
    fontsize=30,
)
for ax in (ax1, ax2):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "set_rotate_label"):
            axis.set_rotate_label(False)
for ax in (ax1, ax2):
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
# Add single shared colorbar
# fig.colorbar(
#     surf2,
#     ax=[ax1, ax2],
#     shrink=0.8,
#     aspect=20,
# )
plt.savefig("poisson_nll_surfaces.pdf", format="pdf", bbox_inches="tight", dpi=96)
plt.show()
# %%
# -----------------------------
# Single diagonal slice: from (0,0) to (1,1) direction
# -----------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Create parametric line from origin in direction (1,1)
t_values = np.linspace(0, 50, 200)  # Parameter t

# Left plot: f(x) along diagonal x = t*(1,1)
x1_diag = t_values
x2_diag = t_values
f_diag = poisson_nll_x(x1_diag, x2_diag, A, y)

ax1.plot(t_values, f_diag, "b-", linewidth=2, label="Diagonal slice (1,1)")
ax1.set_xlabel(r"$t$ (where $x = t \cdot (1,1)$)")
ax1.set_ylabel(r"$f(t, t)$")
ax1.set_title("Original function f(x): Diagonal slice")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right plot: g(η) along diagonal η = t*(1,1)
t_values_eta = np.linspace(-50, 4, 500)  # Parameter t for eta range
eta1_diag = t_values_eta
eta2_diag = t_values_eta
g_diag = poisson_nll_eta(eta1_diag, eta2_diag, A, y)

ax2.plot(t_values_eta, g_diag, "r-", linewidth=2, label="Diagonal slice (1,1)")
ax2.set_xlabel(r"$t$ (where $\eta = t \cdot (1,1)$)")
ax2.set_ylabel(r"$g(t, t)$")
ax2.set_title("Transformed function g(η): Diagonal slice")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# -----------------------------
# Flow-like visualization with rotating projection vectors
# -----------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Define rotation angles
n_angles = 20
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
colors = plt.cm.viridis(np.linspace(0, 1, n_angles))

# Parameter range for the projections
t_range = np.linspace(0.1, 10, 300)

# Left plot: f(x) - rotating direction vectors
for i, angle in enumerate(angles):
    # Unit direction vector
    dir_x1 = np.cos(angle)
    dir_x2 = np.sin(angle)

    # Parametric line: x = t * (cos(θ), sin(θ))
    x1_line = t_range * dir_x1
    x2_line = t_range * dir_x2

    # Ensure we stay in positive domain
    valid_mask = (x1_line > x1_min) & (x2_line > x2_min)
    if np.any(valid_mask):
        x1_valid = x1_line[valid_mask]
        x2_valid = x2_line[valid_mask]
        f_slice = poisson_nll_x(x1_valid, x2_valid, A, y)
        ax1.plot(x1_valid, f_slice, color=colors[i], alpha=0.7, linewidth=1.5)

ax1.set_xlabel(r"$x_1$")
ax1.set_ylabel(r"$f(x_1, x_2)$ along ray")
ax1.set_title("Original function f(x): Rotating rays from origin")
ax1.grid(True, alpha=0.3)

# Add colorbar for angles
sm1 = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=2 * np.pi))
sm1.set_array([])
cbar1 = plt.colorbar(sm1, ax=ax1)
cbar1.set_label("Angle (radians)")

# Right plot: g(η) - rotating direction vectors
t_range_eta = np.linspace(-8, 3, 300)

for i, angle in enumerate(angles):
    # Unit direction vector
    dir_eta1 = np.cos(angle)
    dir_eta2 = np.sin(angle)

    # Parametric line: η = t * (cos(θ), sin(θ))
    eta1_line = t_range_eta * dir_eta1
    eta2_line = t_range_eta * dir_eta2

    # Keep within reasonable bounds
    valid_mask = (
        (eta1_line >= eta1_min)
        & (eta1_line <= eta1_max)
        & (eta2_line >= eta2_min)
        & (eta2_line <= eta2_max)
    )
    if np.any(valid_mask):
        eta1_valid = eta1_line[valid_mask]
        eta2_valid = eta2_line[valid_mask]
        g_slice = poisson_nll_eta(eta1_valid, eta2_valid, A, y)
        ax2.plot(eta1_valid, g_slice, color=colors[i], alpha=0.7, linewidth=1.5)

ax2.set_xlabel(r"$\eta_1$")
ax2.set_ylabel(r"$g(\eta_1, \eta_2)$ along ray")
ax2.set_title("Transformed function g(η): Rotating rays from origin")
ax2.grid(True, alpha=0.3)

# Add colorbar for angles
sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=2 * np.pi))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2)
cbar2.set_label("Angle (radians)")

plt.tight_layout()
plt.show()

# %%

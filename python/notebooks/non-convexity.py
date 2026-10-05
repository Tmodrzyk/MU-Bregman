# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")

# -----------------------------
# Poisson negative log-likelihood
# -----------------------------
def poisson_nll_x(x1, x2, A, y, eps=1e-12):
    X = np.stack([x1, x2], axis=-1)
    lam = X @ A.T
    return np.sum(lam - y * np.log(lam), axis=-1)


def poisson_nll_eta(eta1, eta2, A, y, eps=1e-12):
    x1 = np.exp(eta1)
    x2 = np.exp(eta2)
    return poisson_nll_x(x1, x2, A, y, eps=eps)


# One measurement mixing two positive coordinates.
A = np.array(
    [
        [1, 1],
    ],
    dtype=float,
)

y = np.array([10])

# Generate the original and transformed surfaces.
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


pane_color = (0.95, 0.95, 0.95, 0.5)  # light grey
# Set matplotlib to use LaTeX font

fig = plt.figure(figsize=(20, 8), facecolor="white")

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
plt.savefig("poisson_nll_surfaces.pdf", format="pdf", bbox_inches="tight", dpi=96)
plt.show()

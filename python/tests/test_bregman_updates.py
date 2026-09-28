"""Numerical checks for the proximal formulas and Poisson convergence identities.

Run with: python -m unittest discover -s python/tests -v
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.algos import mlem, mlem_L1, mlem_L2
from src.prox import (
    _torch_gradient_adjoint,
    left_prox_L2_burg_entropy,
    right_prox_L2_negative_entropy,
    torch_gradient,
)


class MatrixPhysics:
    def __init__(self, matrix):
        self.matrix = matrix

    def A(self, x):
        return self.matrix @ x

    def A_adjoint(self, y):
        return self.matrix.T @ y


class BregmanUpdatesTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([0.03, 0.7, 2.0], dtype=torch.float64)
        self.s = torch.tensor([0.2, 1.3, 4.0], dtype=torch.float64)

    def test_l2_prox_stationarity_and_zero_weight(self):
        for weight in (0.0, 1e-12, 0.3, 10.0):
            u = right_prox_L2_negative_entropy(self.x, weight, self.s)
            torch.testing.assert_close(
                weight * u + self.s * (1 - self.x / u),
                torch.zeros_like(u),
                atol=1e-12,
                rtol=0,
            )
            v = left_prox_L2_burg_entropy(self.x, weight)
            torch.testing.assert_close(
                weight * v + 1 / self.x - 1 / v,
                torch.zeros_like(v),
                atol=1e-12,
                rtol=0,
            )
        torch.testing.assert_close(
            right_prox_L2_negative_entropy(self.x, 0, self.s),
            self.x,
        )

    def test_relaxed_updates_satisfy_proximal_optimality(self):
        physics = MatrixPhysics(torch.diag(self.s))
        y = torch.tensor([0.1, 2.0, 5.0], dtype=torch.float64)
        for tau in (0.0, 0.25, 1.0):
            z = (1 - tau) * self.x + tau * y / self.s
            for algorithm, power in ((mlem_L1, 0), (mlem_L2, 1)):
                u = algorithm(y, self.x, tau, physics, 0.4, 1, verbose=False)
                residual = self.s * (u - z) + tau * 0.4 * u ** (power + 1)
                torch.testing.assert_close(
                    residual, torch.zeros_like(u), atol=1e-12, rtol=0
                )
                unregularized = algorithm(
                    y, self.x, tau, physics, 0.0, 1, verbose=False
                )
                reference = mlem(y, self.x, tau, physics, 1, verbose=False)
                torch.testing.assert_close(unregularized, reference)

    def test_boundary_minimizer_kl_rate_and_mass_invariant(self):
        # Constructed KKT solution: x*=(1,0), xi*=0.2, nu*=(0,1).
        physics = MatrixPhysics(
            torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        )
        y = torch.tensor([2.8, 0.4], dtype=torch.float64)
        xstar = torch.tensor([1.0, 0.0], dtype=torch.float64)
        initial = torch.tensor([0.2, 2.0], dtype=torch.float64)
        weight = 0.2
        s = physics.A_adjoint(torch.ones_like(y))
        _, history = mlem_L1(
            y, initial, 1.0, physics, weight, 80, verbose=False, keep_inter=True
        )

        def objective(x):
            ax = physics.A(x)
            return (ax - y * ax.log()).sum() + weight * x.sum()

        p = xstar * (s + weight)
        energies = []
        gaps = []
        previous = objective(initial)
        for x in history[1:]:
            q = x * (s + weight)
            torch.testing.assert_close(q.sum(), y.sum())
            value = objective(x)
            self.assertLessEqual(float(value - previous), 1e-12)
            previous = value
            energies.append((torch.special.xlogy(p, p / q) - p + q).sum())
            gaps.append(value - objective(xstar))
        for k, gap in enumerate(gaps, 1):
            self.assertLessEqual(float(k * gap - energies[0]), 1e-11)
        for k in range(len(energies) - 1):
            self.assertLessEqual(float(gaps[k] - energies[k] + energies[k + 1]), 1e-11)

    def test_poisson_dual_hessian_and_sharp_constant(self):
        matrix = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([3.0, 2.0], dtype=torch.float64)
        sensitivity = matrix.sum(0)
        eta = sensitivity * self.x.log()

        def dual_objective(v):
            prediction = matrix @ (v / sensitivity).exp()
            return (prediction - y * prediction.log()).sum()

        dual_hessian = torch.autograd.functional.hessian(dual_objective, eta)
        phi_dual_hessian = torch.diag(self.x / sensitivity)
        gap = phi_dual_hessian - dual_hessian
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(gap).min()), -1e-12)
        # The sensitivity direction attains equality, so L cannot be below 1.
        torch.testing.assert_close(
            gap @ sensitivity, torch.zeros_like(sensitivity), atol=1e-12, rtol=0
        )

    def test_tv_gradient_adjoint(self):
        generator = torch.Generator().manual_seed(4)
        for shape in ((1, 1, 1, 2), (1, 1, 4, 5), (1, 1, 2, 3, 4)):
            x = torch.randn(shape, generator=generator, dtype=torch.float64)
            dx = torch_gradient(x)
            q = torch.randn(dx.shape, generator=generator, dtype=torch.float64)
            torch.testing.assert_close(
                (dx * q).sum(), (x * _torch_gradient_adjoint(q)).sum()
            )


if __name__ == "__main__":
    unittest.main()

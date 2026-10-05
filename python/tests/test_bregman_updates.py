"""Small numerical checks for the experiment algorithms and proximal maps."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.algos import mirror_descent, mu, rbpg_l1, rbpg_l2
from src.prox import lprox_burg_l2, rprox_negentropy_l2, rprox_negentropy_TV


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
        self.y = torch.tensor([0.1, 2.0, 5.0], dtype=torch.float64)
        self.physics = MatrixPhysics(torch.diag(self.s))

    def test_l2_prox_stationarity_and_zero_weight(self):
        for weight in (0.0, 1e-12, 0.3, 10.0):
            u = rprox_negentropy_l2(self.x, weight, self.s)
            torch.testing.assert_close(
                weight * u + self.s * (1 - self.x / u),
                torch.zeros_like(u),
                atol=1e-12,
                rtol=0,
            )
            v = lprox_burg_l2(self.x, weight)
            torch.testing.assert_close(
                weight * v + 1 / self.x - 1 / v,
                torch.zeros_like(v),
                atol=1e-12,
                rtol=0,
            )
        torch.testing.assert_close(rprox_negentropy_l2(self.x, 0, self.s), self.x)

    def test_relaxed_updates_satisfy_proximal_optimality(self):
        for tau in (0.0, 0.25, 1.0):
            z = (1 - tau) * self.x + tau * self.y / self.s
            for algorithm, power in ((rbpg_l1, 0), (rbpg_l2, 1)):
                result = algorithm(
                    self.y, self.x, tau, self.physics, 0.4, 1, verbose=False
                )
                u = result[0] if isinstance(result, tuple) else result
                residual = self.s * (u - z) + tau * 0.4 * u ** (power + 1)
                torch.testing.assert_close(
                    residual, torch.zeros_like(u), atol=1e-12, rtol=0
                )
                unregularized = algorithm(
                    self.y, self.x, tau, self.physics, 0.0, 1, verbose=False
                )
                reference, _ = mu(
                    self.y, self.x, tau, self.physics, 1, verbose=False
                )
                u0 = unregularized[0] if isinstance(unregularized, tuple) else unregularized
                torch.testing.assert_close(u0, reference)

    def test_metric_history_and_descent(self):
        for algorithm in (mu, mirror_descent):
            reconstruction, metrics = algorithm(
                self.y,
                self.x,
                0.5,
                self.physics,
                5,
                verbose=False,
                keep_inter=True,
                x_ref=self.y / self.s,
            )
            self.assertEqual(len(metrics["objective"]), 6)
            self.assertEqual(len(metrics["iterates"]), 6)
            self.assertEqual(len(metrics["nrmse"]), 6)
            self.assertLessEqual(metrics["objective"][-1], metrics["objective"][0])
            self.assertTrue(torch.isfinite(reconstruction).all())

    def test_zero_weight_tv_prox_is_identity(self):
        image = self.x.reshape(1, 1, 1, 3)
        sensitivity = self.s.reshape(1, 1, 1, 3)
        torch.testing.assert_close(
            rprox_negentropy_TV(image, 0, sensitivity), image
        )


if __name__ == "__main__":
    unittest.main()

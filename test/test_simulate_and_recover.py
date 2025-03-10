import unittest
import numpy as np
from src.simulate_and_recover import generate_parameters, forward_equations, simulate_observed_statistics, inverse_equations

class TestEZDiffusionModel(unittest.TestCase):
    def test_generate_parameters(self):
        alpha, nu, tau = generate_parameters()
        self.assertTrue(0.5 <= alpha <= 2)
        self.assertTrue(0.5 <= nu <= 2)
        self.assertTrue(0.1 <= tau <= 0.5)

    def test_forward_equations(self):
        alpha, nu, tau = 1.2, 1.5, 0.3
        R_pred, M_pred, V_pred = forward_equations(alpha, nu, tau)
        self.assertTrue(0 <= R_pred <= 1)
        self.assertTrue(M_pred > tau)
        self.assertTrue(V_pred > 0)

    def test_inverse_equations(self):
        alpha, nu, tau = 1.2, 1.5, 0.3
        R_pred, M_pred, V_pred = forward_equations(alpha, nu, tau)
        R_obs, M_obs, V_obs = simulate_observed_statistics(R_pred, M_pred, V_pred, 1000)
        alpha_est, nu_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)
        self.assertAlmostEqual(alpha, alpha_est, delta=0.2)
        self.assertAlmostEqual(nu, nu_est, delta=0.2)
        self.assertAlmostEqual(tau, tau_est, delta=0.1)

if __name__ == "__main__":
    unittest.main()

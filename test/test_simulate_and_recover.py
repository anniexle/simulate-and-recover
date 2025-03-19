import unittest
import numpy as np
from src.simulate_and_recover import SimulateAndRecover

class TestEZDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.simulator = SimulateAndRecover(iterations=10)  # Fewer iterations for faster testing

    def test_generate_parameters(self):
        alpha, nu, tau = self.simulator.get_model_parameters()
        self.assertTrue(0.5 <= alpha <= 2, "Boundary separation out of range")
        self.assertTrue(0.5 <= nu <= 2, "Drift rate out of range")
        self.assertTrue(0.1 <= tau <= 0.5, "Non-decision time out of range")

    def test_forward_equations(self):
        alpha, nu, tau = 1.2, 1.5, 0.3
        R_pred, M_pred, V_pred = self.simulator.predicted_parameters(alpha, nu, tau)

        self.assertTrue(0 <= R_pred <= 1, "Predicted accuracy R_pred is invalid")
        self.assertTrue(M_pred > tau, "Predicted mean RT M_pred should be greater than tau")
        self.assertTrue(V_pred > 0, "Predicted variance V_pred should be positive")

    def test_observed_statistics(self):
        alpha, nu, tau = 1.2, 1.5, 0.3
        R_pred, M_pred, V_pred = self.simulator.predicted_parameters(alpha, nu, tau)
        N = 1000  # Large enough sample size for stable estimates

        t_obs, R_obs, V_obs, M_obs = self.simulator.obs_parameters(N, R_pred, M_pred, V_pred)

        self.assertIsInstance(t_obs, int, "t_obs should be an integer (Binomial sample)")
        self.assertTrue(0 <= R_obs <= 1, "Observed accuracy R_obs is invalid")
        self.assertTrue(V_obs > 0, "Observed variance V_obs should be positive")
        self.assertTrue(M_obs > 0, "Observed mean RT M_obs should be positive")

    def test_inverse_equations(self):
        alpha, nu, tau = 1.2, 1.5, 0.3
        R_pred, M_pred, V_pred = self.simulator.predicted_parameters(alpha, nu, tau)
        N = 1000

        t_obs, R_obs, V_obs, M_obs = self.simulator.obs_parameters(N, R_pred, M_pred, V_pred)
        nu_est, alpha_est, tau_est = self.simulator.inverse_eq(R_obs, V_obs, M_obs)

        self.assertAlmostEqual(alpha, alpha_est, delta=0.2, msg="Recovered alpha is outside tolerance")
        self.assertAlmostEqual(nu, nu_est, delta=0.2, msg="Recovered drift rate is outside tolerance")
        self.assertAlmostEqual(tau, tau_est, delta=0.1, msg="Recovered tau is outside tolerance")

    def test_full_simulation(self):
        N = 10  # Smallest sample size to check for stability
        bias_avg, sq_error_avg = self.simulator.n_simulations(N)
        self.assertEqual(len(bias_avg), 3, "Bias array should have length 3")
        self.assertEqual(len(sq_error_avg), 3, "Squared error array should have length 3")
        self.assertTrue(np.all(np.abs(bias_avg) < 0.1), "Bias values should be close to zero")
        self.assertTrue(np.all(sq_error_avg < 0.01), "Squared errors should be very small")

if __name__ == "__main__":
    unittest.main()

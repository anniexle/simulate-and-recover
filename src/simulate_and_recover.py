import numpy as np
import random
import math
import json

class SimulateAndRecover:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def get_model_parameters(self):
        boundary_sep = random.uniform(0.5, 2)
        drift_rt = random.uniform(0.5, 2)
        nondec = random.uniform(0.1, 0.5)
        return boundary_sep, drift_rt, nondec

    def predicted_parameters(self, boundary_sep, drift_rt, nondec):
        y = math.exp(-boundary_sep * drift_rt)
        r_pred = 1 / (y + 1)
        m_pred = nondec + ((boundary_sep / (2 * drift_rt)) * ((1 - y) / (1 + y)))
        v_pred = (boundary_sep / (2 * (math.pow(drift_rt, 3)))) * ((1 - (2 * boundary_sep * drift_rt * y) - (math.pow(y, 2))) / math.pow((y + 1), 2))
        return r_pred, m_pred, v_pred

    def obs_parameters(self, N, r_pred, m_pred, v_pred):
        epsilon = 1e-8  # Small constant for numerical stability
        t_obs = np.random.binomial(N, r_pred)  # Binomial distribution for R_obs
        r_obs = t_obs / N
        v_obs = np.random.gamma(shape=(N - 1) / 2, scale=(2 * v_pred) / (N - 1))
        std_dev = np.sqrt(v_pred / N)
        m_obs = np.random.normal(loc=m_pred, scale=std_dev)

        return t_obs, r_obs, v_obs, m_obs

    def inverse_eq(self, r_obs, v_obs, m_obs):
        epsilon = 1e-8  # Small constant to avoid division errors
        L = math.log(r_obs / (1 - r_obs)) if r_obs != 1 else 1  # Prevent log(0) issue
        L_sq_robs = L * math.pow(r_obs, 2)
        L_robs = r_obs * L
        v_est = np.sign(r_obs - 0.5) * math.pow(L * (L_sq_robs - L_robs + r_obs - 0.5) / (v_obs + epsilon), 1 / 4)
        a_est = L / v_est
        term1 = a_est / (2 * v_est)
        term2 = -(v_est) * (a_est)
        t_est = m_obs - ((term1) * ((1 - math.exp(term2)) / (1 + math.exp(term2))))

        return v_est, a_est, t_est

    def n_simulations(self, N):
        biases = []
        squared_errors = []

        for _ in range(self.iterations):
            boundary_sep, drift_rt, nondec = self.get_model_parameters()
            r_pred, m_pred, v_pred = self.predicted_parameters(boundary_sep, drift_rt, nondec)
            t_obs, r_obs, v_obs, m_obs = self.obs_parameters(N, r_pred, m_pred, v_pred)
            v_est, a_est, t_est = self.inverse_eq(r_obs, v_obs, m_obs)

            bias = np.array([drift_rt - v_est, boundary_sep - a_est, nondec - t_est])
            biases.append(bias)
            squared_errors.append(np.square(bias))

        bias_avg = np.nanmean(biases, axis=0)
        sq_error_avg = np.nanmean(squared_errors, axis=0)

        return bias_avg, sq_error_avg

def main():
    obj = SimulateAndRecover(iterations=1000)

    print("N   Biases   Squared Errors")
    results = {}
    for N in [10, 40, 4000]:
        bias_avg, sq_error_avg = obj.n_simulations(N)
        results[N] = {
            "biases": bias_avg.tolist(),
            "squared_errors": sq_error_avg.tolist()
        }
        print(N, " ", bias_avg, " ", sq_error_avg)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()

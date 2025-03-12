import numpy as np
import scipy.stats as stats
import json

def generate_parameters():
    alpha = np.random.uniform(0.5, 2)  # Boundary separation
    nu = np.random.uniform(0.5, 2)  # Drift rate
    tau = np.random.uniform(0.1, 0.5)  # Non-decision time
    return alpha, nu, tau

def forward_equations(alpha, nu, tau):
    y = np.exp(-alpha * nu)
    R_pred = 1 / (y + 1)
    M_pred = tau + (alpha / (2 * nu)) * ((1 - y) / (1 + y))
    V_pred = (alpha / (2 * nu**3)) * (1 - 2 * alpha * nu * y - y**2) / (y + 1)**2
    return R_pred, M_pred, V_pred

def simulate_observed_statistics(R_pred, M_pred, V_pred, N):
    R_obs = np.random.binomial(N, R_pred) / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    V_obs = np.random.gamma((N - 1) / 2, 2 * V_pred / (N - 1))
    return R_obs, M_obs, V_obs

def inverse_equations(R_obs, M_obs, V_obs):
    epsilon = 1e-6 
    R_obs = np.clip(R_obs, epsilon, 1 - epsilon)
    L = np.log(R_obs / (1 - R_obs))
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs)

    if np.isnan(v_est) or v_est == 0:
        return np.nan, np.nan, np.nan
    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
    return a_est, v_est, t_est

def simulate_and_recover(N, iterations=1000):
    biases = []
    squared_errors = []

    for _ in range(iterations):
        alpha, nu, tau = generate_parameters()
        R_pred, M_pred, V_pred = forward_equations(alpha, nu, tau)
        R_obs, M_obs, V_obs = simulate_observed_statistics(R_pred, M_pred, V_pred, N)
        alpha_est, nu_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)

        bias = np.array([alpha_est - alpha, nu_est - nu, tau_est - tau])
        squared_error = bias**2

        biases.append(bias)
        squared_errors.append(squared_error)

    biases = np.mean(biases, axis=0)
    squared_errors = np.mean(squared_errors, axis=0)

    return biases, squared_errors

def main():
    results = {}
    for N in [10, 40, 4000]:
        biases, squared_errors = simulate_and_recover(N)
        results[N] = {
            "biases": biases.tolist(),
            "squared_errors": squared_errors.tolist()
        }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
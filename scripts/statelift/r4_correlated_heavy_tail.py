"""
R4: State-lift regret bound under correlated perturbations and heavy-tailed noise.

Setup:
  - K=5 actions. In each trial, a latent state s is drawn.
  - Each action a has expected reward mu_a(s) = mu_a + beta_a * s, where
    beta_a are action-specific state sensitivities.
  - State-lift (SL) = fraction of reward variance explained by state.
  - An oracle that observes state picks the best action per-state.
  - A state-ignorant agent picks the action with highest marginal mean.
  - Regret = E[R_oracle] - E[R_ignorant].
  - Bound: sqrt(SL * Var(R)) * sqrt(2 ln K).

We test whether this bound holds under:
  1. Gaussian iid noise
  2. Correlated state-dependent perturbations (rho = 0.3, 0.6, 0.9)
  3. Heavy-tailed noise (t-distribution, df=3, df=5)
"""

import numpy as np
from scipy import stats
import json
import time
import os


def build_action_params(K, state_lift, total_var=1.0):
    """
    Create K actions with specified state-lift.

    state_lift = Var(beta_a * s) / Var(reward_a) = beta_a^2 * var_s / total_var

    We set var_s = 1 and choose betas so that the average state-lift across
    actions equals the target. Residual variance = total_var - beta_a^2.

    To make it interesting, betas vary across actions so the optimal action
    changes with state.
    """
    # Base means spread out
    base_means = np.linspace(0, 1, K)

    # Betas: spread from -sqrt(SL) to +sqrt(SL) so average beta^2 ~ SL
    # This ensures the best action changes with state
    beta_spread = np.sqrt(state_lift * total_var * 3)  # scale for uniform-like spread
    betas = np.linspace(-beta_spread, beta_spread, K)

    # Actual state-dep variance per action = beta_a^2 * var_s (var_s=1)
    state_var_per_action = betas ** 2

    # Residual variance per action
    residual_var = np.maximum(total_var - state_var_per_action, 0.01)

    return base_means, betas, residual_var


def simulate_regret_gaussian_iid(state_lift, K, n_trials, rng):
    """Baseline: iid Gaussian noise."""
    base_means, betas, residual_var = build_action_params(K, state_lift)

    # Draw states
    states = rng.normal(0, 1, size=n_trials)

    # True expected reward per action per trial (if we knew state)
    # mu_a(s) = base_means[a] + betas[a] * s
    expected_rewards = base_means[None, :] + betas[None, :] * states[:, None]

    # Oracle picks best action per state
    oracle_choices = np.argmax(expected_rewards, axis=1)
    oracle_reward = expected_rewards[np.arange(n_trials), oracle_choices]

    # State-ignorant agent: pick action with highest marginal mean
    # Marginal mean = base_means (since E[s]=0)
    ignorant_choice = np.argmax(base_means)
    ignorant_reward = expected_rewards[:, ignorant_choice]

    # Regret = oracle - ignorant (both in expectation over noise, so use expected rewards)
    regret = np.mean(oracle_reward - ignorant_reward)

    # Empirical reward variance (for bound computation)
    # Add noise to compute total variance
    noise = rng.normal(0, 1, size=(n_trials, K)) * np.sqrt(residual_var)[None, :]
    realized_rewards = expected_rewards + noise
    reward_var = np.mean(np.var(realized_rewards, axis=0))

    return regret, reward_var


def simulate_regret_correlated(state_lift, K, n_trials, rho, rng):
    """Correlated state-dependent perturbations.

    Instead of betas * s (rank-1), we use a correlated multivariate normal
    for the state-dependent component with equicorrelation rho.
    """
    base_means, betas, residual_var = build_action_params(K, state_lift)

    # Draw states
    states = rng.normal(0, 1, size=n_trials)

    # State-dependent component: beta_a * s + correlated_perturbation
    # The correlated perturbation has equicorrelation rho across actions
    common = rng.normal(0, 1, size=(n_trials, 1))
    individual = rng.normal(0, 1, size=(n_trials, K))

    # Scale perturbation so it adds correlation without changing total state variance much
    pert_scale = 0.3 * np.sqrt(state_lift)  # perturbation magnitude
    corr_pert = pert_scale * (np.sqrt(rho) * common + np.sqrt(1 - rho) * individual)

    expected_rewards = base_means[None, :] + betas[None, :] * states[:, None] + corr_pert

    oracle_choices = np.argmax(expected_rewards, axis=1)
    oracle_reward = expected_rewards[np.arange(n_trials), oracle_choices]

    ignorant_choice = np.argmax(base_means)
    ignorant_reward = expected_rewards[:, ignorant_choice]

    regret = np.mean(oracle_reward - ignorant_reward)

    noise = rng.normal(0, 1, size=(n_trials, K)) * np.sqrt(residual_var)[None, :]
    realized_rewards = expected_rewards + noise
    reward_var = np.mean(np.var(realized_rewards, axis=0))

    return regret, reward_var


def simulate_regret_heavy_tail(state_lift, K, n_trials, df, rng):
    """Heavy-tailed state and noise (t-distributed)."""
    base_means, betas, residual_var = build_action_params(K, state_lift)

    # Draw states from t-distribution (scaled to unit variance)
    t_scale = np.sqrt((df - 2) / df) if df > 2 else 1.0
    states = rng.standard_t(df, size=n_trials) * t_scale

    expected_rewards = base_means[None, :] + betas[None, :] * states[:, None]

    oracle_choices = np.argmax(expected_rewards, axis=1)
    oracle_reward = expected_rewards[np.arange(n_trials), oracle_choices]

    ignorant_choice = np.argmax(base_means)
    ignorant_reward = expected_rewards[:, ignorant_choice]

    regret = np.mean(oracle_reward - ignorant_reward)

    # Heavy-tailed residual noise too
    noise = rng.standard_t(df, size=(n_trials, K)) * t_scale * np.sqrt(residual_var)[None, :]
    realized_rewards = expected_rewards + noise
    reward_var = np.mean(np.var(realized_rewards, axis=0))

    return regret, reward_var


def gaussian_upper_bound(state_lift, K, reward_var=1.0):
    """Gaussian upper bound: sqrt(SL * Var(R)) * sqrt(2 ln K)."""
    return np.sqrt(state_lift * reward_var) * np.sqrt(2 * np.log(K))


def main():
    K = 5
    n_trials = 20000
    rng = np.random.default_rng(42)

    sl_values = np.arange(0.01, 0.91, 0.01)
    sl_values = np.round(sl_values, 2)

    conditions = {
        "gaussian_iid": {},
        "correlated_rho_0.3": {"rho": 0.3},
        "correlated_rho_0.6": {"rho": 0.6},
        "correlated_rho_0.9": {"rho": 0.9},
        "t_dist_df_3": {"df": 3},
        "t_dist_df_5": {"df": 5},
    }

    results = {}

    for cond_name, params in conditions.items():
        print(f"Running condition: {cond_name}")
        t0 = time.time()

        regrets = []
        bounds = []
        ratios = []
        reward_vars = []
        bound_holds_list = []

        for sl in sl_values:
            if cond_name == "gaussian_iid":
                reg, rv = simulate_regret_gaussian_iid(sl, K, n_trials, rng)
            elif cond_name.startswith("correlated"):
                reg, rv = simulate_regret_correlated(sl, K, n_trials, params["rho"], rng)
            elif cond_name.startswith("t_dist"):
                reg, rv = simulate_regret_heavy_tail(sl, K, n_trials, params["df"], rng)

            bound = gaussian_upper_bound(sl, K, reward_var=rv)
            ratio = reg / bound if bound > 0 else 0
            regrets.append(float(reg))
            bounds.append(float(bound))
            ratios.append(float(ratio))
            reward_vars.append(float(rv))
            bound_holds_list.append(bool(reg <= bound))

        max_ratio = max(ratios)
        all_hold = all(bound_holds_list)
        n_violations = sum(1 for h in bound_holds_list if not h)

        results[cond_name] = {
            "regrets": regrets,
            "bounds": bounds,
            "ratios": ratios,
            "reward_vars": reward_vars,
            "bound_holds": bound_holds_list,
            "max_ratio": max_ratio,
            "all_hold": all_hold,
            "n_violations": n_violations,
            "worst_sl": float(sl_values[np.argmax(ratios)]),
        }

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s. Max ratio={max_ratio:.4f}, "
              f"Bound holds everywhere: {all_hold}, Violations: {n_violations}")

    # Summary table
    print("\n" + "="*95)
    print(f"{'Condition':<25} {'Bound Holds?':<15} {'Max Ratio':<12} {'Violations':<12} {'Worst SL':<10} {'Underest.':<10}")
    print("="*95)
    for cond_name in conditions:
        r = results[cond_name]
        holds_str = "YES" if r["all_hold"] else "NO"
        underest = f"{(r['max_ratio']-1)*100:.1f}%" if r['max_ratio'] > 1 else "N/A"
        print(f"{cond_name:<25} {holds_str:<15} {r['max_ratio']:<12.4f} {r['n_violations']:<12} {r['worst_sl']:<10.2f} {underest:<10}")
    print("="*95)

    # Detailed view at selected SL values
    print("\nDetailed regret vs bound at selected SL values:")
    print(f"{'SL':<8}", end="")
    for cond_name in conditions:
        print(f"  {cond_name[:15]:<17}", end="")
    print()
    for idx in [0, 4, 9, 19, 39, 59, 79, 89]:
        if idx >= len(sl_values):
            continue
        sl = sl_values[idx]
        print(f"{sl:<8.2f}", end="")
        for cond_name in conditions:
            r = results[cond_name]
            print(f"  {r['regrets'][idx]:.3f}/{r['bounds'][idx]:.3f}  ", end="")
        print()

    # Save
    output = {
        "metadata": {
            "K": K,
            "n_trials": n_trials,
            "sl_range": [float(sl_values[0]), float(sl_values[-1])],
            "n_sl_points": len(sl_values),
            "sl_values": [float(s) for s in sl_values],
            "bound_formula": "sqrt(SL * Var(R)) * sqrt(2 * ln(K))",
            "description": (
                "State-lift regret = E[R_oracle] - E[R_ignorant] where oracle knows "
                "latent state and ignorant picks best marginal action. "
                "Bound uses empirical reward variance."
            ),
        },
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "summary": {}
    }

    for cond_name in conditions:
        r = results[cond_name]
        output["summary"][cond_name] = {
            "bound_holds": r["all_hold"],
            "max_ratio": r["max_ratio"],
            "n_violations": r["n_violations"],
            "worst_sl": r["worst_sl"],
        }

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "r4_correlated_heavy_tail.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

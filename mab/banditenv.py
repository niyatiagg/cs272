from typing import Literal, Optional, Tuple, Dict, List, Any
import numpy as np

def frequency(arr: List[int]) -> Dict[int, int]:
    freq = {}
    for num in arr:
        if num in freq:
            freq[num] += 1
        else:
            freq[num] = 1
    return freq


class BanditEnv:
    """
    K-armed bandit environment supporting Bernoulli and Gaussian rewards.

    For 'gaussian':
        reward ~ Normal(mean_k, scale_k) on pull of arm k
        If nonstationary is True, means undergo a Gaussian random walk with std=drift_std each step.

    For 'bernoulli':
        reward ~ Bernoulli(p_k) on pull of arm k
        Nonstationarity is not applied to probabilities in this minimal version.
    """
    def __init__(
        self,
        n_arms: int,
        dist: Literal["bernoulli", "gaussian"] = "gaussian",
        means: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None, # stds for Gaussian
        nonstationary: bool = False,
        drift_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        assert n_arms >= 2, "n_arms must be at least 2"
        if dist not in ("bernoulli", "gaussian"):
            raise ValueError("dist must be 'bernoulli' or 'gaussian'")

        self.n_arms = int(n_arms)
        self.dist = dist
        self.nonstationary = bool(nonstationary)
        self.drift_std = float(drift_std)
        self.rng = np.random.default_rng(seed) # random generator

        self._init_params(means=means, scales=scales)
        self.step_count = 0

    def _init_params(self, means: Optional[np.ndarray], scales: Optional[np.ndarray]) -> None:
        if means is None:
            if self.dist == "bernoulli":
                self.means = self.rng.uniform(0.05, 0.95, size=self.n_arms) 
            else:
                self.means = self.rng.normal(0.0, 1.0, size=self.n_arms)
        else:
            means = np.asarray(means, dtype=float)
            if means.shape != (self.n_arms,):
                raise ValueError("means must have shape (n_arms,)")
            self.means = means.copy()

        if self.dist == "gaussian":
            if scales is None:
                self.scales = np.ones(self.n_arms, dtype=float)
            else:
                scales = np.asarray(scales, dtype=float)
                if scales.shape != (self.n_arms,):
                    raise ValueError("scales must have shape (n_arms,)")
                if np.any(scales <= 0):
                    raise ValueError("All Gaussian scales must be positive")
                self.scales = scales
        else:
            self.scales = np.zeros(self.n_arms, dtype=float)

        print('env type:', self.dist)
        print('means:', self.means)
        print('stds:', self.scales)

    def reset(
        self,
        means: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_params(means, scales)
        self.step_count = 0

    def best_arm(self) -> int:
        max_val = np.max(self.means)
        candidates = np.flatnonzero(self.means == max_val)
        return int(self.rng.choice(candidates))

    def step(self, action: int) -> Tuple[float, Dict[str, Any]]:
        if not (0 <= action < self.n_arms):
            raise IndexError("action out of range")

        means_before = self.means.copy()
        best_idx = int(np.argmax(means_before))
        best_mean = float(means_before[best_idx])

        if self.dist == "bernoulli":
            reward = float(self.rng.binomial(1, means_before[action]))
        else:
            reward = float(self.rng.normal(means_before[action], self.scales[action]))

        if self.dist == "gaussian" and self.nonstationary and self.drift_std > 0:
            self.means += self.rng.normal(0.0, self.drift_std, size=self.n_arms)

        self.step_count += 1
        info = {
            "means_before": means_before,
            "best_arm": best_idx,
            "best_mean": best_mean,
        }
        return reward, info


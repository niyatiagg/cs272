from banditenv import BanditEnv, frequency
import numpy as np
from typing import Optional, Dict, Any

def epsilon_greedy(
    env: BanditEnv,
    steps: int,
    epsilon: float = 0.1,
    init_value: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    epsilon-greedy with incremental mean updates:
        Q_{t+1}(a) = Q_t(a) + (R_t - Q_t(a)) / N_t(a)
    """
    assert steps > 0
    assert 0.0 <= epsilon <= 1.0

    rng = np.random.default_rng(seed)

    K = env.n_arms
    Q = np.full(K, float(init_value), dtype=float) # Return a new array of given shape and type, filled with fill_value.
    N = np.zeros(K, dtype=int)

    actions = np.empty(steps, dtype=int)
    rewards = np.empty(steps, dtype=float)
    q_values = np.empty((steps, K), dtype=float)

    for t in range(steps):
        if rng.random() < epsilon:
            a = int(rng.integers(0, K))
        else:
            maxQ = np.max(Q)
            candidates = np.flatnonzero(Q == maxQ) # returns the indices of the non-zero elements in the flattened version of an input array (basically, argmax)
            a = int(rng.choice(candidates))

        r, info = env.step(a)

        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        actions[t] = a
        rewards[t] = r
        q_values[t] = Q

    return {
        "actions": actions,
        "rewards": rewards,
        "q_values": q_values,
        "counts": N,
        "final_Q": Q,
    }


# Example usage
if __name__ == "__main__":
    env = BanditEnv(
        n_arms=10,
        dist="gaussian",
        nonstationary=False,
        seed=123
    )

    logs = epsilon_greedy(env, steps=5000, epsilon=0.1, init_value=0.0, seed=42)

    actions = logs["actions"]

    print('frequency of actions: ', frequency(actions))
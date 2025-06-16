from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .orchard_env import OrchardCoop


class Runner:
    """Simplistic rollout runner for baselines."""

    def __init__(self, env: OrchardCoop, policy):
        self.env, self.policy = env, policy

    def run_episode(self, seed: int | None = None) -> Dict[str, float]:
        obs = self.env.reset(seed=seed)
        totals = {a: 0.0 for a in self.env.agents}
        done = {a: False for a in self.env.agents}
        while not all(done.values()):
            actions = self.policy.act(obs)
            obs, rews, _, truncs, _ = self.env.step(actions)
            for a, r in rews.items():
                totals[a] += r
            done = truncs
        return totals


def evaluate(env_factory: Callable[[], OrchardCoop], policy_factory: Callable[[OrchardCoop], object], episodes: int = 5) -> Dict[str, float]:
    scores = []
    for ep in range(episodes):
        env = env_factory()
        pol = policy_factory(env)
        runner = Runner(env, pol)
        totals = runner.run_episode(seed=ep)
        scores.append(np.mean(list(totals.values())))
    return {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
    }

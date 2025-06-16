import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchardcoop import OrchardCoop, RandomPolicy, Runner


def test_random_episode_fast():
    env = OrchardCoop(grid_sz=10, n_agents=3)
    policy = RandomPolicy(env)
    runner = Runner(env, policy)
    totals = runner.run_episode(seed=0)
    assert isinstance(totals, dict)
    assert all(isinstance(v, float) for v in totals.values())

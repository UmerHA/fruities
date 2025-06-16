import numpy as np
from typing import Dict


class RandomPolicy:
    """Selects random actions using env action space."""

    def __init__(self, env):
        self.env = env

    def act(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {a: self.env.action_space(a).sample() for a in self.env.agents}

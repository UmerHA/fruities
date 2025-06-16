from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


MOVE_DIRS = {
    0: np.array([-1, 0]),  # north
    1: np.array([1, 0]),   # south
    2: np.array([0, 1]),   # east
    3: np.array([0, -1]),  # west
}
STAY = 4
SCOUT = 0
FORAGER = 1
GARDENER = 2
ROLE_IDS = [SCOUT, FORAGER, GARDENER]


class Radio:
    """Last-writer-wins buffer of fixed size."""

    def __init__(self, n_bytes: int = 4):
        self.buffer = np.zeros(n_bytes, dtype=np.uint8)

    def write(self, arr: np.ndarray) -> None:
        self.buffer[:] = arr[: len(self.buffer)]

    def read(self) -> np.ndarray:
        return self.buffer.copy()


class OrchardCoop(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, grid_sz: int = 25, n_agents: int = 5, respawn_delay: int = 30):
        self.grid_sz = grid_sz
        self.n_agents = n_agents
        self.respawn_delay = respawn_delay
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.radio = Radio()
        self._seed_rng()
        self.reset()

    # --- spaces ---
    def observation_space(self, agent):
        return spaces.Box(low=0.0, high=1.0, shape=(5, 5, 5), dtype=np.float32)

    def action_space(self, agent):
        return spaces.MultiDiscrete([10, 256])

    # --- core mechanics ---
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        self._init_grid()
        obs = {a: self._obs(a) for a in self.agents}
        return obs

    def step(self, actions):
        self.t += 1
        for a in self.agents:
            self.last_ate[a] = False
        self._apply_moves(actions)
        self._apply_role_actions(actions)
        self._update_grid()
        rews = self._compute_rewards()
        terms = {a: False for a in self.agents}
        truncs = {a: self.t >= 1000 for a in self.agents}
        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rews, terms, truncs, infos

    # --- grid helpers ---
    def _init_grid(self):
        gs = self.grid_sz
        self.apples = np.zeros((gs, gs), dtype=int)
        self.saplings = np.zeros((gs, gs), dtype=int)
        self.mushrooms = np.zeros((gs, gs), dtype=bool)
        # random initial resources
        apple_coords = self.np_random.integers(0, gs, size=(gs // 2, 2))
        for x, y in apple_coords:
            self.apples[x, y] = 1
        sapling_coords = self.np_random.integers(0, gs, size=(gs // 3, 2))
        for x, y in sapling_coords:
            self.saplings[x, y] = self.np_random.integers(1, 3)
        mush_coords = self.np_random.integers(0, gs, size=(gs // 4, 2))
        for x, y in mush_coords:
            self.mushrooms[x, y] = True

        self.agent_pos = {a: self.np_random.integers(0, gs, size=2) for a in self.agents}
        self.agent_role = {a: FORAGER for a in self.agents}
        self.stamina = {a: 100 for a in self.agents}
        self.apple_respawn = {}
        self.last_ate = {a: False for a in self.agents}

    def _wrap(self, pos):
        return pos % self.grid_sz

    def _obs(self, agent):
        pos = self.agent_pos[agent]
        role = self.agent_role[agent]
        view = 2 if role == FORAGER else 2
        rng = range(-view, view + 1)
        patch = np.zeros((5, 5, 5), dtype=np.float32)
        for i, dx in enumerate(rng):
            for j, dy in enumerate(rng):
                coord = self._wrap(pos + np.array([dx, dy]))
                x, y = coord
                patch[i, j, 0] = self.apples[x, y]
                patch[i, j, 1] = self.saplings[x, y]
                patch[i, j, 2] = self.mushrooms[x, y]
                patch[i, j, 3] = int(any((coord == self.agent_pos[a]).all() for a in self.agents))
        patch[:, :, 4] = self.stamina[agent] / 100.0
        return patch

    def _apply_moves(self, actions):
        for a, act in actions.items():
            move = act[0]
            if move in MOVE_DIRS:
                self.agent_pos[a] = self._wrap(self.agent_pos[a] + MOVE_DIRS[move])
            elif move == STAY:
                pass
            elif 6 <= move <= 8:
                self.agent_role[a] = move - 6
            elif move == 9:
                pass
            self.stamina[a] -= 1
            if self.stamina[a] <= 0:
                self.agent_pos[a] = self.np_random.integers(0, self.grid_sz, size=2)
                self.stamina[a] = 100

    def _apply_role_actions(self, actions):
        for a, act in actions.items():
            if act[0] != 9:
                continue
            role = self.agent_role[a]
            pos = tuple(self.agent_pos[a])
            self.last_ate[a] = False
            if role == SCOUT:
                self.radio.write(np.array([act[1]], dtype=np.uint8))
            elif role == FORAGER:
                if self.apples[pos] > 0:
                    self.apples[pos] = 0
                    self.last_ate[a] = True
                    self.apple_respawn[pos] = self.respawn_delay
            elif role == GARDENER:
                if self.saplings[pos] > 0:
                    self.saplings[pos] = max(self.saplings[pos] - 1, 0)
                    self.stamina[a] = max(self.stamina[a] - 5, 0)

    def _update_grid(self):
        remove = []
        for coord, cnt in list(self.apple_respawn.items()):
            self.apple_respawn[coord] -= 1
            if self.apple_respawn[coord] <= 0:
                self.apples[coord] = 1
                remove.append(coord)
        for r in remove:
            del self.apple_respawn[r]

    def _just_ate_apple(self, a):
        return self.last_ate.get(a, False)

    def _count_apples(self):
        return int(self.apples.sum())

    def _collision_pairs(self):
        pos_map = {}
        pairs = []
        for a, p in self.agent_pos.items():
            t = tuple(p)
            if t in pos_map:
                pairs.append((a, pos_map[t]))
            pos_map[t] = a
        return pairs

    def _compute_rewards(self):
        base = {a: 0.0 for a in self.agents}
        for a in self.agents:
            if self._just_ate_apple(a):
                base[a] += 1.0
        if self.t % 50 == 0:
            bonus = 0.3 * self._count_apples()
            for a in base:
                base[a] += bonus
        for pair in self._collision_pairs():
            for a in pair:
                base[a] -= 1.0
        return base

    # --- util ---
    def _seed_rng(self, seed: int | None = None):
        self.np_random = np.random.default_rng(seed)

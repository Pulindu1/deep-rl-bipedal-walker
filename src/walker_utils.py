"""Environment, recording and plotting helpers for the BipedalWalker experiments.

Everything here is a thin layer over Gymnasium so the notebook stays focused on
the agent itself:

* :func:`make_walker`   -- build ``BipedalWalker-v3`` / ``BipedalWalkerHardcore-v3``
* :class:`Recorder`     -- per-episode statistics, TSV logs and periodic video capture
* :class:`InfoTracker`  -- live training curve inside the notebook
* :func:`seed_everything`, :func:`env_info`, :func:`check_device`, :func:`render`

The TSV written by :meth:`Recorder.write_log` is the format consumed by
``scripts/plot_training_curves.py`` and shipped in ``results/logs/``:
``count``, ``reward_sum``, ``squared_reward_sum``, ``length``.
"""

from __future__ import annotations

import random
from pathlib import Path

import gymnasium as gym
import numpy as np

__all__ = [
    "make_walker",
    "Recorder",
    "InfoTracker",
    "seed_everything",
    "env_info",
    "check_device",
    "render",
]

STANDARD_ID = "BipedalWalker-v3"
HARDCORE_ID = "BipedalWalkerHardcore-v3"
EPISODE_CAP = 2000  # steps per episode, matching the runs in results/logs/


def make_walker(
    hardcore: bool = False,
    render_mode: str | None = "rgb_array",
    max_episode_steps: int | None = EPISODE_CAP,
    **kwargs,
):
    """Create a BipedalWalker environment.

    ``BipedalWalkerHardcore-v3`` is the same physics with ladders, stumps and
    pits added to the terrain. Both have a 24-dim observation and a 4-dim
    continuous action in ``[-1, 1]``.

    Gymnasium registers the standard course with a 1600-step limit and the
    hardcore course with 2000. The runs in ``results/logs/`` used 2000 for
    both, so that is the default here; pass ``max_episode_steps=None`` for
    Gymnasium's registered values.
    """
    env_id = HARDCORE_ID if hardcore else STANDARD_ID
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    return gym.make(env_id, render_mode=render_mode, **kwargs)


class Recorder(gym.Wrapper):
    """Track per-episode statistics and optionally record videos.

    Parameters
    ----------
    video:
        Enable periodic video capture (requires ``render_mode="rgb_array"``
        and ``moviepy``).
    video_every:
        Record one episode every ``video_every`` episodes.
    smoothing:
        Window, in episodes, for the running mean/std exposed on ``info``.
    """

    def __init__(
        self,
        env,
        video: bool = False,
        video_folder: str = "videos",
        video_prefix: str = "episode",
        video_every: int = 50,
        logs: bool = True,
        smoothing: int = 10,
    ):
        if video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                name_prefix=video_prefix,
                episode_trigger=lambda episode: episode % video_every == 0,
                disable_logger=True,
            )
        super().__init__(env)

        self.logs = logs
        self.smoothing = smoothing

        # One entry per finished episode.
        self.rewards: list[float] = []
        self.squared_rewards: list[float] = []
        self.lengths: list[int] = []

        self._reward_sum = 0.0
        self._squared_reward_sum = 0.0
        self._length = 0

    def reset(self, **kwargs):
        self._reward_sum = 0.0
        self._squared_reward_sum = 0.0
        self._length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._reward_sum += float(reward)
        self._squared_reward_sum += float(reward) ** 2
        self._length += 1

        if terminated or truncated:
            self.rewards.append(self._reward_sum)
            self.squared_rewards.append(self._squared_reward_sum)
            self.lengths.append(self._length)

            window = self.rewards[-self.smoothing:]
            info = dict(info)
            info["episode"] = {
                "count": len(self.rewards) - 1,
                "r_sum": self._reward_sum,
                "r_squared_sum": self._squared_reward_sum,
                "length": self._length,
                "r_mean_": float(np.mean(window)),
                "r_std_": float(np.std(window)),
            }

        return observation, reward, terminated, truncated, info

    def write_log(self, folder: str = "logs", file: str = "training-log.txt") -> Path:
        """Write the episode statistics as a tab-separated log file."""
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        out = path / file

        with out.open("w") as handle:
            handle.write("count\treward_sum\tsquared_reward_sum\tlength\n")
            for count, (reward, squared, length) in enumerate(
                zip(self.rewards, self.squared_rewards, self.lengths)
            ):
                handle.write(f"{count}\t{reward}\t{squared}\t{length}\n")

        return out


class InfoTracker:
    """Collect the per-episode ``info["episode"]`` dicts and plot them live."""

    def __init__(self):
        self.info: list[dict] = []

    def track(self, info: dict) -> None:
        episode = info.get("episode")
        if episode:
            self.info.append(episode)

    def _series(self, key):
        return np.array([entry[key] for entry in self.info], dtype=float)

    def plot(self, r_sum: bool = True, r_mean_: bool = True, r_std_: bool = True) -> None:
        if not self.info:
            return

        import matplotlib.pyplot as plt

        try:  # keep a single, updating figure when running in Jupyter
            from IPython.display import clear_output

            clear_output(wait=True)
        except ImportError:
            pass

        episodes = self._series("count")
        _, ax = plt.subplots(figsize=(9, 4))

        if r_sum:
            ax.plot(episodes, self._series("r_sum"), linestyle=":", marker="x",
                    markersize=3, linewidth=0.6, alpha=0.6, label="episode reward")
        if r_mean_:
            mean = self._series("r_mean_")
            ax.plot(episodes, mean, linewidth=2.0, label="running mean")
            if r_std_:
                std = self._series("r_std_")
                ax.fill_between(episodes, mean - std, mean + std, alpha=0.2,
                                label="running std")

        ax.set_xlabel("episode index")
        ax.set_ylabel("reward")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
        plt.show()


def seed_everything(seed: int, env=None):
    """Seed Python, NumPy, PyTorch and (optionally) the environment."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    if env is None:
        return seed, None, None

    observation, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return seed, observation, info


def env_info(env, print_out: bool = False):
    """Return ``(discrete_act, discrete_obs, act_dim, obs_dim)`` for ``env``."""
    act_space, obs_space = env.action_space, env.observation_space

    discrete_act = isinstance(act_space, gym.spaces.Discrete)
    discrete_obs = isinstance(obs_space, gym.spaces.Discrete)
    act_dim = act_space.n if discrete_act else act_space.shape[0]
    obs_dim = obs_space.n if discrete_obs else obs_space.shape[0]

    if print_out:
        print(f"observation space: {obs_space} (dim {obs_dim})")
        print(f"action space:      {act_space} (dim {act_dim})")

    return discrete_act, discrete_obs, act_dim, obs_dim


def check_device():
    """Print and return the torch device training will run on."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} on {device}")
    return device


def render(env, clear: bool = False) -> None:
    """Show the current ``rgb_array`` frame inline in a notebook."""
    frame = env.render()
    if frame is None:
        return

    import matplotlib.pyplot as plt

    if clear:
        try:
            from IPython.display import clear_output

            clear_output(wait=True)
        except ImportError:
            pass

    plt.figure(figsize=(6, 4))
    plt.imshow(frame)
    plt.axis("off")
    plt.show()

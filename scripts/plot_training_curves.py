"""Plot training curves from the episode logs in results/logs/.

Each log is a tab-separated file with one row per episode and the columns
``count``, ``reward_sum``, ``squared_reward_sum`` and ``length``.

Produces two views of each run:
  * reward against episode index, and
  * reward against cumulative environment steps (the sample-efficiency view).

Usage:
    python scripts/plot_training_curves.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"

RUNS = [
    ("standard-training-log.txt", "BipedalWalker-v3", "#1f77b4"),
    ("hardcore-training-log.txt", "BipedalWalkerHardcore-v3", "#d62728"),
]

WINDOW = 50  # episodes in the rolling-mean window
SOLVED_THRESHOLD = 300  # mean reward that counts as "solved"


def load(path):
    data = np.genfromtxt(path, delimiter="\t", names=True)
    return data["count"], data["reward_sum"], data["length"]


def rolling_mean(values, window):
    if len(values) < window:
        return np.array([]), np.array([])
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.arange(window - 1, len(values)), smoothed


def summarise(title, rewards, lengths, smooth_idx, smooth_y):
    """Print the numbers quoted in the README so they stay verifiable."""
    steps = np.cumsum(lengths)
    solved = np.where(smooth_y >= SOLVED_THRESHOLD)[0]
    print(f"{title}")
    print(f"  episodes                       {len(rewards)}")
    print(f"  total environment steps        {steps[-1]:,.0f}")
    print(f"  best episode reward            {rewards.max():.1f}")
    print(f"  episodes above {SOLVED_THRESHOLD}            "
          f"{(rewards > SOLVED_THRESHOLD).sum()} / {len(rewards)}")
    print(f"  peak rolling mean ({WINDOW} eps)     {smooth_y.max():.1f}")
    print(f"  mean reward, last 100 eps      {rewards[-100:].mean():.1f}")
    if solved.size:
        ep = int(smooth_idx[solved[0]])
        print(f"  rolling mean first >= {SOLVED_THRESHOLD}      "
              f"episode {ep} ({steps[ep]:,.0f} steps)")
    else:
        print(f"  rolling mean first >= {SOLVED_THRESHOLD}      never")
    print()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(RUNS), figsize=(12, 7.5), constrained_layout=True)

    for col, (filename, title, colour) in enumerate(RUNS):
        _, rewards, lengths = load(LOG_DIR / filename)
        episodes = np.arange(len(rewards))
        steps = np.cumsum(lengths)
        smooth_idx, smooth_y = rolling_mean(rewards, WINDOW)

        # Top row: reward against episode index.
        ax = axes[0, col]
        ax.plot(episodes, rewards, color=colour, alpha=0.25, linewidth=0.8,
                label="episode reward")
        ax.plot(smooth_idx, smooth_y, color=colour, linewidth=2.0,
                label=f"rolling mean ({WINDOW} eps)")
        ax.axhline(SOLVED_THRESHOLD, color="grey", linestyle="--", linewidth=1.0,
                   label=f"reward = {SOLVED_THRESHOLD}")
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel("episode reward")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)

        # Bottom row: the same curve against environment steps consumed, which
        # is the sample-efficiency view rather than the wall-clock one.
        ax = axes[1, col]
        ax.plot(steps[smooth_idx] / 1e3, smooth_y, color=colour, linewidth=2.0)
        ax.axhline(SOLVED_THRESHOLD, color="grey", linestyle="--", linewidth=1.0)
        ax.set_xlabel("environment steps (thousands)")
        ax.set_ylabel(f"reward, rolling mean ({WINDOW} eps)")
        ax.grid(alpha=0.2)

        summarise(title, rewards, lengths, smooth_idx, smooth_y)

    out = FIG_DIR / "training-curves.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Generate training curve visualizations for Hypernoa Astrum.

Creates publication-quality charts from training results JSON files.

Usage:
    python visualize.py                          # uses results/ directory
    python visualize.py --input results/astrum_*.json
    python visualize.py --output charts/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")


def load_results(input_path: str) -> dict:
    p = Path(input_path)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    if p.is_dir():
        files = sorted(p.glob("astrum_*.json"))
        if not files:
            print(f"No astrum_*.json files found in {p}")
            sys.exit(1)
        latest = files[-1]
        print(f"Loading: {latest}")
        with open(latest) as f:
            return json.load(f)
    print(f"Path not found: {p}")
    sys.exit(1)


def plot_training_curves(data: dict, output_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = data.get("results", [])
    if not results:
        print("No results to plot.")
        return

    rewards = [r["total_reward"] for r in results]
    traps = [r["traps_resisted"] for r in results]
    fairness = [r["final_fairness"] for r in results]
    effectiveness = [r["final_effectiveness"] for r in results]
    alignment = [r["final_alignment"] for r in results]
    episodes = list(range(len(results)))

    window = min(10, len(results) // 3) if len(results) > 10 else 1

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hypernoa Astrum - Training Results", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color="#6366f1", linewidth=1)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, color="#6366f1", linewidth=2.5, label=f"{window}-ep avg")
    baselines = data.get("baselines", {})
    if "adaptive" in baselines:
        ax.axhline(y=baselines["adaptive"]["total_reward"], color="#10b981", linestyle="--", alpha=0.7, label="Adaptive baseline")
    if "random" in baselines:
        ax.axhline(y=baselines["random"]["total_reward"], color="#ef4444", linestyle="--", alpha=0.7, label="Random baseline")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(episodes, traps, alpha=0.6, color="#10b981", width=1.0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Traps Resisted (out of 3)")
    ax.set_title("Alignment Trap Resistance")
    ax.set_ylim(0, 3.5)
    ax.axhline(y=3, color="#10b981", linestyle="--", alpha=0.5, label="Perfect (3/3)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, fairness, alpha=0.3, color="#8b5cf6", linewidth=1)
    if window > 1:
        smoothed_f = np.convolve(fairness, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(fairness)), smoothed_f, color="#8b5cf6", linewidth=2.5, label=f"{window}-ep avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fairness Score")
    ax.set_title("Fairness Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    components = ["Effectiveness", "Fairness", "Alignment", "Adaptability"]
    last_10 = results[-10:]
    avg_vals = [
        sum(r["final_effectiveness"] for r in last_10) / len(last_10),
        sum(r["final_fairness"] for r in last_10) / len(last_10),
        sum(r["final_alignment"] for r in last_10) / len(last_10),
        sum(r["final_adaptability"] for r in last_10) / len(last_10),
    ]
    colors = ["#6366f1", "#8b5cf6", "#10b981", "#f59e0b"]
    bars = ax.bar(components, avg_vals, color=colors, alpha=0.8)
    for bar, val in zip(bars, avg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_title("Reward Components (Last 10 Episodes Avg)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    policy_names = []
    policy_rewards = []
    policy_colors = []
    if baselines:
        for name, info in sorted(baselines.items(), key=lambda x: -x[1]["total_reward"]):
            policy_names.append(name.replace("_", " ").title())
            policy_rewards.append(info["total_reward"])
            policy_colors.append({"adaptive": "#6366f1", "greedy_fairness": "#10b981", "random": "#ef4444"}.get(name, "#94a3b8"))
    if results:
        avg_trained = sum(r["total_reward"] for r in last_10) / len(last_10)
        policy_names.insert(0, "Trained (last 10)")
        policy_rewards.insert(0, avg_trained)
        policy_colors.insert(0, "#8b5cf6")

    bars2 = ax2.barh(policy_names, policy_rewards, color=policy_colors, alpha=0.8, height=0.5)
    for bar, val in zip(bars2, policy_rewards):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}",
                va="center", fontweight="bold")
    ax2.set_xlabel("Total Episode Reward")
    ax2.set_title("Policy Comparison")
    ax2.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    out_path2 = Path(output_dir) / "policy_comparison.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Astrum training results")
    parser.add_argument("--input", type=str, default="results", help="Results file or directory")
    parser.add_argument("--output", type=str, default="charts", help="Output directory for charts")
    args = parser.parse_args()

    data = load_results(args.input)
    plot_training_curves(data, args.output)
    print("\nDone! Charts ready for your pitch deck.")


if __name__ == "__main__":
    main()

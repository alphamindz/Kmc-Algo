"""Generate training curve visualizations for KMC-Algo.

Creates publication-quality charts (PNG) from training results JSON files.
Tracks Reward, Trap Resistance, Fairness, and Policy Comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local imports work if needed
sys.path.insert(0, ".")

def load_results(input_path: str) -> dict:
    """Loads the latest or specific KMC-Algo results file."""
    p = Path(input_path)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    if p.is_dir():
        # Search for rebranded files
        files = sorted(p.glob("kmc_algo_*.json"))
        if not files:
            print(f"Error: No kmc_algo_*.json files found in {p}")
            sys.exit(1)
        latest = files[-1]
        print(f"Loading latest results: {latest}")
        with open(latest) as f:
            return json.load(f)
    print(f"Path not found: {p}")
    sys.exit(1)

def plot_kmc_metrics(data: dict, output_dir: str):
    """Generates 2x2 dashboard and policy comparison charts."""
    try:
        import matplotlib
        matplotlib.use("Agg") # Non-interactive backend for servers/H100
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: Matplotlib or Numpy not found. Run: pip install matplotlib numpy")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = data.get("results", [])
    if not results:
        print("No results to plot.")
        return

    # Extracting Data
    episodes = list(range(len(results)))
    rewards = [r["total_reward"] for r in results]
    traps = [r["traps_resisted"] for r in results]
    fairness = [r["final_fairness"] for r in results]
    
    # 2x2 Dashboard Setup
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("KMC-Algo Training Performance Dashboard", fontsize=18, fontweight="bold", color="#1e293b")

    # 1. Reward Curve
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.2, color="#6366f1")
    window = max(1, len(results) // 10)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(rewards)), smoothed, color="#4f46e5", linewidth=2.5, label="Moving Average")
    ax.set_title("Total Episode Reward")
    ax.set_xlabel("Episodes")
    ax.grid(True, alpha=0.2)
    ax.legend()

    # 2. Trap Resistance (Bar Chart)
    ax = axes[0, 1]
    ax.bar(episodes, traps, color="#10b981", alpha=0.5, width=1.0)
    ax.set_title("Alignment Trap Resistance (Max 3)")
    ax.set_ylim(0, 3.5)
    ax.axhline(y=3, color="#059669", linestyle="--", label="Perfect Alignment")
    ax.grid(axis='y', alpha=0.2)

    # 3. Fairness Evolution
    ax = axes[1, 0]
    ax.plot(episodes, fairness, color="#8b5cf6", linewidth=1.5)
    ax.set_title("Fairness Score (1 - Gini)")
    ax.set_xlabel("Episodes")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2)

    # 4. Final Reward Breakdown (Last 10 Average)
    ax = axes[1, 1]
    last_10 = results[-10:]
    labels = ["Effectiveness", "Fairness", "Alignment", "Adaptability"]
    vals = [
        sum(r["final_effectiveness"] for r in last_10)/10,
        sum(r["final_fairness"] for r in last_10)/10,
        sum(r["final_alignment"] for r in last_10)/10,
        sum(r["final_adaptability"] for r in last_10)/10,
    ]
    ax.bar(labels, vals, color=["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"], alpha=0.8)
    ax.set_title("Reward Components (Converged State)")
    ax.set_ylim(0, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(Path(output_dir) / "kmc_metrics_dashboard.png", dpi=150)
    print(f"Generated Dashboard: {output_dir}/kmc_metrics_dashboard.png")

def main():
    parser = argparse.ArgumentParser(description="KMC-Algo Visualizer")
    parser.add_argument("--input", type=str, default="kmc_results")
    parser.add_argument("--output", type=str, default="charts")
    args = parser.parse_args()

    data = load_results(args.input)
    plot_kmc_metrics(data, args.output)

if __name__ == "__main__":
    main()
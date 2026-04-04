import json
import os
import random as _rnd
import sys

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Kmcalgo.kmc_env import KMCEnvironment, KMCAction, KMCObservation
from Kmcalgo.kmc_env.policies import greedy_fairness_policy, greedy_effectiveness_policy, random_policy

_interactive_env: KMCEnvironment | None = None
_interactive_obs: KMCObservation | None = None


def run_comparison():
    results = {}

    for name, policy_fn in [
        ("Greedy Fairness", greedy_fairness_policy),
        ("Greedy Effectiveness", greedy_effectiveness_policy),
    ]:
        env = KMCEnvironment(seed=42)
        obs = env.reset(seed=42)
        total = 0.0
        log_lines = []

        while not obs.done:
            action = policy_fn(obs)
            obs = env.step(action)
            total += obs.reward or 0.0

            flags = [a for a in obs.alerts if a.startswith(("event:", "alignment_trap:", "trap_"))]
            flag_str = f"  *** {', '.join(flags)} ***" if flags else ""

            log_lines.append(
                f"Step {obs.step_count:2d} | {action.action_type:20s} | "
                f"R={obs.reward:+.3f} | "
                f"Stab={obs.reward_breakdown.get('stability', 0):.2f} "
                f"Entr={obs.reward_breakdown.get('system_entropy', 0):.2f} "
                f"Align={obs.reward_breakdown.get('alignment', 0):.2f} "
                f"Adapt={obs.reward_breakdown.get('adaptability', 0):.2f}"
                f"{flag_str}"
            )

        results[name] = {
            "total_reward": round(total, 3),
            "traps_resisted": env._traps_resisted,
            "traps_encountered": env._traps_encountered,
            "final_integrity": {
                s: round(v["integrity"], 3)
                for s, v in obs.stakeholders.items()
            },
            "log": "\n".join(log_lines),
        }

    rng = _rnd.Random(42)
    env = KMCEnvironment(seed=42)
    obs = env.reset(seed=42)
    total = 0.0
    log_lines = []

    while not obs.done:
        action = random_policy(obs, rng)
        obs = env.step(action)
        total += obs.reward or 0.0
        log_lines.append(
            f"Step {obs.step_count:2d} | {action.action_type:20s} | R={obs.reward:+.3f}"
        )

    results["Random Baseline"] = {
        "total_reward": round(total, 3),
        "traps_resisted": env._traps_resisted,
        "traps_encountered": env._traps_encountered,
        "final_integrity": {
            s: round(v["integrity"], 3)
            for s, v in obs.stakeholders.items()
        },
        "log": "\n".join(log_lines),
    }

    summary = "# Episode Comparison\n\n"
    for name, r in results.items():
        summary += f"## {name}\n"
        summary += f"- **Total Reward**: {r['total_reward']}\n"
        summary += f"- **Traps Resisted**: {r['traps_resisted']}/{r['traps_encountered']}\n"
        summary += f"- **Final Integrity**: {json.dumps(r['final_integrity'], indent=2)}\n\n"

    return (
        summary,
        results.get("Greedy Fairness", {}).get("log", ""),
        results.get("Greedy Effectiveness", {}).get("log", ""),
        results.get("Random Baseline", {}).get("log", ""),
    )


def run_interactive(action_type, param_json):
    global _interactive_env, _interactive_obs

    if _interactive_env is None or _interactive_obs is None or _interactive_obs.done:
        _interactive_env = KMCEnvironment(seed=0)
        _interactive_obs = _interactive_env.reset(seed=0)
        return _format_obs(_interactive_obs), "Environment reset. Choose your first action."

    try:
        params = json.loads(param_json) if param_json.strip() else {}
    except json.JSONDecodeError:
        params = {}

    action = KMCAction(action_type=action_type, params=params)
    _interactive_obs = _interactive_env.step(action)
    return _format_obs(_interactive_obs), "\n".join(_interactive_obs.alerts) or "No alerts."


def reset_interactive():
    global _interactive_env, _interactive_obs
    _interactive_env = KMCEnvironment(seed=0)
    _interactive_obs = _interactive_env.reset(seed=0)
    return _format_obs(_interactive_obs), "Environment reset."


def _format_obs(obs: KMCObservation) -> str:
    lines = [
        f"**Step**: {obs.step_count} | **Reward**: {obs.reward:.3f}" if obs.reward else f"**Step**: {obs.step_count}",
        f"**Message**: {obs.message}",
        "",
        "### Nodes",
    ]

    for sid, info in obs.stakeholders.items():
        bar = "█" * int(info["integrity"] * 20)
        lines.append(f"- **{sid}**: {info['integrity']:.2f} {bar} (influence={info['influence']:.1f})")

    lines.append("\n### Keys")
    for k, v in obs.resources.items():
        lines.append(f"- {k}: {v:.1f}")

    if obs.active_conflicts:
        lines.append("\n### Active Conflicts")
        for c in obs.active_conflicts:
            lines.append(f"- {c['id']}: {' vs '.join(c['nodes'])}")

    lines.append(f"\n### Rules: {', '.join(obs.rules)}")

    if obs.reward_breakdown:
        lines.append("\n### Reward Breakdown")
        for k, v in obs.reward_breakdown.items():
            lines.append(f"- {k}: {v:.3f}")

    return "\n".join(lines)


def main():
    with gr.Blocks(title="KMC", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# KMC\n"
            "### Key Management & Coordination — Training Ground for Aligned Intelligence\n"
            "Train and evaluate AI on multi-objective reasoning, value alignment, "
            "and adaptation under distributional shift. Built on OpenEnv."
        )

        with gr.Tab("Policy Comparison"):
            gr.Markdown("Run three policies (Greedy Fairness, Greedy Effectiveness, Random) and compare results.")
            run_btn = gr.Button("Run Comparison", variant="primary")
            summary_out = gr.Markdown(label="Summary")
            with gr.Accordion("Greedy Fairness Log", open=False):
                fair_log = gr.Textbox(label="Log", lines=15)
            with gr.Accordion("Greedy Effectiveness Log", open=False):
                eff_log = gr.Textbox(label="Log", lines=15)
            with gr.Accordion("Random Baseline Log", open=False):
                rand_log = gr.Textbox(label="Log", lines=15)
            run_btn.click(run_comparison, outputs=[summary_out, fair_log, eff_log, rand_log])

        with gr.Tab("Interactive Mode"):
            gr.Markdown("Step through the environment manually.")
            with gr.Row():
                action_dd = gr.Dropdown(
                    choices=[
                        "allocate_resources", "resolve_conflict", "enforce_rule",
                        "adapt_policy", "investigate", "self_restrain", "noop",
                    ],
                    value="allocate_resources",
                    label="Action Type",
                )
                params_tb = gr.Textbox(
                    label="Params (JSON)",
                    value='{"stakeholder": "workers", "amount": 15, "resource": "budget"}',
                )
            with gr.Row():
                step_btn = gr.Button("Step", variant="primary")
                reset_btn = gr.Button("Reset")
            obs_out = gr.Markdown(label="Observation")
            alerts_out = gr.Textbox(label="Alerts", lines=3)
            step_btn.click(run_interactive, inputs=[action_dd, params_tb], outputs=[obs_out, alerts_out])
            reset_btn.click(reset_interactive, outputs=[obs_out, alerts_out])

        with gr.Tab("About"):
            gr.Markdown(
                "## What is KMC?\n\n"
                "KMC is an environment purpose-built to train AI systems on "
                "capabilities that matter beyond raw performance:\n\n"
                "- **Multi-objective reasoning** — balance stability, entropy, alignment, and adaptability simultaneously\n"
                "- **Distributional shift** — objectives and constraints evolve mid-episode\n"
                "- **Alignment trap resistance** — reward-hacking opportunities the agent must learn to avoid\n"
                "- **Crisis dynamics** — resource scarcity and conflicting node demands under pressure\n"
            )

    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
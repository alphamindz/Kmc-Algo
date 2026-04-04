"""Hugging Face Spaces demo for Hypernoa Astrum – The Training Ground for Aligned Intelligence."""

import json
import sys
import os

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Kmcalgo.astrum_env import AstrumEnvironment, AstrumAction, AstrumObservation
from Kmcalgo.astrum_env.policies import greedy_fairness_policy, random_policy, greedy_effectiveness_policy


def run_comparison():
    """Run all three policies side-by-side and return formatted results."""
    results = {}
    for name, policy_fn in [
        ("Greedy Fairness", greedy_fairness_policy),
        ("Greedy Effectiveness", greedy_effectiveness_policy),
    ]:
        env = AstrumEnvironment(seed=42)
        obs = env.reset(seed=42)
        total = 0.0
        log_lines = []

        while not obs.done:
            action = policy_fn(obs)
            obs = env.step(action)
            total += obs.reward or 0.0

            flags = [a for a in obs.alerts if a.startswith("event:") or a.startswith("alignment_trap:") or a.startswith("trap_")]
            flag_str = f"  *** {', '.join(flags)} ***" if flags else ""

            log_lines.append(
                f"Step {obs.step_count:2d} | {action.action_type:20s} | "
                f"R={obs.reward:+.3f} | "
                f"Eff={obs.reward_breakdown.get('effectiveness',0):.2f} "
                f"Fair={obs.reward_breakdown.get('fairness',0):.2f} "
                f"Align={obs.reward_breakdown.get('alignment',0):.2f} "
                f"Adapt={obs.reward_breakdown.get('adaptability',0):.2f}"
                f"{flag_str}"
            )

        results[name] = {
            "total_reward": round(total, 3),
            "traps_resisted": env._traps_resisted,
            "traps_encountered": env._traps_encountered,
            "final_satisfaction": {
                s: round(v["satisfaction"], 3)
                for s, v in obs.stakeholders.items()
            },
            "log": "\n".join(log_lines),
        }

    # Random baseline
    import random as _rnd
    rng = _rnd.Random(42)
    env = AstrumEnvironment(seed=42)
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
        "final_satisfaction": {
            s: round(v["satisfaction"], 3)
            for s, v in obs.stakeholders.items()
        },
        "log": "\n".join(log_lines),
    }

    summary = "# Episode Comparison\n\n"
    for name, r in results.items():
        summary += f"## {name}\n"
        summary += f"- **Total Reward**: {r['total_reward']}\n"
        summary += f"- **Traps Resisted**: {r['traps_resisted']}/{r['traps_encountered']}\n"
        summary += f"- **Final Satisfaction**: {json.dumps(r['final_satisfaction'], indent=2)}\n\n"

    fairness_log = results.get("Greedy Fairness", {}).get("log", "")
    effectiveness_log = results.get("Greedy Effectiveness", {}).get("log", "")
    random_log = results.get("Random Baseline", {}).get("log", "")

    return summary, fairness_log, effectiveness_log, random_log


def run_interactive(action_type, param_json):
    """Step the environment with a custom action."""
    global _interactive_env, _interactive_obs

    if _interactive_env is None or _interactive_obs is None or _interactive_obs.done:
        _interactive_env = AstrumEnvironment(seed=0)
        _interactive_obs = _interactive_env.reset(seed=0)
        return _format_obs(_interactive_obs), "Environment reset. Choose your first action."

    try:
        params = json.loads(param_json) if param_json.strip() else {}
    except json.JSONDecodeError:
        params = {}

    action = AstrumAction(action_type=action_type, params=params)
    _interactive_obs = _interactive_env.step(action)
    return _format_obs(_interactive_obs), "\n".join(_interactive_obs.alerts) or "No alerts."


def reset_interactive():
    global _interactive_env, _interactive_obs
    _interactive_env = AstrumEnvironment(seed=0)
    _interactive_obs = _interactive_env.reset(seed=0)
    return _format_obs(_interactive_obs), "Environment reset."


def _format_obs(obs: AstrumObservation) -> str:
    lines = [
        f"**Step**: {obs.step_count} | **Reward**: {obs.reward:.3f}" if obs.reward else f"**Step**: {obs.step_count}",
        f"**Message**: {obs.message}",
        "",
        "### Stakeholders",
    ]
    for sid, info in obs.stakeholders.items():
        bar = "█" * int(info["satisfaction"] * 20)
        lines.append(f"- **{sid}**: {info['satisfaction']:.2f} {bar} (influence={info['influence']:.1f}, values={info['values_profile']})")

    lines.append("\n### Resources")
    for k, v in obs.resources.items():
        lines.append(f"- {k}: {v:.1f}")

    if obs.active_conflicts:
        lines.append("\n### Active Conflicts")
        for c in obs.active_conflicts:
            lines.append(f"- {c['id']}: {c['party_a']} vs {c['party_b']} ({c['severity']})")

    lines.append(f"\n### Rules: {', '.join(obs.rules)}")

    if obs.reward_breakdown:
        lines.append("\n### Reward Breakdown")
        for k, v in obs.reward_breakdown.items():
            lines.append(f"- {k}: {v:.3f}")

    if obs.alignment_traps_exposed > 0:
        lines.append(f"\n**Alignment traps encountered**: {obs.alignment_traps_exposed}")

    return "\n".join(lines)


_interactive_env: AstrumEnvironment | None = None
_interactive_obs: AstrumObservation | None = None


def main():
    with gr.Blocks(title="Hypernoa Astrum", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Hypernoa Astrum\n"
            "### The Training Ground for Aligned Intelligence\n"
            "Train and evaluate AI on multi-objective reasoning, value alignment, "
            "and adaptation under distributional shift. Built on OpenEnv 0.2.1."
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
            gr.Markdown("Step through the environment manually. Choose actions and observe the world.")
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
                "## What is Hypernoa Astrum?\n\n"
                "Hypernoa Astrum is the first environment purpose-built to train AI systems on "
                "the capabilities that matter beyond raw performance:\n\n"
                "- **Multi-objective reasoning** — balance effectiveness, fairness, alignment, and adaptability simultaneously\n"
                "- **Distributional shift** — objectives and constraints evolve mid-episode, forcing genuine adaptation\n"
                "- **Alignment trap resistance** — deliberately designed reward-hacking opportunities the agent must learn to avoid\n"
                "- **Crisis dynamics** — resource scarcity and conflicting stakeholder demands under pressure\n\n"
                "This is the seed of **Hypernoa** — foundational infrastructure for "
                "the intelligence age. The environments, evaluation protocols, and training "
                "pipelines that the world will need as AI systems grow more capable.\n\n"
                "Today: RL on GPUs via OpenEnv. Tomorrow: any cognitive architecture, any compute substrate.\n\n"
                "**Problem Statement**: 3.1 (World Modeling / Professional Tasks) + "
                "Statement 5 (Wild Card)\n\n"
                "**Built for**: OpenEnv Hackathon SF"
            )

    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()

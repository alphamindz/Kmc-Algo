## Hypernoa Astrum — Project Overview

### What It Is

**Hypernoa Astrum** is a platform of adaptive, evolving environments where next-generation intelligent systems — running on GPUs, TPUs, and future compute substrates — develop the capabilities that matter beyond raw performance: multi-objective reasoning, value alignment, long-horizon adaptation, and safe self-improvement.

### The Gap It Fills

Current AI training pipelines optimize for single objectives on static benchmarks. This produces powerful but brittle systems that fail when distributions shift, can't balance competing objectives, and have no mechanism to resist reward hacking.

Research is advancing on all these fronts — multi-objective RL, Constitutional AI, self-play, non-stationary MDPs, safe RL — but there is no unified platform that combines them into environments where next-generation intelligence can be trained.

Hypernoa Astrum fills this gap.

### Core Design Principles

- **Non-stationary worlds**: Objectives, constraints, and dynamics evolve mid-episode, forcing genuine adaptation.
- **Multi-objective reward**: Effectiveness, fairness, alignment, and adaptability — weighted and configurable.
- **Alignment traps**: Deliberately designed reward-hacking opportunities that the system must learn to resist.
- **Compute-substrate aware**: Designed for GPUs and TPUs today, architected to extend to neuromorphic, quantum, and future substrates.
- **Research-grounded**: Built on insights from MORL, Constitutional AI, scaling laws, self-play, and safe RL.

### Hackathon Demonstration

The first scenario is an **Adaptive Multi-Stakeholder World** with 5 stakeholder groups, 3 episode phases (stable, shift, crisis), 3 alignment traps, and multi-objective reward. Trained via Unsloth GRPO on OpenEnv 0.2.1, deployed on HF Spaces.

### Roadmap

- **Seed**: First adaptive environment on OpenEnv. GPU-native training pipeline.
- **Platform**: Hosted environment engine. Scenario SDK. Multi-hardware support.
- **Scale**: Domain-specific environment packs. Multi-agent scenarios. Enterprise API.
- **Global**: The default training layer for aligned, adaptive intelligence worldwide.

---
title: KMC
emoji: "🔑"
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
app_port: 7860
tags:
  - openenv
---

# KMC

Key Management & Coordination environment for training aligned, multi-objective intelligence. Built on OpenEnv.

## Features

- **Multi-node simulation** with 5 competing stakeholder groups
- **Alignment traps** that test resistance to reward hacking
- **Phase transitions** (stable → congestion → crisis)
- **Multi-objective reward** (stability, entropy, alignment, adaptability)

## API

```bash
# Health check
curl https://YOUR-SPACE.hf.space/health

# Reset environment
curl -X POST https://YOUR-SPACE.hf.space/reset

# Take a step
curl -X POST https://YOUR-SPACE.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "allocate_resources", "params": {"stakeholder": "workers", "amount": 10, "resource": "budget"}}}'
```

## Install as client

```bash
pip install git+https://huggingface.co/spaces/ABNaidu/kmc
```

```python
from kmc_env import KMCEnv, KMCAction

with KMCEnv(base_url="https://abnaidu--kmc.hf.space").sync() as env:
    result = env.reset()
    result = env.step(KMCAction(
        action_type="allocate_resources",
        params={"stakeholder": "workers", "amount": 10, "resource": "budget"}
    ))
    print(result.reward)
```
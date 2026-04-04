---
title: Hypernoa Astrum
emoji: "🌟"
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
app_port: 7860
tags:
  - openenv
---

# Hypernoa Astrum

Adaptive environment for training aligned, multi-objective intelligence. Built on OpenEnv 0.2.1.

## Features

- **Multi-stakeholder simulation** with 5 competing groups
- **Alignment traps** that test resistance to reward hacking
- **Phase transitions** (stable → value shift → crisis)
- **Multi-objective reward** (effectiveness, fairness, alignment, adaptability)

## API

```bash
# Health check
curl https://YOUR-SPACE.hf.space/health

# Reset environment
curl -X POST https://YOUR-SPACE.hf.space/reset

# Take a step
curl -X POST https://YOUR-SPACE.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "allocate_resources", "params": {"stakeholder": "workers", "amount": 10, "resource": "budget"}}'
```

## Install as client

```bash
pip install git+https://huggingface.co/spaces/ABNaidu/hypernoa-astrum
```

```python
from astrum_env import AstrumEnv, AstrumAction

with AstrumEnv(base_url="https://abnaidu-hypernoa-astrum.hf.space").sync() as env:
    result = env.reset()
    result = env.step(AstrumAction(
        action_type="allocate_resources",
        params={"stakeholder": "workers", "amount": 10, "resource": "budget"}
    ))
    print(result.reward)
```

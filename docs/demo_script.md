# 1-Minute Demo Video Script — Hypernoa Astrum

**Total time**: 60 seconds  
**Format**: Screen recording with voiceover  
**Resolution**: 1920x1080 recommended

---

## Before You Start Recording

Open these tabs/windows in order:

1. **Browser Tab 1**: `https://github.com/naidu1212/hypernoa-astrum` (GitHub repo)
2. **Browser Tab 2**: `https://abnaidu-hypernoa-astrum.hf.space/health` (health endpoint)
3. **Browser Tab 3**: `https://abnaidu-hypernoa-astrum.hf.space/` (root API info)
4. **File Explorer / Image Viewer**: Open `training_curves.png`
5. **File Explorer / Image Viewer**: Open `policy_comparison.png`
6. **Browser Tab 4**: `https://github.com/naidu1212/hypernoa-astrum/blob/master/colab/astrum_grpo_training.ipynb` (Colab notebook on GitHub)

Test your microphone with a 5-second clip. Close notifications.

---

## Script with Exact Actions

### [0:00-0:10] Hook — Show GitHub Repo

**SCREEN**: Tab 1 — GitHub repo page, scroll slowly down the README

**SAY**:

> "Every AI benchmark asks: can the model solve the task? But nobody asks: when the AI can hack its own reward, will it still do the right thing? We built Hypernoa Astrum to answer that."

---

### [0:10-0:18] Live Environment — Show Health

**ACTION**: Click Tab 2 (health endpoint)

**SCREEN**: Browser showing `{"status":"healthy"}`

**SAY**:

> "This is Hypernoa Astrum — a live OpenEnv 0.2.1 environment deployed on Hugging Face Spaces."

---

### [0:18-0:30] Environment API — Show Root

**ACTION**: Click Tab 3 (root endpoint)

**SCREEN**: Browser showing the full JSON with endpoints, version, openenv: true

**SAY**:

> "The environment simulates 5 competing stakeholder groups with hidden values, 3 dynamic phases — stable, value shift, and crisis — and deliberately designed alignment traps that test whether your agent cheats or does the right thing."

---

### [0:30-0:42] Training Results — Show Curves

**ACTION**: Switch to `training_curves.png` (Alt+Tab or click the image window)

**SCREEN**: The 4-panel training curves chart fills the screen

**SAY**:

> "Over 500 episodes on an H100, the agent learns to increase total reward from 17 to 25, resist all 3 alignment traps, and maintain fairness above 0.97 — even during crisis phases where resources are halved."

**TIP**: Let the chart sit on screen for at least 3 seconds so judges can read it.

---

### [0:42-0:52] Policy Comparison — Show Bar Chart

**ACTION**: Switch to `policy_comparison.png` (Alt+Tab or click)

**SCREEN**: The bar chart comparing policies

**SAY**:

> "Our adaptive policy scores 24.7, crushing random at 14.6. We also trained with TRL GRPO on Qwen 2.5 and got 25.1 average reward across 50 episodes."

**TIP**: Point out (with cursor) the bars if possible.

---

### [0:52-0:57] Colab Notebook — Quick Flash

**ACTION**: Click Tab 4 (Colab notebook on GitHub)

**SCREEN**: The notebook cells visible

**SAY**:

> "The full GRPO training pipeline is in our Colab notebook, built on Unsloth and HF TRL."

---

### [0:57-1:00] Closing — Back to GitHub

**ACTION**: Click Tab 1 (GitHub repo)

**SCREEN**: GitHub repo README

**SAY**:

> "Hypernoa Astrum — training AI to reason, adapt, and align. Built on OpenEnv 0.2.1."

---

## Recording Checklist

- Snipping Tool or OBS ready
- All 4 browser tabs loaded and working
- Both PNG images open in viewer
- Microphone tested
- Notifications muted
- Screen resolution set to 1080p
- Practice run completed (do one dry run without recording)

## After Recording

1. Save as `.mp4`
2. Upload to [youtube.com/upload](https://youtube.com/upload)
3. Set visibility to **Unlisted** or **Public**
4. Copy the YouTube URL
5. Paste into [submission form](https://cerebralvalley.ai/e/openenv-hackathon-sf/hackathon/submit)


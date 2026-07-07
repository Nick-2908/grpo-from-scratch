# 15 Days of GRPO — the writeup

I gave myself 15 days to build GRPO — the RL algorithm DeepSeek used to train R1 — completely from scratch. No TRL, no verl, no copying a trainer class and tweaking it. Just PyTorch, `transformers`, a small Qwen model, and one rule: **I don't move to the next concept until I can explain the current one and show it running.**

This is the honest log of how that went, including the parts where I was wrong.

---

## Days 1–2 · Theory first (NOTES.md)

Before writing any training code I sat down with the DeepSeekMath paper and wrote out the whole lineage by hand:

> REINFORCE (high variance) → add a baseline (advantage) → PPO (critic + clipping) → GRPO (drop the critic, use the group mean as the baseline).

The thing that actually took me time here wasn't the loss function — it was realizing that **RL on LLM completions is a bandit problem**, not the multi-step RL I'd studied. One completion, one scalar reward at the end. So the return `G_t`, the discount γ, Q-values, V-values — all of it just drops out. The single reward gets broadcast to every token of the completion. I first wrote "γ = 0" in my notes and had to correct myself: it's effectively γ = 1 (no discounting), because the signal has to reach *every* token undecayed, not just the last one. That correction is still in NOTES.md; I left my wrong first draft visible on purpose.

## Day 3 · REINFORCE (`01_reinforce.py`)

Smallest possible thing that learns: sample 4 completions from Qwen2.5-0.5B, reward = 1.0 if the completion is under 35 tokens, loss = `-(reward · seq_logp)`. That's it. It moves — average reward creeps up over 120 steps.

The unglamorous stuff took most of the day: getting per-token log-probs by shifting logits against targets, and masking so the prompt tokens and padding don't contribute to the loss. Every later script is built on these two things being right.

## Day 4 · Baseline (`02_reinforce_baseline.py`)

Same setup, but now with a `use_baseline` toggle and a function to measure the global gradient norm, so I could actually *see* the variance reduction instead of taking the textbook's word for it. Subtracting the group mean from rewards doesn't change what the model is asked to do — it changes how noisy the updates are on the way there.

## Day 5 · The importance ratio (`03_ppo_ratio_unclipped.py`)

This is where PPO enters. I snapshot the log-probs at rollout time (`old_logp`, no grad), then do multiple optimization epochs on the same batch, tracking `ratio = exp(new_logp − old_logp)`.

Two things I verified that day:

1. At epoch 0 the ratio is exactly 1.0 — the policy equals its own snapshot. If it isn't 1.0, your snapshot is wrong. This became my standard sanity check for everything after.
2. With **no constraint**, the ratio explodes: 1.0 → 6.9 → 20.4 → **35.0** in just three epochs on one batch. That's not a bug, that's the actual reason PPO exists. I didn't have to believe the paper — I watched it happen (`images/03_ratio_explosion.png`).

Also the day I properly internalized why we work in log space: subtracting log-probs and exponentiating is dividing probabilities, without underflowing from multiplying hundreds of tiny numbers together.

## Day 6 · Clipping (`04_ppo_clipped.py`)

Added the clipped surrogate: `-min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`.

Here's what surprised me: **clipping alone did not save me.** At lr=1e-5 the ratio still blew up to ~29,500 by epoch 3. The reason, once I stared at it: clipping only kills the gradient in the direction that inflates the objective. For negative-advantage samples the pessimistic branch stays unclipped *by design* — so a too-large learning rate can still run away through that path. Dropped to lr=1e-6 and got a smooth, stable drift (1.0 → 0.65 over 4 epochs, `images/04_clipped_ratio.png`).

Lesson that stuck: clip range and learning rate are coupled. Clipping is a seatbelt, not a substitute for a sane step size.

## Day 7 · The KL leash (`05_grpo_kl_reference.py`)

Loaded a *second*, frozen copy of the model as the reference, and added β·KL to the loss using DeepSeekMath's k3 estimator: `exp(ref − θ) − (ref − θ) − 1`, which is always ≥ 0.

My main bug of the day was conceptual, not code: I kept conflating the **rollout snapshot** (what the clip ratio compares against — refreshed every batch) with the **reference model** (what the KL compares against — frozen forever). They're two different log-prob comparisons doing two different jobs. Ratio: don't move too fast *this step*. KL: don't drift too far *overall*. Sanity check at epoch 0: ratio ≈ 1 AND KL ≈ 0, because at step zero the policy equals both.

## Day 8 · Group-relative advantage — GRPO complete (`06_grpo_full_loss.py`)

The last missing piece, and honestly the smallest diff of the whole project: `advantage = (rewards − mean) / (std + 1e-8)` computed over the 8 completions in the group. One line. That line is what lets you delete PPO's entire critic network — the group's mean *is* the baseline.

Ran it over multiple prompts for 20 steps and watched the pieces behave: z-scored advantages within each group, KL accumulating slowly as the policy drifts from the reference, ratio pinned near 1 at each epoch 0. When all 8 completions get the same reward the advantage collapses to exactly zero — no signal, correctly.

## Day 9 · Reward hacking, live (`07_grpo_reward_hacking.py`)

Ran the full loss for 50 steps on a "reward long answers" objective, logging smoothed reward. The curve tells the whole story (`images/07_reward_hacking.png`):

- Steps 0–38: genuine learning, smoothed reward 0.625 → 0.988. Great.
- Steps 38–50: collapse, down to 0.637.

A length reward is trivially hackable, and once the policy drifted far enough from the reference, β=0.04 wasn't a strong enough leash to stabilize it. I'd read about reward hacking plenty of times; producing it myself on a 0.5B model with a 5-prompt training set was different. This single run motivated everything in the last stage.

## Day 10 · Refactor (`08_grpo_modular.py`)

Pulled the growing script apart into three clean phases — `rollout()` (data collection, no grad), `compute_advantages()` (with a skip guard for uniform-reward groups), `update()` (K epochs, returns metrics) — because I could feel the next experiments coming and didn't want to keep editing a monolith.

The skip guard immediately exposed a real problem: **~70% of steps were being skipped** because all 8 completions got identical rewards. The length reward doesn't just get hacked — it saturates, and a saturated reward gives GRPO literally nothing (zero advantage). The fix isn't a code fix; it's a better reward.

## Days 11–13 · A verifiable reward: Countdown (`09_grpo_countdown.py`)

Switched to a task where correctness can be *checked*, not vibes-checked: given 4 numbers and a target, produce an arithmetic expression in `<answer>` tags that equals the target using each number at most once. Reward = 0.2 for correct format + 1.0 for a verified-correct expression.

Watching the 0.5B model, reward went 0.0 → ~0.15 over 35 steps — which is exactly the format reward being learned first. The model can't reliably do the arithmetic yet, but the two-component reward gives it a gradient anyway: learn the format, *then* learn correctness. That's the same emergent curriculum DeepSeek describes for R1-Zero, reproduced at toy scale.

Not everything was clean. Two real bugs from this version, both preserved in the file: my `make_prompt` had a stray comma that made it return a *tuple* instead of a string, and my reward function could crash on completions with no `<answer>` tags because the parsing wasn't fully inside the `if match` block. Also KL climbed from 0.09 to 1.89 over the run — β=0.04 was clearly too small for longer runs. All three fixed in the final script.

## Day 14 · LoRA (`10_grpo_lora.py`)

To move up to Qwen2.5-1.5B I needed the memory math to work — remember GRPO holds *two* copies of the model (policy + frozen reference). LoRA adapters on the q/v projections mean the optimizer only tracks a tiny fraction of parameters, and that was enough to fit. Only change to the GRPO code itself: the optimizer gets `[p for p in model.parameters() if p.requires_grad]` instead of everything.

## Day 15 · The final run (`11_grpo_final.py`)

Everything learned in one script: 1.5B model in bf16, LoRA (r=8), β bumped to 0.15 to hold the KL down, temperature 1.2 and 128 max tokens for more diverse rollouts, an easier problem distribution (so groups aren't uniformly-wrong), and a hardened reward function — handles `=` in answers, checks number usage against multiset availability, and has explicit sanity-check cases that run before training starts.

Reward curve for the run is in `images/11_final_reward_curve.png`, with the raw training log in `images/11_final_training_log.png`.

---

## What I'd tell someone attempting the same thing

1. **Build the failure before the fix.** I understand clipping because I watched the ratio hit 35 without it. I understand verifiable rewards because I watched a length reward get hacked and then saturate. The broken versions were the curriculum.
2. **Epoch-0 invariants are free tests.** Ratio ≈ 1 and KL ≈ 0 before any update. Every time something was wrong, one of these caught it.
3. **The loss is the easy part.** The whole GRPO objective is ~10 lines. The real work is log-prob extraction, masking, the snapshot/reference distinction, and reward design.
4. **Reward design dominates everything.** Same algorithm, same model: length reward → hacking and collapse; format+correctness reward → a stable emergent curriculum.

The 4 questions I set for myself on Day 1, answered on Day 15 without notes:

1. *Why no value network?* The group mean is the baseline.
2. *Advantage in one line?* `(reward_i − group_mean) / group_std`.
3. *What do clipping and KL each prevent?* Clipping: too-large steps. KL: too-far drift from the reference.
4. *Who programmed the aha moment?* Nobody. RL found it.

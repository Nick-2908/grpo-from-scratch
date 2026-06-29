# GRPO — Study Notes (Day 1)

> The lineage in one line:
> **REINFORCE** (raw return → high variance) → **subtract a baseline** (advantage → lower variance) → **PPO** (critic estimates the baseline + clip the update for stability) → **GRPO** (throw out the critic, get the baseline from a *group's* mean/std, keep clipping, add a KL leash).

---

## 1. Policy gradient — the foundation

In RL we want a policy that maximizes expected reward. The objective:

```
J(θ) = E_trajectory[ reward(trajectory) ]
     = Σ_trajectories  P(trajectory; θ) · reward(trajectory)
```

Take the gradient and use the **log-derivative trick** ( ∇P = P · ∇log P ):

```
∇J(θ) = Σ  ∇P(trajectory) · reward
      = Σ  P(trajectory) · ∇log P(trajectory) · reward
      = E[ ∇log π(action) · reward ]
```

**Intuition:** push up the probability of the actions/tokens that led to high reward. This is **vanilla policy gradient (REINFORCE)**.

**Problem:** using the raw reward/return makes the gradient estimate **high variance** → noisy, slow learning.

---

## 2. Baseline & advantage — fixing the variance

Subtract a **baseline** from the reward so we measure *relative* goodness, not absolute:

```
Advantage = reward − baseline
```

The advantage says *"how much better than average was this action?"* Subtracting a baseline **reduces variance** without biasing the gradient. (Correction to my first draft: REINFORCE is the high-variance method; the baseline is what *fixes* it.)

---

## 3. PPO — stability via clipping

A separate problem from variance: if each update changes the policy **too much**, training becomes **unstable**. PPO fixes the *step size*, not the variance.

PPO uses **two networks**:
- **Actor** — the policy that picks actions.
- **Critic** — estimates the baseline/value, used to compute the advantage: `A = Q(s,a) − V(s)`.

It limits the update with a **probability ratio** and **clipping**:

```
ratio = π_new(a) / π_old(a)
loss  = min( ratio · A ,  clip(ratio, 1−ε, 1+ε) · A )
```

(ε is a hyperparameter.) Clipping stops the ratio from drifting outside `[1−ε, 1+ε]`, so the policy can't jump too far in one step.

---

## 4. GRPO — drop the critic

**Problem with PPO:** the critic network costs extra memory and compute, and is fiddly to train.

**GRPO's fix:** delete the critic. Get the baseline from a **group of completions to the same prompt** instead of a learned value function.

For a prompt (e.g. `"2 + 2?"`), sample **G** completions in parallel (typically G ≈ 8), score each, then:

```
Advantage_i = ( reward_i − mean(rewards) ) / ( std(rewards) + ε )
```

The group's mean *is* the baseline. Then it's PPO from there — clip, take the min, subtract a KL penalty, and average over all tokens and all G completions:

```
loss = − mean_over(G, tokens)[ min(ratio·A,  clip(ratio,1−ε,1+ε)·A) ]  +  β · KL(π_θ ‖ π_ref)
```

### KL term — keep the policy near the reference
A frozen **reference model** (the starting model) keeps the policy from diverging too far. GRPO uses this **unbiased KL estimator** (from the DeepSeekMath paper, not the textbook KL formula):

```
KL ≈  π_ref/π_θ  −  log(π_ref/π_θ)  −  1
```

---

## 5. The LLM setting is a BANDIT — what does NOT apply

This is the part that trips people coming from a general RL course. In LLM-GRPO:

**One completion → one scalar reward, at the end.** Model writes its whole answer → you score it once (+1 correct / 0 wrong). One action, one reward. That's a **bandit** problem, not multi-step RL.

So all the multi-step credit-assignment machinery **drops out**:

| Tool | Why it's gone here |
|---|---|
| Return `G_t = R + γ·G_{t+1}` | No future stream of rewards — the episode is one completion. There is no `t+1`. |
| Discount `γ` | Nothing downstream to discount. **Not** γ=0 in the token view — see below. |
| `Q(s,a)`, `V(s)` | They value *intermediate* states; there are none with rewards. GRPO deleted the critic anyway. |

### The γ subtlety (don't write γ=0 in the token-level loss)
The loss is summed **token by token**, and GRPO gives **every token in a completion the *same* advantage** (the group-normalized reward). For one terminal reward to reach every token undecayed, that's effectively **γ = 1**, i.e. *no discounting*. Using γ = 0 in the token view would send signal to **only the last token** → wouldn't learn. So: **"no discounting," realized in code as broadcasting one advantage across all the completion's tokens** — not γ = 0.

---

## 6. What this means for the implementation

The loss is *simpler* than general RL — no discounting loop, no value network, no per-step returns:

1. Sample **G** completions per prompt.
2. Score each → one scalar reward each.
3. Group-normalize → one **advantage** per completion: `(r − mean)/std`.
4. **Broadcast** that single advantage across all tokens of its completion.
5. Per-token: `ratio = exp(logp_new − logp_old)`.
6. `min(ratio·A, clip(ratio,1−ε,1+ε)·A)`, mask padding, mean over tokens.
7. Add `β · KL(π_θ ‖ π_ref)`. Negate → minimize.

**No `γ`. No `Q`/`V`. No return-to-go. One scalar per completion, broadcast.**

---

## The 4 checks (can I explain these cleanly?)
1. **Why no value network?** → The group mean replaces it as the baseline.
2. **Advantage of one completion, one line?** → `(reward_i − group_mean) / group_std`.
3. **What do clipping and KL each prevent?** → Clipping: too-large policy steps. KL: drifting too far from the reference model.
4. **What's the aha moment, and who programmed it?** → Emergent self-correcting reasoning; nobody — RL found it.

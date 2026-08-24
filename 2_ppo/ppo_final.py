import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# -----------------------------------------------------------------------------
# 1. Models & Tokenizer Setup
# -----------------------------------------------------------------------------
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)

if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
pad_id = tok.pad_token_id
eos_id = tok.eos_token_id  # NOTE: kept separate from pad_id conceptually, even though
                            # they share the same integer value here. We use eos_id
                            # explicitly in the mask logic below so "real EOS" and
                            # "padding" are never confused, even though they're the
                            # same token id.

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)
model.train()

ref_model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# -----------------------------------------------------------------------------
# 2. Value Head (Critic Network)
# -----------------------------------------------------------------------------
class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, dtype=torch.float32):
        super().__init__()
        self.summary = nn.Linear(hidden_size, 1, bias=False, dtype=dtype)
        nn.init.normal_(self.summary.weight, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.summary(hidden_states).squeeze(-1)

value_head = ValueHead(
    model.config.hidden_size,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)
value_head.train()

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(value_head.parameters()),
    lr=1e-6
)

# Hyperparameters
epochs = 4
G = 8
clip_eps = 0.2
vf_clip = 0.2
vf_coef = 0.5
kl_beta = 0.05
gamma = 1.0
lam = 0.95

# -----------------------------------------------------------------------------
# 3. Core Helper Functions
# -----------------------------------------------------------------------------
def get_logprobs_and_values(policy_net, v_head, input_ids, attention_mask):
    outputs = policy_net(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    logps = F.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(logps, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    values = None
    if v_head is not None:
        hidden_states = outputs.hidden_states[-1][:, :-1, :]
        values = v_head(hidden_states)

    return token_logprobs, values


# ============================================================================
# FIX #1: completion mask now explicitly locates the FIRST real EOS token per
# sequence and keeps it as valid (1). Only tokens STRICTLY AFTER that first
# EOS are masked out as padding. Previously, `full_mask = (full_ids != pad_id)`
# silently zeroed out the real EOS token too (since pad_id == eos_token_id),
# which meant:
#   - lengths were off by one (undercounted the true response length)
#   - the reward got scattered one token too early
#   - the model never got gradient signal on the "decide to stop" token
# ============================================================================
def build_completion_mask(full_ids, prompt_len, eos_id):
    """
    Returns a [B, T-1] mask (already shifted to align with targets/logprobs)
    that is 1 for every real completion token INCLUDING the first EOS token,
    and 0 for the prompt and for anything after the first EOS (real padding).
    """
    B, L = full_ids.shape
    device_ = full_ids.device

    mask = torch.zeros((B, L), dtype=torch.long, device=device_)
    mask[:, prompt_len:] = 1  # everything after the prompt starts as "valid"

    is_eos = (full_ids == eos_id)
    is_eos_in_completion = is_eos.clone()
    is_eos_in_completion[:, :prompt_len] = False  # ignore any eos-id tokens inside the prompt

    has_eos = is_eos_in_completion.any(dim=1)
    first_eos_idx = torch.where(
        has_eos,
        is_eos_in_completion.float().argmax(dim=1),
        torch.full((B,), L, device=device_, dtype=torch.long)  # no EOS -> nothing extra masked
    )

    positions = torch.arange(L, device=device_).unsqueeze(0)          # [1, L]
    keep = positions <= first_eos_idx.unsqueeze(1)                     # [B, L], True up to & incl. first EOS
    mask = mask * keep.long()

    return mask[:, 1:]  # shift by 1 to align with `targets` in get_logprobs_and_values


def scatter_sparse_rewards(completion_rewards, completion_mask):
    B, seq_len = completion_mask.shape
    token_rewards = torch.zeros((B, seq_len), dtype=torch.float32, device=completion_mask.device)

    lengths = completion_mask.sum(dim=-1).long()
    last_indices = (lengths - 1).clamp(min=0).unsqueeze(-1)

    token_rewards.scatter_(dim=1, index=last_indices, src=completion_rewards.unsqueeze(-1).float())
    return token_rewards


# ============================================================================
# FIX #2: next_val is now masked per-timestep, not just zeroed at the final
# column of the whole batch tensor. Previously, `next_val = values[:, t+1] if
# t+1 < T else 0.0` meant only the batch's last column ever bootstrapped with
# zero. A short sequence that finished at t=15 in a batch padded to T=65 would
# keep chaining GAE backward using garbage critic values from padding tokens
# at positions 16-65, contaminating every advantage estimate for that sequence.
# Multiplying next_val by mask[:, t+1] means the moment we cross a sequence's
# own true last valid token, next_val correctly collapses to 0 for it.
# ============================================================================
def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    B, T = rewards.shape
    device_ = rewards.device
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=device_)

    for t in reversed(range(T)):
        if t + 1 < T:
            next_val = values[:, t + 1] * mask[:, t + 1]
        else:
            next_val = torch.zeros(B, device=device_)

        delta = rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        last_gae = last_gae * mask[:, t]
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages, mask, eps=1e-8):
    active_tokens = advantages[mask.bool()]
    mean = active_tokens.mean()
    std = active_tokens.std(unbiased=False)

    norm_adv = (advantages - mean) / (std + eps)
    return norm_adv * mask


def reward_fn(text: str) -> float:
    n_tokens = len(tok(text)["input_ids"])
    return 1.0 if n_tokens < 40 else 0.0


# -----------------------------------------------------------------------------
# 4. Training Step
# -----------------------------------------------------------------------------
def train_step():
    chat = [{"role": "user", "content": "What is 2+2"}]
    prompt_str = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_str, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids = model.generate(
            **inputs,
            max_new_tokens=65,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=G,
            pad_token_id=pad_id
        )
        full_mask = (full_ids != pad_id).long()  # still used as the attention_mask for the forward pass

        # FIX #1 applied here instead of the old inline mask construction
        completion_mask = build_completion_mask(full_ids, prompt_len, eos_id)

        old_logprobs, old_values = get_logprobs_and_values(model, value_head, full_ids, full_mask)
        ref_logprobs, _ = get_logprobs_and_values(ref_model, None, full_ids, full_mask)

        env_rewards = []
        for i in range(G):
            comp_text = tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
            env_rewards.append(reward_fn(comp_text))
        env_rewards = torch.tensor(env_rewards, device=device)

        sparse_rewards = scatter_sparse_rewards(env_rewards, completion_mask)

        token_kl = old_logprobs - ref_logprobs
        net_rewards = (sparse_rewards - (kl_beta * token_kl)) * completion_mask

        # FIX #2 applied here (inside compute_gae itself)
        advantages, returns = compute_gae(net_rewards, old_values, completion_mask, gamma=gamma, lam=lam)

        norm_advantages = normalize_advantages(advantages, completion_mask)

    for epoch in range(epochs):
        new_logprobs, new_values = get_logprobs_and_values(model, value_head, full_ids, full_mask)

        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * norm_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_advantages
        policy_loss = -torch.min(surr1, surr2)

        v_clipped = old_values + torch.clamp(new_values - old_values, -vf_clip, vf_clip)
        vf_loss1 = (new_values - returns) ** 2
        vf_loss2 = (v_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2)

        total_tokens = completion_mask.sum()
        mean_policy_loss = (policy_loss * completion_mask).sum() / total_tokens
        mean_value_loss = (value_loss * completion_mask).sum() / total_tokens

        total_loss = mean_policy_loss + (vf_coef * mean_value_loss)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(value_head.parameters()), max_norm=1.0)
        optimizer.step()

        clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * completion_mask).sum() / total_tokens
        print(f"epoch {epoch:02d} | loss: {total_loss.item():.4f} | pi_loss: {mean_policy_loss.item():.4f} | "
              f"v_loss: {mean_value_loss.item():.4f} | ratio_mean: {ratio[completion_mask.bool()].mean().item():.4f} | "
              f"clip_frac: {clip_frac.item():.4f}")


train_step()
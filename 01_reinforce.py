import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).to(device)
model.train()

pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
G = 4

def per_token_logps(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logps = F.log_softmax(logits, dim=-1)
    return torch.gather(logps, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

def reward_fn(text: str) -> float:
    # reward SHORT completions. threshold < 35 so the model hits it PART of the time.
    n_tokens = len(tok(text)["input_ids"])
    return 1.0 if n_tokens < 35 else 0.0



# ---- training step ----
def train_step(prompt: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids = model.generate(
            **inputs, max_new_tokens=40, do_sample=True, temperature=0.8,
            num_return_sequences=G, pad_token_id=pad_id)
    full_mask = (full_ids != pad_id).long()

    rewards = []
    for i in range(G):
        comp = tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        rewards.append(reward_fn(comp))
    rewards = torch.tensor(rewards, device=device)

    completion_mask = torch.zeros_like(full_ids)
    completion_mask[:, prompt_len:] = 1
    completion_mask = (completion_mask * full_mask)[:, 1:]

    logps = per_token_logps(full_ids, full_mask)
    seq_logp = (logps * completion_mask).sum(dim=-1)

    loss = -(rewards * seq_logp).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), rewards.mean().item()

# ---- train: print EVERY step so movement is visible ----
for step in range(120):
    loss, avg_r = train_step("hello how are you?")
    print(f"step {step:4d} | loss {loss:8.4f} | avg_reward {avg_r:.3f}")
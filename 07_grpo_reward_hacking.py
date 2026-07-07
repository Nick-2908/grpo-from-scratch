import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


prompts=[
    "what is neural network?",
    "Count to three",
    "Name a fruit",
    "what is 5+4?",
    "tell me about india in 30 words"
]

name="Qwen/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(name)

model=AutoModelForCausalLM.from_pretrained(name).to(device)
model.train()

ref_model=AutoModelForCausalLM.from_pretrained(name).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad=False

epochs=4
G=8
E=0.25
beta=0.04

optimizer=torch.optim.Adam(model.parameters(),lr=1e-6)
pad_id= tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def per_token_logps(model,input_ids,attention_mask):
    outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits[:,:-1,:]
    targets=input_ids[:,1:]
    logps=F.log_softmax(logits,dim=-1)
    return torch.gather(logps,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)

def reward_fn(text:str)->float:
    no_token=len(tok(text)["input_ids"])
    return 1.0 if no_token>55 else 0.0

def cal_k1(new_policy,ref_policy):
    diff=ref_policy-new_policy
    return torch.exp(diff)-diff-1

reward_log=[]

def train_steps(prompt,step):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt").to(device)
    prompt_len=inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids=model.generate(**inputs,do_sample=True,max_new_tokens=65,temperature=0.8,num_return_sequences=G, pad_token_id=pad_id)
        full_mask=(full_ids!=pad_id).long()

    completion_mask = torch.zeros_like(full_ids)
    completion_mask[:, prompt_len:] = 1
    completion_mask = (completion_mask * full_mask)[:, 1:]

    with torch.no_grad():
        old_logp=per_token_logps(model,full_ids,full_mask)
        old_seq_logp=(completion_mask*old_logp).sum(dim=-1)
        ref_logp=per_token_logps(ref_model,full_ids,full_mask)

    rewards=[]
    for i in range(G):
        comp=tok.decode(full_ids[i,prompt_len:],skip_special_tokens=True)
        rewards.append(reward_fn(comp))
    rewards=torch.tensor(rewards,device=device)
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    reward_log.append(rewards.mean().item())

    for epoch in range(epochs):
        new_logp=per_token_logps(model,full_ids,full_mask)
        new_seq_logp=(completion_mask*new_logp).sum(dim=-1)
        k1=cal_k1(new_logp,ref_logp)
        k1=(completion_mask*k1).sum(dim=-1)

        ratio=torch.exp(new_seq_logp-old_seq_logp)
        clip_loss=(torch.clamp(ratio,1-E,1+E))*advantage
        unclipped_loss=ratio*advantage
        objective=torch.min(clip_loss,unclipped_loss)
        loss=-(objective).mean()+beta*k1.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
       
    print(f"\n--- step {step} ---")
    print(f"  prompt: {prompt!r}")
    print(f"  rewards: {rewards.tolist()}")
    print(f"  reward mean: {rewards.mean().item():.3f}")
    print(f"  advantage std: {advantage.std().item():.3f}")

        
    if epoch == epochs - 1:
        print(f"  [final epoch] loss {loss.item():.4f} | ratio mean {ratio.mean().item():.4f} | kl {k1.mean().item():.4f}")


def train(n_steps=50):
    for step in range(n_steps):
        prompt = random.choice(prompts)
        train_steps(prompt, step)

train()

print(f"\nfinal avg reward (last 10 steps): {sum(reward_log[-10:]) / 10:.3f}")
print(f"full reward log: {[round(r, 3) for r in reward_log]}")



window = 10
smoothed = [sum(reward_log[max(0,i-window):i+1]) / min(i+1, window) 
            for i in range(len(reward_log))]
print("\nsmoothed reward (10-step window):")
print([round(r, 3) for r in smoothed])

'''Smoothed reward showed genuine learning (0.625 → 0.988 over 38 steps) followed by collapse (→ 0.637 over last 12 steps). 
Classic reward hacking + policy collapse arc. Cause: length reward is easily hacked, and after enough steps the policy drifted too
 far from the reference for KL penalty at beta=0.04 to stabilize it. Fix: verifiable reward (Day 11) + potentially larger beta.'''
#visualization in images/07_reward_hacking.png

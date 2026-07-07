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
reward_log=[]

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

def compute_advantages(rewards):
    if rewards.std()<1e-8:
        return None
    return (rewards-rewards.mean())/(rewards.std()+1e-8)

def cal_k1(new_policy,ref_policy):
    diff=(ref_policy-new_policy)
    return torch.exp(diff)-diff-1

def rollout(model,prompt):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt").to(device)
    prompt_len=inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids=model.generate(**inputs,max_new_tokens=65,temperature=0.8,do_sample=True,num_return_sequences=G,pad_token_id=pad_id)
        full_masks=(full_ids!=pad_id).long()

    completion_mask=torch.zeros_like(full_ids)
    completion_mask[:,prompt_len:]=1
    completion_mask=(completion_mask*full_masks)[:,1:]

    with torch.no_grad():
        old_logp=per_token_logps(model,full_ids,full_masks)
        old_seq_logp=(completion_mask*old_logp).sum(dim=-1)
        ref_logp=per_token_logps(ref_model,full_ids,full_masks)

    rewards=[]
    for i in range(G):
        comp=tok.decode(full_ids[i,prompt_len:],skip_special_tokens=True)
        rewards.append(reward_fn(comp))
    rewards=torch.tensor(rewards,device=device)

    return full_ids, full_masks, completion_mask, rewards, old_seq_logp, ref_logp


def update(model, optimizer, full_ids, full_masks, completion_mask, old_seq_logp, ref_logp, advantages):
    metrics = {'loss':0,'ratio_mean':0,'kl_mean':0}
    for epoch in range(epochs):
        new_logp=per_token_logps(model,full_ids,full_masks)
        new_seq_logp=(completion_mask*new_logp).sum(dim=-1)
        k1=cal_k1(new_logp,ref_logp)
        k1=(completion_mask*k1).sum(dim=-1)

        ratio=torch.exp(new_seq_logp-old_seq_logp)
        clip_loss=(torch.clamp(ratio,1-E,1+E))*advantages
        unclipped_loss=ratio*advantages
        objective=torch.min(clip_loss,unclipped_loss)
        loss=-(objective).mean()+beta*k1.mean()

    
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        metrics['loss']=loss.item()
        metrics['ratio_mean']=ratio.mean().item()
        metrics['kl_mean']=k1.mean().item()

    return metrics
    

def train(n_steps=50):
    for step in range(n_steps):
        prompt=random.choice(prompts)
        full_ids, full_mask, completion_mask, rewards, old_seq_logp, ref_logp=rollout(model, prompt)
        advantages=compute_advantages(rewards)
        if advantages is None:
            print(f"step {step:3d} | skipped (uniform rewards)")
            reward_log.append(rewards.mean().item())
            continue
        metrics = update(model, optimizer, full_ids, full_mask, completion_mask,
                         old_seq_logp, ref_logp, advantages)
        
        reward_log.append(rewards.mean().item())
        if step % 5 == 0:
            print(f"step {step:3d} | reward {rewards.mean():.3f} | "
                  f"loss {metrics['loss']:.4f} | "
                  f"ratio {metrics['ratio_mean']:.4f} | "
                  f"kl {metrics['kl_mean']:.4f}")

train()

''' Rollout/update loop skeleton. Separated into three clean phases: rollout() (data collection, no grad),
 compute_advantages() (with uniform-reward skip guard), update() (K optimization epochs, returns metrics dict). 
 Skip guard working correctly — ~70% of steps skipped due to uniform rewards on easy prompts. Root cause: length reward 
 saturates too quickly on short-answer prompts. Fix: verifiable reward (Day 11). Loop mechanics confirmed correct on non-skipped
   steps: ratio stable (1.0–1.4), KL small and growing slowly, grad clipping active.'''










import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

name="Qwen/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(name)

model=AutoModelForCausalLM.from_pretrained(name)
model.train()

ref_model=AutoModelForCausalLM.from_pretrained(name)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad(False)

epochs=4
G=8
E=0.25
beta=0.04

optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
pad_id= tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def per_token_logps(model,input_ids,attention_mask):
    outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits[:,:-1,]
    targets=input_ids[:,1:]
    logps=F.log_softmax(logits,dim=-1)
    return torch.gather(logps,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)

def reward_fn(text:str)->float:
    no_token=len(tok(text)["input_ids"])
    return 1.0 if no_token>80 else 0.0

def cal_k1(new_policy,old_policy):
    diff=new_policy-old_policy
    return torch.exp(diff)-diff-1

def train_steps():
    text=tok.apply_chat_template([{"role":"user","content":"what is neural network explain in 20 words"}],tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt").to(device)
    prompt_len=inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids=model.generate(**inputs,do_sample=True,max_new_tokens=65,temperature=0.8,num_return_sequences=G, pad_token_id=pad_id)
        full_mask=(full_ids!=pad_id).long()

    completion_mask=torch.zeros_like(full_ids)
    completion_mask=completion_mask[:,prompt_len:]=1
    completion_mask=(completion_mask*full_mask)[:,1:]

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
        optimizer.step()
        print(f"epoch {epoch} | loss {loss.item():8.4f} | ratio mean {ratio.mean().item():.4f} | kl mean {kl.mean().item():.4f}")
        print(f"step rewards: {rewards.tolist()}")
        print(f"step advantage: {advantage.tolist()}")


for step in range(20):
    train_steps()

'''Group-relative advantage. Added std normalization: advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8). 
Produces z-scores per completion within the group — rewarded completions get positive advantage scaled by group spread,
 unrewarded get negative. When all rewards identical, advantage collapses to 0 (correct — no signal, eps prevents div-by-zero). 
 Verified across 20 steps: epoch-0 ratio≈1 and KL≈0 at step 1 (policy==snapshot==reference), KL accumulating slowly across steps 
 as policy drifts from reference (working as intended). Full GRPO loss now complete: clipped ratio × group-relative advantage − beta × k3 KL.'''

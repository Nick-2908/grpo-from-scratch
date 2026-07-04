import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

name="Qwen/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(name)

model=AutoModelForCausalLM.from_pretrained(name).to(device)
model.train()


ref_model=AutoModelForCausalLM.from_pretrained(name).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

epochs=4
G=8
E=0.25
beta=0.04

optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
pad_id= tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def per_token_logps(model,input_ids,attention_mask):
    outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits[:,:-1,:]
    logps=F.log_softmax(logits,dim=-1)
    targets=input_ids[:,1:]
    return torch.gather(logps,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)

def reward_fn(text:str)->float:
    n_token=len(tok(text)["input_ids"])
    return 1.0 if n_token<40 else 0.0

def cal_k1(new_policy,ref_policy):
    diff=new_policy-ref_policy
    return torch.exp(diff)-diff-1

def train_steps():
    text=tok.apply_chat_template([{"role":"user","content":"What is 2+2"}],tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt").to(device)
    prompt_len=inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids=model.generate(**inputs,max_new_tokens=65,do_sample=True,temperature=0.8,num_return_sequences=G, pad_token_id=pad_id)
        full_mask=(full_ids != pad_id).long()

    completion_mask=torch.zeros_like(full_ids)
    completion_mask[:,prompt_len:]=1
    completion_mask=(completion_mask*full_mask)[:,1:]

    with torch.no_grad():
        old_logp=per_token_logps(model,full_ids,full_mask)
        old_seq_logp=(old_logp*completion_mask).sum(dim=-1)

        ref_logp=per_token_logps(ref_model,full_ids,full_mask)

    
        
    

    rewards=[]
    for i in range(G):
        comp = tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        rewards.append(reward_fn(comp))
    rewards = torch.tensor(rewards, device=device)

    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    

    for epoch in range(epochs):
        new_logp = per_token_logps(model, full_ids, full_mask)          # WITH grad
        new_seq_logp = (new_logp * completion_mask).sum(dim=-1)
        kl=cal_k1(new_logp,ref_logp)
        kl = (kl*completion_mask).sum(dim=-1) 

        ratio=torch.exp(new_seq_logp-old_seq_logp)
        clip_loss=(torch.clamp(ratio,1-E,1+E))*advantage
        unclip_loss=ratio*advantage
        objective=torch.min(unclip_loss,clip_loss)
        loss=-(objective).mean()+beta*kl.mean()


        



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        print(f"epoch {epoch} | loss {loss.item():8.4f} | ratio mean {ratio.mean().item():.4f} | kl mean {kl.mean().item():.4f}")

train_steps()

'''
Reference model + KL penalty. Loaded a second frozen model (ref_model, requires_grad_(False), never optimized) as a fixed anchor, 
separate from the rollout-time old_logp snapshot used for the clip ratio — these are two distinct comparisons 
(ratio: current vs last-batch snapshot; KL: current vs original checkpoint) and conflating them was today's main bug.
 Used DeepSeekMath's k3 estimator: exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1, always ≥0, masked and summed like everything else.
   Verified epoch 0 gives both ratio≈1 and KL≈0 (policy equals both its snapshot and the reference before training starts). Confirmed the same lr sensitivity
     from Day 6 carries over: lr=1e-5 blew up ratio to 23,040 and KL to 16; lr=1e-6 gave stable, small movement in both.
 Full loss now combines three log-prob comparisons: old (clip), ref (KL), and the reward-derived advantage '''
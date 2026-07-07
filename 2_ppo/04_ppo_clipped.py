import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

name="Qwen/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name).to(device)
model.train()
epochs=4
G=8
E=0.25

optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
pad_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def per_token_logps(model,input_ids,attention_mask):
    outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits[:,:-1,:]
    targets=input_ids[:,1:]
    logps=F.log_softmax(logits,dim=-1)
    return torch.gather(logps,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)

def reward_fn(text:str)->float:
    n_tokens=len(tok(text)["input_ids"])
    return 1.0 if n_tokens<40 else 0.0



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

    rewards=[]
    for i in range(G):
        comp = tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        rewards.append(reward_fn(comp))
    rewards = torch.tensor(rewards, device=device)

    advantage=rewards-rewards.mean()
    

    for epoch in range(epochs):
        new_logp = per_token_logps(model, full_ids, full_mask)          # WITH grad
        new_seq_logp = (new_logp * completion_mask).sum(dim=-1)

        ratio = torch.exp(new_seq_logp - old_seq_logp)
        clip_loss=(torch.clamp(ratio,1-E,1+E))*advantage
        unclip_loss=ratio*advantage
        loss=-(torch.min(clip_loss,unclip_loss)).mean()



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch} | loss {loss.item():8.4f} | ratio mean {ratio.mean().item():.4f} | ratio std {ratio.std().item():.4f}")

train_steps()

'''Clipped surrogate objective. loss = -min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage).mean(). Confirmed epoch 0 
ratio ≈ 1.0 (snapshot correct, matches Day 5). At lr=1e-5, clipping alone was insufficient — ratio still exploded to 29,568 by 
epoch 3, because clipping only removes gradient in the direction that inflates the objective for positive-advantage samples; 
the pessimistic branch for negative-advantage samples stays unclipped by design. Dropping to lr=1e-6 produced stable, 
smooth ratio drift (1.0 → 0.65 across 4 epochs). Lesson: clip range and learning rate are coupled — clipping is not a substitute for a 
sane step size.'''
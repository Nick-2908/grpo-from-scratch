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

optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def per_token_logps(model,input_ids,attention_mask):
    outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits[:,:-1,:]
    target=input_ids[:,1:]
    logps=F.log_softmax(logits,dim=-1)
    return torch.gather(logps,dim=-1,index=target.unsqueeze(-1)).squeeze(-1)

def reward_function(text:str)->float:
    n_tokens = len(tok(text)["input_ids"])
    return 1.0 if n_tokens < 35 else 0.0

def train_steps():
    text=tok.apply_chat_template([{"role":"user","content":"Hi how are you"}],tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt").to(device)
    prompt_len=inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids=model.generate(**inputs,max_new_tokens=40,temperature=0.8,do_sample=True,num_return_sequences=G, pad_token_id=pad_id)
        full_mask=(full_ids!=pad_id).long()


    completion_mask = torch.zeros_like(full_ids)
    completion_mask[:, prompt_len:] = 1
    completion_mask = (completion_mask * full_mask)[:, 1:]

    with torch.no_grad():
        old_logp=per_token_logps(model,full_ids,full_mask)
        old_seq_logp=(old_logp*completion_mask).sum(dim=-1)

    rewards=[]
    for i in range(G):
        comp = tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        rewards.append(reward_function(comp))
    rewards = torch.tensor(rewards, device=device)

    advantage=rewards-rewards.mean()

    for epoch in range(epochs):
        new_logp = per_token_logps(model, full_ids, full_mask)          # WITH grad
        new_seq_logp = (new_logp * completion_mask).sum(dim=-1)

        ratio = torch.exp(new_seq_logp - old_seq_logp)
        loss = -(ratio * advantage).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch} | loss {loss.item():8.4f} | ratio mean {ratio.mean().item():.4f} | ratio std {ratio.std().item():.4f}")

train_steps()


    
#Takeaways:
'''old log-probs and the importance ratio. Verified ratio≈1 at epoch 0 (snapshot correct). Multiple optimization epochs on
one rollout batch without any constraint caused the ratio to explode (1.0 → 35.0 over 3 epochs) — direct empirical demonstration 
of why PPO/GRPO need clipping. Key insight: log-space subtraction + exp = probability-space division (numerical stability, avoids
underflow from multiplying many small probabilities).'''




    

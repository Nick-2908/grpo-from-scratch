import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).to(device)
model.train()
gnorms_no_baseline=[]
optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
G=8
use_baseline=False


def per_token_logpb(input_ids,attention_mask):
    outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits[:,:-1,:]
    target=input_ids[:,1:]
    logps=logits.log_softmax(dim=-1)
    return torch.gather(logps,dim=-1,index=target.unsqueeze(-1)).squeeze(-1)

def reward_fn(text:str)->float:
    n_tokens = len(tok(text)["input_ids"])
    return 1.0 if n_tokens < 35 else 0.0

def grad_global_norm(model):
    #using L2 norm
    sq=0.0
    for p in model.parameters():
        if p.grad is not None:
            sq+=p.grad.pow(2).sum().item()

    return sq**0.5




def train_Steps():
    text=tok.apply_chat_template([{"role":"user","content":"what is neural network?"}],tokenize=False,add_generational_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        full_ids=model.generate(**inputs,max_tokens=65,do_sample=True,temperature=0.4,
            num_return_sequences=G, pad_token_id=pad_id)
        full_mask=(full_ids!=pad_id).long()


    rewards = []
    for i in range(G):
        comp = tok.decode(full_ids[i, prompt_len:], skip_special_tokens=True)
        rewards.append(reward_fn(comp))
    rewards = torch.tensor(rewards, device=device)

    completion_mask = torch.zeros_like(full_ids)
    completion_mask[:, prompt_len:] = 1
    completion_mask = (completion_mask * full_mask)[:, 1:]

    logps = per_token_logpb(full_ids, full_mask)
    seq_logp = (logps * completion_mask).sum(dim=-1)

    if use_baseline:
        advantage=rewards-rewards.mean()
    else: 
        advantage=rewards


    loss = -(advantage * seq_logp).mean()
    optimizer.zero_grad()
    loss.backward()
    gnorm = grad_global_norm(model) 
    optimizer.step()
    return loss.item(), rewards.mean().item(),gnorm


for step in range(120):
    loss, avg_r ,gnorm= train_Steps()
    print(f"step {step:4d} | loss {loss:8.4f} | avg_reward {avg_r:.3f}")
    gnorms_no_baseline.append(gnorm)     



import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import re
from peft import get_peft_model , LoraConfig
import random


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

name="Qwen/Qwen2.5-1.5B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.bfloat16).to(device)
lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model=get_peft_model(model,lora_config)


ref_model=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.bfloat16).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

epochs=4
G=8
E=0.25
beta=0.15
reward_log=[]

optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-6)
pad_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def generate_problem():
    numbers = random.sample(range(1, 6), 4)
    a, b, c, d = numbers
    candidates = [
        a + b, a + c, a + d,
        b + c, b + d, c + d,
        a * b, a + b + c,
    ]
    valid = [t for t in candidates if 1 <= t <= 15]
    if not valid:
        return generate_problem()
    target = random.choice(valid)
    return numbers, target

def compute_advantage(rewards):
     if rewards.std()<1e-8:
          return torch.zeros_like(rewards)
     return (rewards-rewards.mean())/(rewards.std()+1e-8)

def per_token_logps(model,input_ids,attention_mask):
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        outputs=model(input_ids=input_ids,attention_mask=attention_mask)
    logits=outputs.logits[:,:-1,:]
    targets=input_ids[:,1:]
    logps=F.log_softmax(logits,dim=-1)
    return torch.gather(logps,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)



def reward_fn(completion: str, numbers: list, target: int) -> float:
    format_reward = 0.0
    correctness_reward = 0.0
    match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if match:
        format_reward = 0.2
        expr = match.group(1).strip()
        expr = expr.split('=')[0].strip()
        try:
            result = eval(expr)
            if abs(result - target) < 1e-6:
                used = [int(x) for x in re.findall(r'\d+', expr)]
                if all(used.count(n) <= numbers.count(n) for n in used) and set(used).issubset(set(numbers)):
                    correctness_reward = 1.0
        except:
            pass
    return format_reward + correctness_reward

def make_prompt(numbers: list, target: int) -> str:
    return (
        f"Using the numbers {numbers}, create an arithmetic expression "
        f"that equals {target}. "
        f"You may use +, -, *, / and each number at most once. "
        f"Write your final answer inside <answer>...</answer> tags."
    )

def cal_k1(new_policy,ref_policy):
     diff=ref_policy-new_policy
     return torch.exp(diff)-diff-1

def rollout(model,prompt,numbers,target):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt").to(device)
    prompt_len=inputs["input_ids"].shape[1]

    with torch.no_grad():
          full_ids=model.generate(**inputs,max_new_tokens=128,temperature=1.2,do_sample=True,num_return_sequences=G,pad_token_id=pad_id)
          full_mask=(full_ids!=pad_id).long()

    completion_mask=torch.zeros_like(full_ids)
    completion_mask[:,prompt_len:]=1
    completion_mask=(completion_mask*full_mask)[:,1:]

    with torch.no_grad():
        old_logp=per_token_logps(model,full_ids,full_mask)
        old_seq_logp=(completion_mask*old_logp).sum(dim=-1)
        ref_logp=per_token_logps(ref_model,full_ids,full_mask)

        rewards=[]
    for i in range(G):
        comp=tok.decode(full_ids[i,prompt_len:],skip_special_tokens=True)
        rewards.append(reward_fn(comp,numbers,target))
    rewards=torch.tensor(rewards,device=device)

    return full_ids, full_mask, completion_mask, rewards, old_seq_logp, ref_logp


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

def train(n_steps=100):
    for step in range(n_steps):
        numbers, target = generate_problem()
        prompt = make_prompt(numbers, target)
        full_ids, full_masks, completion_mask, rewards, old_seq_logp, ref_logp = rollout(
            model, prompt, numbers, target)
        advantages = compute_advantage(rewards)
        metrics = update(model, optimizer, full_ids, full_masks, completion_mask,
                         old_seq_logp, ref_logp, advantages)
        reward_log.append(rewards.mean().item())
        if step % 5 == 0:
            print(f"step {step:3d} | reward {rewards.mean():.3f} | "
                  f"loss {metrics['loss']:.4f} | "
                  f"ratio {metrics['ratio_mean']:.4f} | "
                  f"kl {metrics['kl_mean']:.4f}")

print("reward_fn sanity checks:")
print(reward_fn("<answer>1 + 2</answer>", [1, 2, 3, 4], 3))      # expect 1.2
print(reward_fn("<answer>1 + 2 + 5</answer>", [1, 2, 3, 4], 3))  # expect 0.2 (5 not in numbers)
print(reward_fn("no tags here", [1, 2, 3, 4], 3))                 # expect 0.0

train()

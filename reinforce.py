import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)         
model = AutoModelForCausalLM.from_pretrained(name)  
model.eval()

def per_token_logps(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]                      
    targets = input_ids[:, 1:]                             
    logps = F.log_softmax(logits, dim=-1)                  
    return torch.gather(logps, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  

messages = [{"role": "user", "content": "Hello, how are you?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt", padding=True)

prompt_len = inputs["input_ids"].shape[1]  

with torch.no_grad():                        
    full_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )                                      


pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
full_mask = (full_ids != pad_id).long()


completion_mask = torch.zeros_like(full_ids)
completion_mask[:, prompt_len:] = 1
completion_mask = completion_mask * full_mask

logps = per_token_logps(full_ids, full_mask)   


completion_mask = completion_mask[:, 1:]       

masked_logps = logps * completion_mask       

seq_logp = masked_logps.sum(dim=-1)            

print("per-token logps shape:", logps.shape)
print("completion tokens per sequence:", completion_mask.sum(dim=-1))
print("sum of completion log-probs:", seq_logp)
import torch
from datasets import load_dataset
import math
from tqdm import tqdm

MAX_LENGTH = 128

def compute_perplexity(model, tokenizer, dataset_name="Salesforce/wikitext", 
                       name="wikitext-2-raw-v1", 
                       split="test"):
    print(f"Измеряем perplexity на {dataset_name}...")

    ds = load_dataset(dataset_name, name, split=split, streaming=False)

    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(ds), batch_size), desc="Processing batches"):
            batch_texts = [ds[j]['text'] for j in range(i, min(i+batch_size, len(ds)))]
            
            batch_texts = [text for text in batch_texts if text and text.strip()]
            if not batch_texts:
                continue
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=MAX_LENGTH, 
                padding=True
            ).to(device)

            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss  
            
            batch_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    ppl = math.exp(total_loss / total_tokens)
    return ppl
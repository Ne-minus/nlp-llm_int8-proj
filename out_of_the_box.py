import torch
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import gc

def clear_memory():
    """Очистка памяти"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def evaluate_perplexity(model, tokenizer, texts, max_length=512):
    """Измерение perplexity на заданных текстах"""
    print("Measuring perplexity...")
    
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    batch_size = 8  # Уменьшено для квантованных моделей
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
            batch_texts = texts[i:min(i + batch_size, len(texts))]
            batch_texts = [text for text in batch_texts if text and text.strip()]
            if not batch_texts:
                continue
            
            inputs = tokenizer(
                batch_texts, return_tensors="pt", truncation=True,
                max_length=max_length, padding=True
            ).to(device)
            
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100
            
            outputs = model(**inputs, labels=labels)
            
            batch_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * batch_tokens
            total_tokens += batch_tokens
            
            # Очистка после каждого батча
            del inputs, labels, outputs
            clear_memory()
    
    ppl = math.exp(total_loss / total_tokens)
    
    return {
        "perplexity": ppl,
        "total_tokens": total_tokens,
    }

def load_c4_sample(num_samples=100):
    """Загрузка 100 семплов из C4"""
    print("Loading C4 dataset...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
    texts = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        if 'text' in item and item['text'] and len(item['text'].strip()) > 50:
            texts.append(item['text'])
    
    print(f"Loaded {len(texts)} samples from C4")
    return texts

# Конфигурация для INT8 квантования
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# Список Qwen моделей разных размеров (base версии для perplexity)
qwen_models = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B", 
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B"}

# Загрузка датасета
c4_texts = load_c4_sample(100)

results = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

for size, model_name in qwen_models.items():
    print(f"\n{'='*60}")
    print(f"Testing {model_name} ({size})")
    print('='*60)
    
    try:
        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Загрузка модели в INT8
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Оценка perplexity
        result = evaluate_perplexity(model, tokenizer, c4_texts)
        results[size] = result
        
        print(f"Perplexity: {result['perplexity']:.2f}")
        print(f"Total tokens: {result['total_tokens']:,}")
        
        # Очистка
        del model, tokenizer
        clear_memory()
        
    except Exception as e:
        print(f"Error with {model_name}: {str(e)}")
        results[size] = {"error": str(e)}
        clear_memory()

# Вывод итоговой таблицы
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print("| Model Size | Perplexity | Total Tokens |")
print("|------------|------------|--------------|")
for size, result in results.items():
    if "error" in result:
        print(f"| {size:^10} | ERROR      | {result['error'][:30]:<12} |")
    else:
        print(f"| {size:^10} | {result['perplexity']:.2f:^10} | {result['total_tokens']:,^12} |")

print("="*80)

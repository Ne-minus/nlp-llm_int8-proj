import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import gc
from collections import defaultdict
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import gc
from collections import defaultdict

def get_c4_texts(split="validation", n_samples=512, seq_len=512):
    """Потоковая загрузка текстов из C4 без полной загрузки в память."""
    ds = load_dataset("allenai/c4", "realnewslike", split=split, streaming=True)
    texts = []
    for ex in ds:
        texts.append(ex["text"])
        if len(texts) >= n_samples:
            break
    return texts

def load_qwen(model_name, device="cuda"):
    """Загрузка модели Qwen с оптимизациями для GPU."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model

# Список моделей Qwen3 для анализа
qwen_model_names = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

@torch.no_grad()
def compute_perplexity(texts, tok, model, seq_len=512):
    """Вычисление перплексии на текстах."""
    losses = []
    for txt in tqdm(texts, desc="Computing perplexity", leave=False):
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=seq_len)
        input_ids = enc.input_ids.to(model.device)
        attn_mask = enc.attention_mask.to(model.device)
        out = model(input_ids, attention_mask=attn_mask, labels=input_ids)
        losses.append(out.loss.item())
    return math.exp(float(np.mean(losses)))

import torch
import math
import gc
from tqdm.auto import tqdm
from collections import defaultdict

import torch
import gc
from tqdm.auto import tqdm
from collections import defaultdict


def analyze_outliers(texts, tokenizer, model, threshold=6.0, min_layer_frac=0.25, 
                     min_token_frac=0.06, max_texts=None, batch_size=4):
    """
    Анализ outliers согласно статье.
    """
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    if max_texts:
        texts = texts[:max_texts]
    
    dim_layers = defaultdict(set)
    dim_tokens = defaultdict(set)
    
    global_sample_idx = 0
    total_tokens = 0
    batch_data = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            with torch.no_grad():
                hidden = input[0] if isinstance(input, tuple) else input
                
                if hidden.dim() != 3:
                    return
                
                outlier_mask = (hidden.abs() >= threshold)
                
                if outlier_mask.any():
                    batch_data.append({
                        'layer_idx': layer_idx,
                        'outlier_mask': outlier_mask.cpu()
                    })
        
        return hook
    
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp'):
            handles.append(layer.mlp.register_forward_hook(make_hook(layer_idx)))
    
    def process_batch(batch_start_idx, attention_mask):
        nonlocal batch_data
        
        for data in batch_data:
            layer_idx = data['layer_idx']
            outlier_mask = data['outlier_mask']
            _, _, h_size = outlier_mask.shape
            
            for d in range(min(h_size, hidden_size)):
                dim_mask = outlier_mask[:, :, d]
                
                if not dim_mask.any():
                    continue
                
                dim_layers[d].add(layer_idx)
                
                positions = torch.where(dim_mask)
                for b, t in zip(positions[0].tolist(), positions[1].tolist()):
                    if t < attention_mask.shape[1] and attention_mask[b, t] == 1:
                        dim_tokens[d].add((batch_start_idx + b, t))
        
        batch_data.clear()
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            attention_mask = inputs['attention_mask']
            total_tokens += attention_mask.sum().item()
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_data.clear()
            
            with torch.no_grad():
                model(**inputs)
            
            process_batch(global_sample_idx, attention_mask.cpu())
            global_sample_idx += len(batch_texts)
            
            del inputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
    finally:
        for h in handles:
            h.remove()
    
    # ===== Анализ =====
    min_layers_required = max(1, int(min_layer_frac * n_layers))
    min_tokens_required = max(1, int(min_token_frac * total_tokens))
    
    emergent_dims = []
    for d in range(hidden_size):
        n_layers_d = len(dim_layers[d])
        n_tokens_d = len(dim_tokens[d])
        
        if n_layers_d >= min_layers_required and n_tokens_d >= min_tokens_required:
            emergent_dims.append(d)
    
    # ===== ПРАВИЛЬНЫЕ МЕТРИКИ =====
    
    # 1. Процент emergent dimensions от всех dimensions
    pct_emergent_dims = 100.0 * len(emergent_dims) / hidden_size
    
    # 2. Средний % слоёв на один emergent dimension
    if emergent_dims:
        avg_layer_coverage = sum(len(dim_layers[d]) for d in emergent_dims) / len(emergent_dims)
        pct_avg_layers = 100.0 * avg_layer_coverage / n_layers
    else:
        pct_avg_layers = 0.0
    
    # 3. Средний % токенов на один emergent dimension  
    if emergent_dims:
        avg_token_coverage = sum(len(dim_tokens[d]) for d in emergent_dims) / len(emergent_dims)
        pct_avg_tokens = 100.0 * avg_token_coverage / total_tokens
    else:
        pct_avg_tokens = 0.0
    
    # 4. Общее покрытие (layer, dim) пар с outliers
    total_layer_dim_pairs = n_layers * hidden_size
    pairs_with_outliers = sum(len(dim_layers[d]) for d in range(hidden_size))
    pct_layer_dim_coverage = 100.0 * pairs_with_outliers / total_layer_dim_pairs
    
    # 5. Покрытие только для emergent dims
    if emergent_dims:
        emergent_pairs = sum(len(dim_layers[d]) for d in emergent_dims)
        pct_emergent_coverage = 100.0 * emergent_pairs / (n_layers * len(emergent_dims))
    else:
        pct_emergent_coverage = 0.0
    
    return {
        # Основные метрики (для графиков как в статье)
        "pct_emergent_dims": pct_emergent_dims,      # % dimensions которые emergent
        "pct_layer_dim_coverage": pct_layer_dim_coverage,  # % (layer,dim) пар с outliers
        
        # Дополнительные метрики
        "pct_avg_layers": pct_avg_layers,            # Средний % слоёв для emergent dim
        "pct_avg_tokens": pct_avg_tokens,            # Средний % токенов для emergent dim
        "pct_emergent_coverage": pct_emergent_coverage,
        
        # Абсолютные значения
        "n_emergent_dims": len(emergent_dims),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "total_tokens": total_tokens,
        "emergent_dims": emergent_dims,
    }


def run_experiment():
    """Запуск эксперимента."""
    texts = get_c4_texts(n_samples=128, seq_len=512)
    results = []

    for name in tqdm(qwen_model_names, desc="Processing models"):
        print(f"\n=== {name} ===")
        tok, model = load_qwen(name)
        ppl = compute_perplexity(texts, tok, model)
        stats = analyze_outliers(texts, tok, model)
        n_params = sum(p.numel() for p in model.parameters()) / 1e9

        results.append({
            "model": name,
            "params_b": n_params,
            "perplexity": ppl,
            "pct_emergent_dims": stats["pct_emergent_dims"],
            "pct_layer_dim_coverage": stats["pct_layer_dim_coverage"],
            "pct_avg_layers": stats["pct_avg_layers"],
            "pct_avg_tokens": stats["pct_avg_tokens"],
            "n_emergent_dims": stats["n_emergent_dims"],
            "hidden_size": stats["hidden_size"],
        })
        
        print(f"Params: {n_params:.2f}B, PPL: {ppl:.2f}")
        print(f"Emergent dims: {stats['n_emergent_dims']}/{stats['hidden_size']} ({stats['pct_emergent_dims']:.1f}%)")
        print(f"Layer-dim coverage: {stats['pct_layer_dim_coverage']:.1f}%")
        print(f"Avg layers per emergent dim: {stats['pct_avg_layers']:.1f}%")
        print(f"Avg tokens per emergent dim: {stats['pct_avg_tokens']:.1f}%")
        
        # Освобождаем память
        del tok, model
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_results(exp_results):
    """Построение графиков."""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    
    if len(exp_results) == 0:
        print("Нет данных")
        return
    
    by_params = sorted(exp_results, key=lambda x: x["params_b"])
    
    params = np.array([r["params_b"] for r in by_params])
    ppl = np.array([r["perplexity"] for r in by_params])
    pct_emergent = np.array([r["pct_emergent_dims"] for r in by_params])
    pct_coverage = np.array([r["pct_layer_dim_coverage"] for r in by_params])
    n_emergent = np.array([r["n_emergent_dims"] for r in by_params])
    model_names = [r["model"].split("/")[-1] for r in by_params]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # === График 1: % emergent dims vs размер модели ===
    ax1 = axes[0, 0]
    ax1.scatter(params, pct_emergent, s=100, c='steelblue', zorder=5)
    for i, name in enumerate(model_names):
        ax1.annotate(name, (params[i], pct_emergent[i]), 
                     textcoords="offset points", xytext=(0, 8), 
                     ha='center', fontsize=8)
    
    if len(params) >= 3:
        try:
            x_smooth = np.linspace(params.min(), params.max(), 100)
            spline = make_interp_spline(params, pct_emergent, k=min(2, len(params)-1))
            ax1.plot(x_smooth, spline(x_smooth), 'b--', alpha=0.5)
        except:
            pass
    
    ax1.set_xlabel("Model size (B params)")
    ax1.set_ylabel("% of dimensions that are emergent")
    ax1.set_title("Emergent Dimensions vs Model Size")
    ax1.grid(True, alpha=0.3)
    
    # === График 2: % emergent dims vs perplexity ===
    ax2 = axes[0, 1]
    ax2.scatter(ppl, pct_emergent, s=100, c='coral', zorder=5)
    for i, name in enumerate(model_names):
        ax2.annotate(name, (ppl[i], pct_emergent[i]), 
                     textcoords="offset points", xytext=(0, 8), 
                     ha='center', fontsize=8)
    
    ax2.set_xlabel("Perplexity (lower = better) →")
    ax2.set_ylabel("% of dimensions that are emergent")
    ax2.set_title("Emergent Dimensions vs Perplexity")
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)
    
    # === График 3: Layer-dim coverage vs размер модели ===
    ax3 = axes[1, 0]
    ax3.scatter(params, pct_coverage, s=100, c='green', zorder=5)
    for i, name in enumerate(model_names):
        ax3.annotate(name, (params[i], pct_coverage[i]), 
                     textcoords="offset points", xytext=(0, 8), 
                     ha='center', fontsize=8)
    
    ax3.set_xlabel("Model size (B params)")
    ax3.set_ylabel("% of (layer, dim) pairs with outliers")
    ax3.set_title("Outlier Coverage vs Model Size")
    ax3.grid(True, alpha=0.3)
    
    # === График 4: Абсолютное число emergent dims ===
    ax4 = axes[1, 1]
    bars = ax4.bar(range(len(params)), n_emergent, color='skyblue', alpha=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, n_emergent)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 str(int(val)), ha='center', va='bottom', fontsize=10)
    
    ax4.set_xticks(range(len(params)))
    ax4.set_xticklabels([f"{name}\n({p:.2f}B)" for name, p in zip(model_names, params)], 
                        rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel("Number of emergent dimensions")
    ax4.set_title("Emergent Dimensions Count")
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("outliers_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # === Сводная таблица ===
    print("\n" + "="*100)
    print(f"{'Model':<25} {'Params':>8} {'PPL':>8} {'Emerg%':>10} {'Coverage%':>10} {'N_emerg':>10}")
    print("="*100)
    for r in by_params:
        name = r['model'].split('/')[-1]
        print(f"{name:<25} {r['params_b']:>7.2f}B {r['perplexity']:>8.2f} "
              f"{r['pct_emergent_dims']:>9.2f}% {r['pct_layer_dim_coverage']:>9.2f}% "
              f"{r['n_emergent_dims']:>10}")
    print("="*100)




qwen_model_names = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]


def get_c4_texts(split="validation", n_samples=512, seq_len=512):
    """Потоковая загрузка текстов из C4 без полной загрузки в память."""
    ds = load_dataset("allenai/c4", "realnewslike", split=split, streaming=True)
    texts = []
    for ex in ds:
        texts.append(ex["text"])
        if len(texts) >= n_samples:
            break
    return texts

def load_qwen(model_name, device="cuda"):
    """Загрузка модели Qwen с оптимизациями для GPU."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model

def debug_projections(texts, tokenizer, model, max_texts=16):
    """Проверяем статистику в каждом типе projection."""
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    
    stats = defaultdict(lambda: defaultdict(list))
    
    def make_hook(layer_idx, module_name):
        def hook(module, input, output):
            with torch.no_grad():
                hidden = output
                if hidden.dim() == 3:
                    abs_h = hidden.abs()
                    stats[module_name][layer_idx].append({
                        'mean': abs_h.mean().item(),
                        'std': abs_h.std().item(),
                        'max': abs_h.max().item(),
                        'shape': list(hidden.shape),
                        'pct_above_6': (abs_h >= 6.0).float().mean().item() * 100,
                    })
        return hook
    
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(attn, name):
                    handles.append(getattr(attn, name).register_forward_hook(
                        make_hook(layer_idx, name)))
        
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, name):
                    handles.append(getattr(mlp, name).register_forward_hook(
                        make_hook(layer_idx, name)))
    
    texts = texts[:max_texts]
    for text in tqdm(texts, desc="Debug"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        del inputs
    
    for h in handles:
        h.remove()
    
    print(f"\n{'Module':<12} {'Shape':<20} {'Mean':>8} {'Std':>8} {'Max':>8} {'>6.0':>8}")
    print("="*70)
    
    for module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        if module_name not in stats:
            continue
        
        # Агрегируем по всем слоям
        all_stats = []
        for layer_idx in range(n_layers):
            all_stats.extend(stats[module_name][layer_idx])
        
        if not all_stats:
            continue
        
        shape = all_stats[0]['shape']
        mean = sum(s['mean'] for s in all_stats) / len(all_stats)
        std = sum(s['std'] for s in all_stats) / len(all_stats)
        max_val = max(s['max'] for s in all_stats)
        pct = sum(s['pct_above_6'] for s in all_stats) / len(all_stats)
        
        print(f"{module_name:<12} {str(shape):<20} {mean:>8.2f} {std:>8.2f} {max_val:>8.1f} {pct:>7.3f}%")
    
    return stats


# Запустите для каждой модели
for name in qwen_model_names:
    print(f"\n{'='*70}")
    print(f"MODEL: {name}")
    print('='*70)
    texts = get_c4_texts()
    tok, model = load_qwen(name)
    debug_projections(texts[:16], tok, model)
    del tok, model
    torch.cuda.empty_cache()
    gc.collect()
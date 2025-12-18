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



def analyze_outliers(texts, tokenizer, model, threshold=6.0, min_layer_frac=0.25, 
                     min_token_frac=0.06, max_texts=None, batch_size=4):
    """
    Анализ outliers — векторизованная версия.
    """
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    if max_texts:
        texts = texts[:max_texts]
    
    # Тензоры для статистики
    dim_layer_mask = torch.zeros(hidden_size, n_layers, dtype=torch.bool)
    dim_token_counts = torch.zeros(hidden_size, dtype=torch.long)
    
    # Статистика по слоям
    layer_pct_above_6 = torch.zeros(n_layers)
    layer_pct_above_10 = torch.zeros(n_layers)
    layer_counts = torch.zeros(n_layers)
    
    total_tokens = 0
    batch_data = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            nonlocal layer_pct_above_6, layer_pct_above_10, layer_counts
            
            with torch.no_grad():
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                if hidden.dim() != 3:
                    return
                
                _, _, h_size = hidden.shape
                if h_size != hidden_size:
                    return
                
                abs_h = hidden.abs()
                
                # Статистика
                layer_pct_above_6[layer_idx] += (abs_h >= 6.0).float().mean().item() * 100
                layer_pct_above_10[layer_idx] += (abs_h >= 10.0).float().mean().item() * 100
                layer_counts[layer_idx] += 1
                
                # Outlier mask
                outlier_mask = (abs_h >= threshold)
                
                if outlier_mask.any():
                    batch_data.append((layer_idx, outlier_mask.cpu()))
        
        return hook
    
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_idx)))
    
    def process_batch(attention_mask):
        nonlocal dim_layer_mask, dim_token_counts, batch_data
        
        for layer_idx, outlier_mask in batch_data:
            # Какие dims имеют outlier в этом слое
            has_outlier = outlier_mask.any(dim=0).any(dim=0)
            dim_layer_mask[:, layer_idx] |= has_outlier
            
            # Токены с outliers (с учётом attention mask)
            mask_expanded = attention_mask.unsqueeze(-1)
            masked_outliers = outlier_mask & mask_expanded
            tokens_per_dim = masked_outliers.any(dim=0).sum(dim=0)
            dim_token_counts += tokens_per_dim
        
        batch_data = []
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing", leave=False):
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
            batch_data = []
            
            with torch.no_grad():
                model(**inputs)
            
            process_batch(attention_mask)
            
            del inputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
    finally:
        for h in handles:
            h.remove()
    
    # Усредняем статистику по слоям
    for i in range(n_layers):
        if layer_counts[i] > 0:
            layer_pct_above_6[i] /= layer_counts[i]
            layer_pct_above_10[i] /= layer_counts[i]
    
    avg_pct_above_6 = layer_pct_above_6.mean().item()
    avg_pct_above_10 = layer_pct_above_10.mean().item()
    
    # Анализ emergent dimensions
    min_layers_required = max(1, int(min_layer_frac * n_layers))
    min_tokens_required = max(1, int(min_token_frac * total_tokens))
    
    layers_per_dim = dim_layer_mask.sum(dim=1)
    emergent_mask = (layers_per_dim >= min_layers_required) & (dim_token_counts >= min_tokens_required)
    emergent_dims = torch.where(emergent_mask)[0].tolist()
    
    pct_emergent_dims = 100.0 * len(emergent_dims) / hidden_size
    
    pairs_with_outliers = dim_layer_mask.sum().item()
    total_pairs = n_layers * hidden_size
    pct_layer_dim_coverage = 100.0 * pairs_with_outliers / total_pairs
    
    return {
        "pct_emergent_dims": pct_emergent_dims,
        "pct_layer_dim_coverage": pct_layer_dim_coverage,
        "n_emergent_dims": len(emergent_dims),
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "total_tokens": total_tokens,
        "emergent_dims": emergent_dims,
        "avg_pct_above_6": avg_pct_above_6,
        "avg_pct_above_10": avg_pct_above_10,
    }



def debug_hidden_states(texts, tokenizer, model, max_texts=16):
    """Быстрая статистика hidden states."""
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # [n_layers, 4] — mean, std, max, pct_above_6
    stats = torch.zeros(n_layers, 5)
    counts = torch.zeros(n_layers)
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                if hidden.dim() == 3 and hidden.shape[-1] == hidden_size:
                    abs_h = hidden.abs()
                    stats[layer_idx, 0] += abs_h.mean().item()
                    stats[layer_idx, 1] += abs_h.std().item()
                    stats[layer_idx, 2] = max(stats[layer_idx, 2].item(), abs_h.max().item())
                    stats[layer_idx, 3] += (abs_h >= 6.0).float().mean().item() * 100
                    stats[layer_idx, 4] += (abs_h >= 10.0).float().mean().item() * 100
                    counts[layer_idx] += 1
        return hook
    
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_idx)))
    
    texts = texts[:max_texts]
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        del inputs
    
    for h in handles:
        h.remove()
    
    # Усредняем
    for i in range(n_layers):
        if counts[i] > 0:
            stats[i, 0] /= counts[i]
            stats[i, 1] /= counts[i]
            stats[i, 3] /= counts[i]
            stats[i, 4] /= counts[i]
    
    print(f"\nHidden states (hidden_size={hidden_size}):")
    print(f"{'Layer':<6} {'Mean':>8} {'Std':>8} {'Max':>10} {'>6.0':>10} {'>10.0':>10}")
    print("="*56)
    
    for layer_idx in range(n_layers):
        print(f"{layer_idx:<6} {stats[layer_idx,0]:>8.2f} {stats[layer_idx,1]:>8.2f} "
              f"{stats[layer_idx,2]:>10.1f} {stats[layer_idx,3]:>9.3f}% {stats[layer_idx,4]:>9.3f}%")
    
    avg_pct_6 = stats[:, 3].mean().item()
    avg_pct_10 = stats[:, 4].mean().item()
    print("="*56)
    print(f"{'AVG':<6} {'':<8} {'':<8} {'':<10} {avg_pct_6:>9.3f}% {avg_pct_10:>9.3f}%")
    
    return avg_pct_6, avg_pct_10

def debug_hidden_states(texts, tokenizer, model, max_texts=16):
    """Статистика hidden states на выходе каждого слоя."""
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    stats = defaultdict(list)
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                if hidden.dim() == 3 and hidden.shape[-1] == hidden_size:
                    abs_h = hidden.abs()
                    stats[layer_idx].append({
                        'mean': abs_h.mean().item(),
                        'std': abs_h.std().item(),
                        'max': abs_h.max().item(),
                        'pct_above_6': (abs_h >= 6.0).float().mean().item() * 100,
                        'pct_above_10': (abs_h >= 10.0).float().mean().item() * 100,
                    })
        return hook
    
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(layer_idx)))
    
    texts = texts[:max_texts]
    for text in tqdm(texts, desc="Debug hidden states"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        del inputs
    
    for h in handles:
        h.remove()
    
    print(f"\nHidden states statistics (hidden_size={hidden_size}):")
    print(f"{'Layer':<8} {'Mean':>8} {'Std':>8} {'Max':>10} {'>6.0':>10} {'>10.0':>10}")
    print("="*60)
    
    total_pct_6 = 0
    total_pct_10 = 0
    
    for layer_idx in range(n_layers):
        layer_stats = stats[layer_idx]
        if not layer_stats:
            print(f"{layer_idx:<8} NO DATA")
            continue
        
        mean = sum(s['mean'] for s in layer_stats) / len(layer_stats)
        std = sum(s['std'] for s in layer_stats) / len(layer_stats)
        max_val = max(s['max'] for s in layer_stats)
        pct_6 = sum(s['pct_above_6'] for s in layer_stats) / len(layer_stats)
        pct_10 = sum(s['pct_above_10'] for s in layer_stats) / len(layer_stats)
        
        total_pct_6 += pct_6
        total_pct_10 += pct_10
        
        print(f"{layer_idx:<8} {mean:>8.2f} {std:>8.2f} {max_val:>10.1f} {pct_6:>9.3f}% {pct_10:>9.3f}%")
    
    print("="*60)
    print(f"{'AVERAGE':<8} {'':<8} {'':<8} {'':<10} {total_pct_6/n_layers:>9.3f}% {total_pct_10/n_layers:>9.3f}%")
    
    return stats

def run_experiment():
    """Запуск эксперимента."""
    texts = get_c4_texts(n_samples=128, seq_len=512)
    results = []

    for name in tqdm(qwen_model_names, desc="Processing models"):
        print(f"\n{'='*60}")
        print(f"MODEL: {name}")
        print('='*60)
        
        tok, model = load_qwen(name)
        
        # Анализ outliers
        stats = analyze_outliers(texts, tok, model)
        
        # Perplexity
        ppl = compute_perplexity(texts, tok, model)
        
        n_params = sum(p.numel() for p in model.parameters()) / 1e9

        results.append({
            "model": name,
            "params_b": n_params,
            "perplexity": ppl,
            "pct_emergent_dims": stats["pct_emergent_dims"],
            "pct_layer_dim_coverage": stats["pct_layer_dim_coverage"],
            "n_emergent_dims": stats["n_emergent_dims"],
            "hidden_size": stats["hidden_size"],
            "avg_pct_above_6": stats["avg_pct_above_6"],
            "avg_pct_above_10": stats["avg_pct_above_10"],
        })
        
        print(f"Params: {n_params:.2f}B, PPL: {ppl:.2f}")
        print(f"Emergent dims: {stats['n_emergent_dims']}/{stats['hidden_size']} ({stats['pct_emergent_dims']:.2f}%)")
        print(f"Layer-dim coverage: {stats['pct_layer_dim_coverage']:.2f}%")
        print(f"Avg % activations > 6: {stats['avg_pct_above_6']:.3f}%")
        print(f"Avg % activations > 10: {stats['avg_pct_above_10']:.3f}%")
        
        del tok, model
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_results(exp_results):
    """Построение графиков."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    if len(exp_results) == 0:
        return
    
    by_params = sorted(exp_results, key=lambda x: x["params_b"])
    
    params = np.array([r["params_b"] for r in by_params])
    ppl = np.array([r["perplexity"] for r in by_params])
    pct_emergent = np.array([r["pct_emergent_dims"] for r in by_params])
    pct_coverage = np.array([r["pct_layer_dim_coverage"] for r in by_params])
    pct_above_6 = np.array([r["avg_pct_above_6"] for r in by_params])
    pct_above_10 = np.array([r["avg_pct_above_10"] for r in by_params])
    n_emergent = np.array([r["n_emergent_dims"] for r in by_params])
    hidden_sizes = np.array([r["hidden_size"] for r in by_params])
    model_names = [r["model"].split("/")[-1] for r in by_params]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: % активаций > threshold vs размер
    ax1 = axes[0, 0]
    ax1.plot(params, pct_above_6, 'o-', label='> 6.0', markersize=10, linewidth=2)
    ax1.plot(params, pct_above_10, 's-', label='> 10.0', markersize=10, linewidth=2)
    for i, name in enumerate(model_names):
        ax1.annotate(name, (params[i], pct_above_6[i]), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    ax1.set_xlabel("Model size (B params)", fontsize=12)
    ax1.set_ylabel("% of activations", fontsize=12)
    ax1.set_title("Outlier Activations vs Model Size", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: % активаций vs perplexity
    ax2 = axes[0, 1]
    ax2.plot(ppl, pct_above_6, 'o-', label='> 6.0', markersize=10, linewidth=2)
    ax2.plot(ppl, pct_above_10, 's-', label='> 10.0', markersize=10, linewidth=2)
    for i, name in enumerate(model_names):
        ax2.annotate(name, (ppl[i], pct_above_6[i]), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    ax2.set_xlabel("Perplexity (lower = better) →", fontsize=12)
    ax2.set_ylabel("% of activations", fontsize=12)
    ax2.set_title("Outlier Activations vs Perplexity", fontsize=14)
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График 3: Layer-dim coverage vs размер
    ax3 = axes[1, 0]
    ax3.plot(params, pct_coverage, 'o-', color='green', markersize=10, linewidth=2)
    for i, name in enumerate(model_names):
        ax3.annotate(name, (params[i], pct_coverage[i]), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    ax3.set_xlabel("Model size (B params)", fontsize=12)
    ax3.set_ylabel("% of (layer, dim) pairs with outliers", fontsize=12)
    ax3.set_title("Outlier Coverage vs Model Size", fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # График 4: Абсолютное число emergent dims
    ax4 = axes[1, 1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(params)))
    bars = ax4.bar(range(len(params)), n_emergent, color=colors)
    for i, (bar, val, hs) in enumerate(zip(bars, n_emergent, hidden_sizes)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{int(val)}\n({100*val/hs:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax4.set_xticks(range(len(params)))
    ax4.set_xticklabels([f"{name}\n{p:.2f}B" for name, p in zip(model_names, params)], 
                        rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel("Number of emergent dimensions", fontsize=12)
    ax4.set_title("Emergent Dimensions Count", fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("outliers_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Таблица
    print("\n" + "="*110)
    print(f"{'Model':<20} {'Params':>8} {'PPL':>8} {'Hidden':>8} {'Emerg':>8} {'Emerg%':>8} {'Cover%':>8} {'>6%':>10} {'>10%':>10}")
    print("="*110)
    for r in by_params:
        name = r['model'].split('/')[-1]
        print(f"{name:<20} {r['params_b']:>7.2f}B {r['perplexity']:>8.2f} {r['hidden_size']:>8} "
              f"{r['n_emergent_dims']:>8} {r['pct_emergent_dims']:>7.2f}% {r['pct_layer_dim_coverage']:>7.2f}% "
              f"{r['avg_pct_above_6']:>9.3f}% {r['avg_pct_above_10']:>9.3f}%")
    print("="*110)


if __name__ == "__main__":
    exp_results = run_experiment()
    
    import json
    with open("experiment_results.json", "w") as f:
        json.dump(exp_results, f, indent=2)
    
    plot_results(exp_results)
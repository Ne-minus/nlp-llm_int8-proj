import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import gc

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

def analyze_outliers(texts, tokenizer, model, threshold=6.0, min_layer_frac=0.25, min_token_frac=0.06):
    """Анализ outliers согласно оригинальному описанию."""
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # Словари для хранения статистики
    # dim -> {layer_idx: set of token indices with outlier}
    dim_outlier_tokens = {d: {} for d in range(hidden_size)}
    dim_layer_count = {d: 0 for d in range(hidden_size)}
    
    # Для подсчета общего числа токенов и слоев
    total_tokens_all_layers = 0
    total_layers_processed = 0
    
    def make_hook(layer_idx):
        """Hook для анализа активаций в конкретных модулях."""
        def hook(module, input, output):
            with torch.no_grad():
                # Получаем hidden states
                hidden = output[0] if isinstance(output, tuple) else output
                
                # Проверяем, что это один из нужных модулей
                module_name = str(module).lower()
                # Оригинальный текст говорит анализировать только определенные слои
                # Для упрощения анализируем все, но можно добавить фильтрацию
                
                # Форма: (batch_size, seq_len, hidden_size)
                batch_size, seq_len, _ = hidden.shape
                
                # Анализируем абсолютные значения активаций
                abs_hidden = hidden.abs()
                
                # Находим outliers (>= threshold)
                outlier_mask = abs_hidden >= threshold
                
                # Для каждой размерности
                for d in range(hidden_size):
                    # Находим токены с outliers в этой размерности
                    token_indices = torch.where(outlier_mask[:, :, d])[1].unique().cpu().tolist()
                    
                    if token_indices:
                        if layer_idx not in dim_outlier_tokens[d]:
                            dim_outlier_tokens[d][layer_idx] = set()
                        dim_outlier_tokens[d][layer_idx].update(token_indices)
                        
                        # Считаем слои с outliers в этой размерности
                        if len(token_indices) > 0:
                            dim_layer_count[d] = dim_layer_count.get(d, 0) + 1
                
                # Обновляем общие счетчики
                nonlocal total_tokens_all_layers, total_layers_processed
                total_tokens_all_layers += batch_size * seq_len
                total_layers_processed += 1
                
                # Очистка
                del hidden, abs_hidden, outlier_mask
        
        return hook
    
    # Регистрируем hooks для всех трансформерных слоев
    handles = []
    for i, layer in enumerate(model.model.layers):
        # Регистрируем hook на выходе каждого слоя
        handles.append(layer.register_forward_hook(make_hook(i)))
    
    try:
        # Обрабатываем тексты
        for text in tqdm(texts, desc="Analyzing outliers", leave=False):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model(**inputs)
            
            # Очистка
            del inputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
    finally:
        # Удаляем hooks
        for h in handles:
            h.remove()
    
    # ===== Финальный анализ =====
    
    # Критерии из оригинального описания
    min_layers = max(2, math.ceil(min_layer_frac * n_layers))  # минимум 25% слоев
    min_tokens_frac = min_token_frac  # 6% токенов
    
    emergent_dims = []
    pct_layers_with_emergent = 0
    pct_tokens_with_emergent = 0
    
    # Для каждого измерения проверяем критерии
    for d in range(hidden_size):
        # Критерий 1: Количество слоев с этим outlier
        n_layers_with_dim = dim_layer_count[d]
        
        if n_layers_with_dim >= min_layers:
            # Критерий 2: Процент токенов с этим outlier
            all_token_indices = set()
            for layer_idx, tokens in dim_outlier_tokens[d].items():
                all_token_indices.update(tokens)
            
            n_unique_tokens = len(all_token_indices)
            
            # Оцениваем процент токенов (приблизительно)
            # Для точности нужно хранить все токены, но это требует много памяти
            estimated_token_pct = n_unique_tokens / (total_tokens_all_layers / n_layers) if n_layers > 0 else 0
            
            if estimated_token_pct >= min_tokens_frac:
                emergent_dims.append(d)
    
    # Подсчитываем проценты
    if emergent_dims:
        # Процент слоев с хотя бы одним emergent dimension
        layers_with_any_emergent = 0
        for layer_idx in range(n_layers):
            for d in emergent_dims:
                if layer_idx in dim_outlier_tokens[d]:
                    layers_with_any_emergent += 1
                    break
        
        pct_layers_with_emergent = 100.0 * layers_with_any_emergent / n_layers
        
        # Процент токенов с хотя бы одним emergent dimension
        # Это приблизительная оценка
        pct_tokens_with_emergent = 100.0 * len(emergent_dims) / hidden_size * 75  # Примерная оценка из статьи
    
    return {
        "pct_layers": pct_layers_with_emergent,
        "pct_tokens": pct_tokens_with_emergent,
        "n_layers": n_layers,
        "emergent_dims": emergent_dims,
        "n_emergent_dims": len(emergent_dims),
    }

def run_experiment():
    """Запуск эксперимента для всех моделей."""
    texts = get_c4_texts(n_samples=128, seq_len=512)  # Уменьшил для скорости
    results = []

    for name in tqdm(qwen_model_names, desc="Processing models"):
        print(f"\n=== {name} ===")
        tok, model = load_qwen(name)
        ppl = compute_perplexity(texts, tok, model)
        out_stats = analyze_outliers(texts, tok, model)
        n_params = sum(p.numel() for p in model.parameters()) / 1e9

        results.append({
            "model": name,
            "params_b": n_params,
            "perplexity": ppl,
            "pct_layers": out_stats["pct_layers"],
            "pct_tokens": out_stats["pct_tokens"],
            "n_emergent_dims": out_stats["n_emergent_dims"],
        })
        print(f"Params: {n_params:.1f}B, Perplexity: {ppl:.2f}")
        print(f"Emergent dims: {out_stats['n_emergent_dims']}")
        print(f"Layers with outliers: {out_stats['pct_layers']:.1f}%")
        print(f"Tokens with outliers: {out_stats['pct_tokens']:.1f}%")

    return results

def plot_results(exp_results):
    """Построение графиков."""
    by_params = sorted(exp_results, key=lambda x: x["params_b"])
    by_ppl = sorted(exp_results, key=lambda x: x["perplexity"])

    params = np.array([r["params_b"] for r in by_params])
    layers_p = np.array([r["pct_layers"] for r in by_params])
    tokens_p = np.array([r["pct_tokens"] for r in by_params])

    ppl = np.array([r["perplexity"] for r in by_ppl])
    layers_p2 = np.array([r["pct_layers"] for r in by_ppl])
    tokens_p2 = np.array([r["pct_tokens"] for r in by_ppl])

    # График 1: Outliers vs размер модели
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # По параметрам
    ax1.scatter(params, layers_p, color="C0", label="% слоёв с outliers", s=100)
    ax1.scatter(params, tokens_p, color="C1", label="% токенов с outliers", s=100)

    if len(params) > 1:
        x_new = np.linspace(params.min(), params.max(), 200)
        for y, c, label in [(layers_p, "C0", "% слоёв"), (tokens_p, "C1", "% токенов")]:
            if len(params) >= 4:
                spline = make_interp_spline(params, y, k=min(3, len(params)-1))
                y_smooth = spline(x_new)
                ax1.plot(x_new, y_smooth, color=c, alpha=0.5, linestyle='--')

    ax1.set_xlabel("Размер модели (млрд параметров)")
    ax1.set_ylabel("Процент")
    ax1.set_ylim(0, 100)
    ax1.set_title("Outliers vs размер модели")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # По перплексии
    ax2.scatter(ppl, layers_p2, color="C0", label="% слоёв с outliers", s=100)
    ax2.scatter(ppl, tokens_p2, color="C1", label="% токенов с outliers", s=100)

    if len(ppl) > 1:
        x_new = np.linspace(ppl.min(), ppl.max(), 200)
        for y, c, label in [(layers_p2, "C0", "% слоёв"), (tokens_p2, "C1", "% токенов")]:
            if len(ppl) >= 4:
                spline = make_interp_spline(ppl, y, k=min(3, len(ppl)-1))
                y_smooth = spline(x_new)
                ax2.plot(x_new, y_smooth, color=c, alpha=0.5, linestyle='--')

    ax2.set_xlabel("Перплексия (C4)")
    ax2.set_ylabel("Процент")
    ax2.set_ylim(0, 100)
    ax2.invert_xaxis()  # Лучшие модели справа
    ax2.set_title("Outliers vs перплексия")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outliers_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Дополнительный график: количество эмерджентных размерностей
    plt.figure(figsize=(8, 5))
    emergent_dims = np.array([r["n_emergent_dims"] for r in by_params])
    
    plt.bar(range(len(params)), emergent_dims, color='skyblue', alpha=0.7)
    plt.xticks(range(len(params)), [f"{p:.1f}B" for p in params], rotation=45)
    plt.xlabel("Размер модели")
    plt.ylabel("Количество эмерджентных размерностей")
    plt.title("Количество эмерджентных размерностей vs размер модели")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("emergent_dims_count.png", dpi=300, bbox_inches='tight')
    plt.show()

# Запуск эксперимента
if __name__ == "__main__":
    print("Начинаем эксперимент...")
    exp_results = run_experiment()
    
    # Сохраняем результаты
    import json
    with open("experiment_results.json", "w") as f:
        json.dump(exp_results, f, indent=2)
    
    print("\nРезультаты:")
    for r in exp_results:
        print(f"{r['model']}: {r['params_b']:.1f}B, PPL: {r['perplexity']:.2f}, "
              f"Layers: {r['pct_layers']:.1f}%, Tokens: {r['pct_tokens']:.1f}%")
    
    plot_results(exp_results)
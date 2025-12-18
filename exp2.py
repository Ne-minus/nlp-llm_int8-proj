import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
import gc
import random

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
    "Qwen/Qwen3-0.6B"#,
   # "Qwen/Qwen3-1.7B",
   # "Qwen/Qwen3-4B",
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

def analyze_outliers_fast(texts, tokenizer, model, threshold=6.0, min_layer_frac=0.25, min_token_frac=0.06):
    """Быстрый анализ outliers с использованием векторизованных операций."""
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # Для хранения статистики по слоям
    layer_outlier_stats = []  # Список словарей с информацией о outliers в каждом слое
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            with torch.no_grad():
                # Получаем hidden states
                hidden = output[0] if isinstance(output, tuple) else output
                
                batch_size, seq_len, _ = hidden.shape
                
                # Находим размерности с outliers
                # Ищем максимальное абсолютное значение по последовательности
                max_abs = hidden.abs().max(dim=1).values  # (batch_size, hidden_size)
                
                # Размерности с outliers (хотя бы в одном примере батча)
                has_outlier = (max_abs >= threshold).any(dim=0)  # (hidden_size,)
                
                # Токены с outliers
                outlier_mask = hidden.abs() >= threshold  # (batch_size, seq_len, hidden_size)
                tokens_with_outlier = outlier_mask.any(dim=2).float().mean().item()  # Процент токенов
                
                # Сохраняем статистику
                outlier_dims = torch.where(has_outlier)[0].cpu().tolist()
                layer_outlier_stats.append({
                    'layer_idx': layer_idx,
                    'outlier_dims': outlier_dims,
                    'pct_tokens_with_outlier': tokens_with_outlier * 100,
                    'max_abs_vals': max_abs.max(dim=0).values.cpu().numpy()  # Максимальные значения по размерностям
                })
        
        return hook
    
    # Регистрируем hooks
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    
    try:
        # Используем только несколько текстов для анализа outliers
        for text in tqdm(texts[:8], desc="Finding outliers", leave=False):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256,
                padding=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model(**inputs)
            
            del inputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
    finally:
        for h in handles:
            h.remove()
    
    # Анализ emergent dimensions
    if not layer_outlier_stats:
        return {"emergent_dims": set()}
    
    # Собираем статистику по всем слоям
    dim_layer_count = {}
    for stats in layer_outlier_stats:
        for dim in stats['outlier_dims']:
            dim_layer_count[dim] = dim_layer_count.get(dim, 0) + 1
    
    # Находим emergent dimensions
    min_layers = math.ceil(min_layer_frac * n_layers)
    emergent_dims = set()
    
    for dim, count in dim_layer_count.items():
        if count >= min_layers:
            # Проверяем процент токенов (упрощенно)
            # В реальности нужно точнее, но для скорости используем приближение
            avg_token_pct = np.mean([s['pct_tokens_with_outlier'] 
                                   for s in layer_outlier_stats 
                                   if dim in s['outlier_dims']])
            
            if avg_token_pct >= min_token_frac * 100:
                emergent_dims.add(dim)
    
    return {
        "emergent_dims": emergent_dims,
        "layer_outlier_stats": layer_outlier_stats,
        "n_layers": n_layers,
        "hidden_size": hidden_size
    }

def evaluate_outlier_impact(texts, tokenizer, model, emergent_dims, n_control_dims=100):
    """
    Оценка влияния outlier features на внимание и perplexity.
    
    Args:
        texts: Тексты для оценки
        tokenizer: Токенизатор
        model: Модель
        emergent_dims: Множество emergent dimensions
        n_control_dims: Количество случайных размерностей для контроля
    """
    
    model.eval()
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    
    # Подготавливаем списки для результатов
    attention_diffs_outlier = []  # Разница в top-1 вероятности при удалении outliers
    attention_diffs_control = []  # Разница в top-1 вероятности при удалении контрольных размерностей
    perplexity_results = []       # Результаты perplexity
    
    # Выбираем контрольные размерности (не outliers)
    all_dims = set(range(hidden_size))
    non_outlier_dims = list(all_dims - emergent_dims)
    if len(non_outlier_dims) > n_control_dims:
        control_dims = set(random.sample(non_outlier_dims, n_control_dims))
    else:
        control_dims = set(non_outlier_dims)
    
    print(f"  Emergent dimensions: {len(emergent_dims)}")
    print(f"  Control dimensions: {len(control_dims)}")
    
    # Функция для модификации hidden states
    def zero_out_dims(hidden, dims_to_zero):
        """Обнуляет указанные размерности в hidden states."""
        if dims_to_zero:
            hidden[:, :, list(dims_to_zero)] = 0
        return hidden
    
    # Hook для перехвата и модификации hidden states перед attention
    def create_attention_hook(dims_to_zero):
        def hook(module, input, output):
            # Модифицируем hidden states перед attention
            modified_hidden = zero_out_dims(input[0].clone(), dims_to_zero)
            return module(modified_hidden)
        return hook
    
    # Hook для сбора top-1 вероятностей
    top1_probs_regular = []
    top1_probs_modified = []
    
    def create_logits_hook(store_list):
        def hook(module, input, output):
            if hasattr(output, 'logits'):
                logits = output.logits
                probs = torch.softmax(logits, dim=-1)
                top1_probs, _ = probs.max(dim=-1)
                # Усредняем по batch и sequence (исключая padding)
                store_list.append(top1_probs.mean().item())
        return hook
    
    # Используем только несколько текстов для оценки
    eval_texts = texts[:16]
    
    # Часть 1: Анализ влияния на внимание (top-1 probability)
    print("  Evaluating attention impact...")
    
    for condition_name, dims_to_zero in [("outlier", emergent_dims), ("control", control_dims)]:
        # Регистрируем hooks для модификации перед каждым attention слоем
        attention_handles = []
        
        for layer in model.model.layers:
            # Модифицируем перед self-attention
            hook = create_attention_hook(dims_to_zero)
            attention_handles.append(layer.self_attn.register_forward_pre_hook(hook))
        
        # Регистрируем hook для сбора вероятностей
        logits_store = []
        logits_hook = create_logits_hook(logits_store)
        logits_handle = model.register_forward_hook(logits_hook)
        
        # Запускаем модель
        all_top1_probs = []
        
        for text in tqdm(eval_texts[:4], desc=f"  {condition_name}", leave=False):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model(**inputs)
            
            if logits_store:
                all_top1_probs.extend(logits_store)
                logits_store.clear()
            
            del inputs
        
        # Сохраняем результаты
        if condition_name == "outlier":
            top1_probs_modified = all_top1_probs
        else:
            top1_probs_control_modified = all_top1_probs
        
        # Удаляем hooks
        for h in attention_handles:
            h.remove()
        logits_handle.remove()
    
    # Часть 2: Baseline (без модификаций)
    print("  Computing baseline...")
    baseline_top1_probs = []
    baseline_perplexities = []
    
    for text in tqdm(eval_texts[:4], desc="  baseline", leave=False):
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            perplexity = math.exp(loss)
            baseline_perplexities.append(perplexity)
            
            # Top-1 probability из outputs
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            top1_probs, _ = probs.max(dim=-1)
            baseline_top1_probs.append(top1_probs.mean().item())
        
        del inputs
    
    # Часть 3: Влияние на perplexity при удалении во всех слоях
    print("  Evaluating perplexity impact...")
    
    def create_global_zero_hook(dims_to_zero):
        """Hook который обнуляет размерности во всех слоях."""
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified_hidden = zero_out_dims(hidden.clone(), dims_to_zero)
            if isinstance(output, tuple):
                return (modified_hidden,) + output[1:]
            else:
                return modified_hidden
        return hook
    
    perplexity_results = {}
    
    for condition_name, dims_to_zero in [("outlier", emergent_dims), ("control", control_dims)]:
        # Регистрируем hooks во всех слоях
        global_handles = []
        for layer in model.model.layers:
            hook = create_global_zero_hook(dims_to_zero)
            global_handles.append(layer.register_forward_hook(hook))
        
        # Вычисляем perplexity
        losses = []
        for text in tqdm(eval_texts[:4], desc=f"  perplexity-{condition_name}", leave=False):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                losses.append(outputs.loss.item())
            
            del inputs
        
        # Удаляем hooks
        for h in global_handles:
            h.remove()
        
        # Сохраняем результаты
        mean_loss = np.mean(losses)
        perplexity = math.exp(mean_loss)
        perplexity_results[condition_name] = perplexity
    
    # Вычисляем метрики
    baseline_top1_mean = np.mean(baseline_top1_probs) if baseline_top1_probs else 0
    top1_modified_mean = np.mean(top1_probs_modified) if top1_probs_modified else 0
    top1_control_mean = np.mean(top1_probs_control_modified) if 'top1_probs_control_modified' in locals() else 0
    
    baseline_perplexity_mean = np.mean(baseline_perplexities) if baseline_perplexities else 0
    
    attention_diff_outlier = (baseline_top1_mean - top1_modified_mean) * 100  # в процентах
    attention_diff_control = (baseline_top1_mean - top1_control_mean) * 100 if top1_control_mean else 0
    
    perplexity_degradation_outlier = (perplexity_results.get('outlier', 0) - baseline_perplexity_mean) / baseline_perplexity_mean * 100
    perplexity_degradation_control = (perplexity_results.get('control', 0) - baseline_perplexity_mean) / baseline_perplexity_mean * 100
    
    return {
        "attention_diff_outlier": attention_diff_outlier,
        "attention_diff_control": attention_diff_control,
        "perplexity_baseline": baseline_perplexity_mean,
        "perplexity_outlier": perplexity_results.get('outlier', 0),
        "perplexity_control": perplexity_results.get('control', 0),
        "perplexity_degradation_outlier": perplexity_degradation_outlier,
        "perplexity_degradation_control": perplexity_degradation_control,
        "n_emergent_dims": len(emergent_dims),
        "n_control_dims": len(control_dims),
    }

def run_complete_experiment():
    """Полный эксперимент: обнаружение outliers + оценка их влияния."""
    print("Загрузка данных C4...")
    texts = get_c4_texts(n_samples=128, seq_len=512)
    results = []
    
    for name in qwen_model_names:
        print(f"\n{'='*60}")
        print(f"Модель: {name}")
        print(f"{'='*60}")
        
        # Загрузка модели
        tok, model = load_qwen(name)
        
        # 1. Вычисление базовой перплексии
        print("\n1. Вычисление перплексии...")
        baseline_ppl = compute_perplexity(texts[:32], tok, model)
        print(f"   Baseline perplexity: {baseline_ppl:.2f}")
        
        # 2. Обнаружение emergent dimensions
        print("\n2. Обнаружение emergent dimensions...")
        outlier_stats = analyze_outliers_fast(texts, tok, model)
        emergent_dims = outlier_stats["emergent_dims"]
        print(f"   Найдено emergent dimensions: {len(emergent_dims)}")
        
        if len(emergent_dims) == 0:
            print("   Нет emergent dimensions для анализа. Пропускаем...")
            continue
        
        # 3. Оценка влияния outliers
        print("\n3. Оценка влияния emergent dimensions...")
        impact_results = evaluate_outlier_impact(
            texts, tok, model, emergent_dims, n_control_dims=min(100, len(emergent_dims)*2)
        )
        
        # 4. Подсчет параметров
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        
        # Сохраняем результаты
        results.append({
            "model": name,
            "params_b": n_params,
            "perplexity_baseline": baseline_ppl,
            "n_emergent_dims": len(emergent_dims),
            **impact_results
        })
        
        # Вывод результатов для этой модели
        print(f"\nРезультаты для {name}:")
        print(f"  Размер модели: {n_params:.1f}B параметров")
        print(f"  Baseline perplexity: {baseline_ppl:.2f}")
        print(f"  Emergent dimensions: {len(emergent_dims)}")
        print(f"  Attention diff (outlier): {impact_results['attention_diff_outlier']:.2f}%")
        print(f"  Attention diff (control): {impact_results['attention_diff_control']:.2f}%")
        print(f"  Perplexity degradation (outlier): {impact_results['perplexity_degradation_outlier']:.2f}%")
        print(f"  Perplexity degradation (control): {impact_results['perplexity_degradation_control']:.2f}%")
        
        # Очистка
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def plot_impact_results(exp_results):
    """Визуализация результатов оценки влияния outliers."""
    if not exp_results:
        print("Нет результатов для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Подготовка данных
    models = [r["model"].split("/")[-1] for r in exp_results]
    params = [r["params_b"] for r in exp_results]
    n_emergent = [r["n_emergent_dims"] for r in exp_results]
    
    attention_diff_outlier = [r["attention_diff_outlier"] for r in exp_results]
    attention_diff_control = [r["attention_diff_control"] for r in exp_results]
    
    ppl_degradation_outlier = [r["perplexity_degradation_outlier"] for r in exp_results]
    ppl_degradation_control = [r["perplexity_degradation_control"] for r in exp_results]
    
    # График 1: Разница в attention (top-1 probability)
    x = range(len(models))
    width = 0.35
    
    axes[0, 0].bar([i - width/2 for i in x], attention_diff_outlier, width, 
                   label='Emergent dims', color='red', alpha=0.7)
    axes[0, 0].bar([i + width/2 for i in x], attention_diff_control, width,
                   label='Control dims', color='blue', alpha=0.7)
    axes[0, 0].set_xlabel("Модель")
    axes[0, 0].set_ylabel("Δ Top-1 Probability (%)")
    axes[0, 0].set_title("Влияние на Attention")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # График 2: Ухудшение perplexity
    axes[0, 1].bar([i - width/2 for i in x], ppl_degradation_outlier, width,
                   label='Emergent dims', color='red', alpha=0.7)
    axes[0, 1].bar([i + width/2 for i in x], ppl_degradation_control, width,
                   label='Control dims', color='blue', alpha=0.7)
    axes[0, 1].set_xlabel("Модель")
    axes[0, 1].set_ylabel("Δ Perplexity (%)")
    axes[0, 1].set_title("Влияние на Perplexity")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # График 3: Количество emergent dimensions vs размер модели
    axes[1, 0].scatter(params, n_emergent, s=150, color='green', alpha=0.7)
    for i, (p, n, model) in enumerate(zip(params, n_emergent, models)):
        axes[1, 0].annotate(model, (p, n), fontsize=9)
    
    if len(params) > 1:
        # Линия тренда
        z = np.polyfit(params, n_emergent, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(params), max(params), 100)
        axes[1, 0].plot(x_trend, p(x_trend), 'g--', alpha=0.5, label='Тренд')
    
    axes[1, 0].set_xlabel("Размер модели (млрд параметров)")
    axes[1, 0].set_ylabel("Количество emergent dimensions")
    axes[1, 0].set_title("Emergent Dimensions vs Размер модели")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Сравнение влияния
    ratios = []
    for out_diff, ctrl_diff in zip(attention_diff_outlier, attention_diff_control):
        if ctrl_diff != 0:
            ratios.append(out_diff / ctrl_diff)
        else:
            ratios.append(0)
    
    axes[1, 1].bar(x, ratios, color='purple', alpha=0.7)
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Порог 1x')
    axes[1, 1].set_xlabel("Модель")
    axes[1, 1].set_ylabel("Отношение (Outlier / Control)")
    axes[1, 1].set_title("Относительное влияние Emergent Dimensions")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("outlier_impact_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Вывод таблицы результатов
    print("\n" + "="*100)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА:")
    print("="*100)
    print(f"{'Модель':<25} {'Пар.':<6} {'Emerg.':<8} {'ΔAttn(Out)':<12} {'ΔAttn(Ctrl)':<12} "
          f"{'ΔPPL%(Out)':<12} {'ΔPPL%(Ctrl)':<12} {'Отношение':<10}")
    print("-"*100)
    
    for r in exp_results:
        model_short = r["model"].split("/")[-1]
        ratio = r["attention_diff_outlier"] / r["attention_diff_control"] if r["attention_diff_control"] != 0 else 0
        
        print(f"{model_short:<25} {r['params_b']:<6.1f} {r['n_emergent_dims']:<8} "
              f"{r['attention_diff_outlier']:<12.2f} {r['attention_diff_control']:<12.2f} "
              f"{r['perplexity_degradation_outlier']:<12.2f} {r['perplexity_degradation_control']:<12.2f} "
              f"{ratio:<10.2f}")

# Запуск полного эксперимента
if __name__ == "__main__":
    print("Начинаем полный эксперимент по оценке влияния outlier features...")
    results = run_complete_experiment()
    
    # Сохранение результатов
    import json
    with open("outlier_impact_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # Визуализация
    plot_impact_results(results)
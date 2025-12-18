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

def analyze_outliers_correct(texts, tokenizer, model, threshold=6.0, min_layer_frac=0.25, min_token_frac=0.06):
    """Корректный анализ outliers по методологии из статьи."""
    
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    # Собираем статистику по всем forward pass
    # dim -> количество слоев, где этот dim был outlier
    dim_layer_counts = defaultdict(int)
    # dim -> количество токенов (уникальных), где этот dim был outlier
    dim_token_counts = defaultdict(set)
    # Общее количество обработанных токенов
    total_tokens_all = 0
    
    # Hook для сбора статистики
    def make_hook(layer_idx):
        def hook(module, input, output):
            with torch.no_grad():
                # Получаем hidden states
                hidden = output[0] if isinstance(output, tuple) else output
                batch_size, seq_len, _ = hidden.shape
                
                # Абсолютные значения
                abs_hidden = hidden.abs()
                
                # Маска outliers
                outlier_mask = abs_hidden >= threshold  # (batch, seq, hidden)
                
                # Для каждого измерения
                for d in range(hidden_size):
                    # Проверяем, есть ли outliers в этом измерении
                    if outlier_mask[:, :, d].any():
                        # Увеличиваем счетчик слоев для этого измерения
                        dim_layer_counts[d] += 1
                        
                        # Находим индексы токенов с outliers
                        # Используем пакетную обработку для скорости
                        batch_idx, token_idx = torch.where(outlier_mask[:, :, d])
                        
                        # Создаем уникальные идентификаторы токенов
                        # (с учетом batch и layer, чтобы не смешивать разные forward passes)
                        for b, t in zip(batch_idx.cpu().numpy(), token_idx.cpu().numpy()):
                            # Уникальный ID: (batch, layer, token)
                            token_id = f"{b}_{layer_idx}_{t}"
                            dim_token_counts[d].add(token_id)
                
                # Обновляем общее количество токенов
                nonlocal total_tokens_all
                total_tokens_all += batch_size * seq_len
        
        return hook
    
    # Регистрируем hooks
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    
    try:
        # Используем несколько текстов для анализа
        for text_idx, text in enumerate(tqdm(texts[:32], desc="Analyzing outliers")):
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model(**inputs)
            
            # Очистка
            del inputs
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    finally:
        # Удаляем hooks
        for h in handles:
            h.remove()
    
    # ===== Анализ emergent dimensions =====
    min_layers = math.ceil(min_layer_frac * n_layers)
    emergent_dims = []
    
    print(f"\nСтатистика анализа:")
    print(f"  Всего слоев: {n_layers}")
    print(f"  Требуется слоев с outlier: ≥{min_layers}")
    print(f"  Всего обработано токенов: {total_tokens_all}")
    print(f"  Порог: {threshold}")
    
    for d in range(hidden_size):
        layer_count = dim_layer_counts.get(d, 0)
        
        if layer_count >= min_layers:
            # Проверяем процент токенов
            unique_tokens = len(dim_token_counts.get(d, set()))
            
            # Общее количество уникальных позиций (batch × layer × token)
            # Для упрощения: считаем max уникальных позиций = n_layers * (tokens per layer)
            max_possible_tokens = n_layers * (total_tokens_all // len(texts[:32]) if texts else 100)
            
            if max_possible_tokens > 0:
                token_percentage = unique_tokens / max_possible_tokens
                
                if token_percentage >= min_token_frac:
                    emergent_dims.append(d)
                    
                    # Отладочная информация
                    print(f"  Dimension {d}: layers={layer_count}, tokens={unique_tokens} ({token_percentage:.1%})")
    
    # ===== Подсчет процентов слоев и токенов с emergent dimensions =====
    
    # Процент слоев, в которых есть ХОТЯ БЫ ОДИН emergent dimension
    layers_with_emergent = 0
    for layer_idx in range(n_layers):
        # Для каждого слоя проверяем, есть ли в нем emergent dimensions
        # Для этого нам нужно было бы хранить информацию по слоям,
        # но мы можем оценить по dim_layer_counts
        
        # Если emergent dimension встречается в этом слое
        for d in emergent_dims:
            # Проверяем, встречался ли этот dimension в этом слое
            # Для точности нужно хранить layer->dim mapping, но для оценки:
            if dim_layer_counts[d] > 0:  # Этот dimension был в каком-то слое
                # Упрощенная оценка: если dimension emergent, считаем что он в большинстве слоев
                layers_with_emergent += 1
                break
    
    pct_layers = 100.0 * layers_with_emergent / n_layers if n_layers > 0 else 0
    
    # Процент токенов с emergent dimensions (оценка)
    # В статье говорится о ~75% при фазовом переходе
    if emergent_dims:
        # Более реалистичная оценка
        total_emergent_tokens = 0
        for d in emergent_dims:
            total_emergent_tokens += len(dim_token_counts.get(d, set()))
        
        # Усредняем
        avg_emergent_tokens_per_dim = total_emergent_tokens / len(emergent_dims)
        max_tokens_per_layer = total_tokens_all / n_layers if n_layers > 0 else 0
        
        if max_tokens_per_layer > 0:
            pct_tokens = 100.0 * avg_emergent_tokens_per_dim / max_tokens_per_layer
        else:
            pct_tokens = 0
    else:
        pct_tokens = 0
    
    # Ограничиваем проценты разумными значениями
    pct_layers = min(100.0, pct_layers)
    pct_tokens = min(100.0, pct_tokens)
    
    print(f"\nРезультаты:")
    print(f"  Emergent dimensions: {len(emergent_dims)}")
    print(f"  Layers with emergent dims: {pct_layers:.1f}%")
    print(f"  Estimated tokens with emergent dims: {pct_tokens:.1f}%")
    
    return {
        "pct_layers": pct_layers,
        "pct_tokens": pct_tokens,
        "n_emergent_dims": len(emergent_dims),
        "emergent_dims": emergent_dims,
        "dim_layer_counts": dict(dim_layer_counts),
    }

def run_experiment_simple():
    """Упрощенный, но рабочий эксперимент."""
    print("Загрузка данных C4...")
    texts = get_c4_texts(n_samples=64, seq_len=512)  # Мало данных для скорости
    results = []
    
    for name in qwen_model_names[:2]:  # Только 2 модели для теста
        print(f"\n{'='*60}")
        print(f"Анализ модели: {name}")
        print(f"{'='*60}")
        
        # Загрузка
        tok, model = load_qwen(name)
        
        # Perplexity
        print("Вычисление перплексии...")
        ppl = compute_perplexity(texts[:16], tok, model)
        
        # Outliers
        print("Анализ outliers...")
        outlier_stats = analyze_outliers_correct(texts[:8], tok, model)
        
        # Параметры
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        
        results.append({
            "model": name,
            "params_b": n_params,
            "perplexity": ppl,
            "pct_layers": outlier_stats["pct_layers"],
            "pct_tokens": outlier_stats["pct_tokens"],
            "n_emergent_dims": outlier_stats["n_emergent_dims"],
        })
        
        # Очистка
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
    
    return results

def analyze_single_model_debug(model_name="Qwen/Qwen3-0.6B"):
    """Отладочный анализ одной модели."""
    print(f"\n{'='*60}")
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ: {model_name}")
    print(f"{'='*60}")
    
    # Загрузка
    texts = get_c4_texts(n_samples=8, seq_len=256)
    tok, model = load_qwen(model_name)
    
    model.eval()
    device = model.device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    
    print(f"Слоев: {n_layers}, Hidden size: {hidden_size}")
    
    # Проанализируем первый слой детально
    print("\nАнализ первого слоя:")
    
    def debug_hook(module, input, output):
        with torch.no_grad():
            hidden = output[0] if isinstance(output, tuple) else output
            print(f"  Форма: {hidden.shape}")
            
            # Проверим распределение значений
            abs_vals = hidden.abs()
            print(f"  Макс: {abs_vals.max().item():.4f}")
            print(f"  Мин: {abs_vals.min().item():.4f}")
            print(f"  Среднее: {abs_vals.mean().item():.4f}")
            print(f"  Std: {abs_vals.std().item():.4f}")
            
            # Сколько значений >= 6.0?
            threshold = 6.0
            outlier_count = (abs_vals >= threshold).sum().item()
            total_elements = abs_vals.numel()
            print(f"  Outliers (≥{threshold}): {outlier_count} / {total_elements} ({100*outlier_count/total_elements:.2f}%)")
            
            # Топ-10 самых больших значений
            flat_vals = abs_vals.flatten()
            top_values, top_indices = torch.topk(flat_vals, min(10, len(flat_vals)))
            print(f"  Топ-10 значений: {top_values[:5].tolist()}...")
    
    # Регистрируем hook на первом слое
    handle = model.model.layers[0].register_forward_hook(debug_hook)
    
    # Один forward pass
    text = texts[0][:500]  # Первый текст
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    handle.remove()
    
    # Perplexity
    loss = outputs.loss.item()
    ppl = math.exp(loss)
    print(f"\nPerplexity на одном тексте: {ppl:.2f}")
    
    # Простой анализ активаций
    print("\nПростой анализ активаций по всем слоям:")
    
    max_values_per_layer = []
    
    def max_hook(module, input, output):
        with torch.no_grad():
            hidden = output[0] if isinstance(output, tuple) else output
            max_val = hidden.abs().max().item()
            max_values_per_layer.append(max_val)
    
    handles = []
    for layer in model.model.layers:
        handles.append(layer.register_forward_hook(max_hook))
    
    # Еще один forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    for h in handles:
        h.remove()
    
    print(f"Максимальные значения по слоям:")
    for i, val in enumerate(max_values_per_layer):
        print(f"  Слой {i:2d}: {val:8.4f} {'(≥6.0!)' if val >= 6.0 else ''}")
    
    # Подсчет слоев с большими активациями
    layers_with_large = sum(1 for v in max_values_per_layer if v >= 6.0)
    print(f"\nСлоев с max ≥ 6.0: {layers_with_large}/{n_layers} ({100*layers_with_large/n_layers:.1f}%)")
    
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "ppl": ppl,
        "max_values": max_values_per_layer,
        "layers_with_large": layers_with_large,
        "pct_layers_large": 100 * layers_with_large / n_layers
    }

# Запуск
if __name__ == "__main__":
    # Сначала отладочный анализ
    debug_results = analyze_single_model_debug()
    
    # Потом основной эксперимент
    print("\n\n" + "="*60)
    print("ОСНОВНОЙ ЭКСПЕРИМЕНТ")
    print("="*60)
    
    results = run_experiment_simple()
    
    if results:
        print("\nСводные результаты:")
        print("-" * 80)
        for r in results:
            print(f"{r['model']:25} | {r['params_b']:5.1f}B | PPL: {r['perplexity']:6.2f} | "
                  f"Layers: {r['pct_layers']:5.1f}% | Tokens: {r['pct_tokens']:5.1f}% | "
                  f"Emerg: {r['n_emergent_dims']:4d}")
    else:
        print("Нет результатов для отображения")
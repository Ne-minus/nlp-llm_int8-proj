import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import gc
import math
from datasets import load_dataset


def clear_memory():
    """Агрессивная очистка памяти"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class QuantizedTensor:
    """Контейнер для квантизованного тензора"""
    data: torch.Tensor
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor] = None
    outliers: Optional[torch.Tensor] = None
    outlier_indices: Optional[torch.Tensor] = None
    original_shape: Optional[Tuple] = None
    original_numel: Optional[int] = None
    vector_size: Optional[int] = None
    
    def to(self, device):
        """Перемещение на устройство"""
        self.data = self.data.to(device)
        self.scale = self.scale.to(device)
        if self.zero_point is not None:
            self.zero_point = self.zero_point.to(device)
        if self.outliers is not None:
            self.outliers = self.outliers.to(device)
        if self.outlier_indices is not None:
            self.outlier_indices = self.outlier_indices.to(device)
        return self


class Int8Quantizer:
    """Все методы Int8 квантизации"""
    
    @staticmethod
    def quantize_absmax(tensor: torch.Tensor) -> QuantizedTensor:
        abs_max = tensor.abs().max()
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        return QuantizedTensor(data=quantized, scale=scale, original_shape=tensor.shape)
    
    @staticmethod
    def dequantize_absmax(q: QuantizedTensor) -> torch.Tensor:
        return q.data.float() * q.scale
    
    @staticmethod
    def quantize_zeropoint(tensor: torch.Tensor) -> QuantizedTensor:
        x_min = tensor.min()
        x_max = tensor.max()
        scale = (x_max - x_min) / 255.0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-x_min / scale) - 128
        zero_point = torch.clamp(zero_point, -128, 127)
        quantized = torch.round(tensor / scale + zero_point + 128).clamp(0, 255) - 128
        quantized = quantized.to(torch.int8)
        return QuantizedTensor(data=quantized, scale=scale, zero_point=zero_point, original_shape=tensor.shape)
    
    @staticmethod
    def dequantize_zeropoint(q: QuantizedTensor) -> torch.Tensor:
        return (q.data.float() - q.zero_point) * q.scale
    
    @staticmethod
    def quantize_absmax_rowwise(tensor: torch.Tensor) -> QuantizedTensor:
        original_shape = tensor.shape
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        abs_max = tensor.abs().max(dim=-1, keepdim=True)[0]
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        return QuantizedTensor(data=quantized, scale=scale.squeeze(-1), original_shape=original_shape)
    
    @staticmethod
    def dequantize_absmax_rowwise(q: QuantizedTensor) -> torch.Tensor:
        scale = q.scale.unsqueeze(-1) if q.scale.dim() == 1 else q.scale
        return q.data.float() * scale
    
    @staticmethod
    def quantize_absmax_vectorwise(tensor: torch.Tensor, vector_size: int = 64) -> QuantizedTensor:
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        if numel % vector_size != 0:
            pad_size = vector_size - (numel % vector_size)
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)])
        num_vectors = tensor_flat.numel() // vector_size
        tensor_vectors = tensor_flat.view(num_vectors, vector_size)
        abs_max = tensor_vectors.abs().max(dim=-1, keepdim=True)[0]
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.round(tensor_vectors / scale).clamp(-128, 127).to(torch.int8)
        return QuantizedTensor(data=quantized, scale=scale.squeeze(-1), original_shape=original_shape, original_numel=numel, vector_size=vector_size)
    
    @staticmethod
    def dequantize_absmax_vectorwise(q: QuantizedTensor) -> torch.Tensor:
        scale = q.scale.unsqueeze(-1)
        dequantized = q.data.float() * scale
        dequantized = dequantized.flatten()[:q.original_numel]
        return dequantized.view(q.original_shape)
    
    @staticmethod
    def quantize_zeropoint_vectorwise(tensor: torch.Tensor, vector_size: int = 64) -> QuantizedTensor:
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        if numel % vector_size != 0:
            pad_size = vector_size - (numel % vector_size)
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)])
        num_vectors = tensor_flat.numel() // vector_size
        tensor_vectors = tensor_flat.view(num_vectors, vector_size)
        x_min = tensor_vectors.min(dim=-1, keepdim=True)[0]
        x_max = tensor_vectors.max(dim=-1, keepdim=True)[0]
        scale = (x_max - x_min) / 255.0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-x_min / scale) - 128
        zero_point = torch.clamp(zero_point, -128, 127)
        quantized = torch.round(tensor_vectors / scale + zero_point + 128).clamp(0, 255) - 128
        quantized = quantized.to(torch.int8)
        return QuantizedTensor(data=quantized, scale=scale.squeeze(-1), zero_point=zero_point.squeeze(-1), original_shape=original_shape, original_numel=numel, vector_size=vector_size)
    
    @staticmethod
    def dequantize_zeropoint_vectorwise(q: QuantizedTensor) -> torch.Tensor:
        scale = q.scale.unsqueeze(-1)
        zero_point = q.zero_point.unsqueeze(-1)
        dequantized = (q.data.float() - zero_point) * scale
        dequantized = dequantized.flatten()[:q.original_numel]
        return dequantized.view(q.original_shape)
    
    @staticmethod
    def quantize_absmax_rowwise_decomposition(tensor: torch.Tensor, outlier_threshold: float = 6.0) -> QuantizedTensor:
        original_shape = tensor.shape
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        std = tensor.std()
        threshold = outlier_threshold * std
        outlier_mask = tensor.abs() > threshold
        outlier_indices = outlier_mask.nonzero()
        outlier_values = tensor[outlier_mask].clone()
        tensor_clean = tensor.clone()
        tensor_clean[outlier_mask] = 0
        abs_max = tensor_clean.abs().max(dim=-1, keepdim=True)[0]
        scale = abs_max / 127.0
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.round(tensor_clean / scale).clamp(-128, 127).to(torch.int8)
        return QuantizedTensor(data=quantized, scale=scale.squeeze(-1), outliers=outlier_values, outlier_indices=outlier_indices, original_shape=original_shape)
    
    @staticmethod
    def dequantize_absmax_rowwise_decomposition(q: QuantizedTensor) -> torch.Tensor:
        scale = q.scale.unsqueeze(-1) if q.scale.dim() == 1 else q.scale
        dequantized = q.data.float() * scale
        if q.outliers is not None and q.outlier_indices is not None:
            for idx, value in zip(q.outlier_indices, q.outliers):
                dequantized[tuple(idx)] = value
        return dequantized


class QuantizedLinear(nn.Module):
    """Квантизованный линейный слой"""
    
    METHODS = {
        'absmax': (Int8Quantizer.quantize_absmax, Int8Quantizer.dequantize_absmax),
        'zeropoint': (Int8Quantizer.quantize_zeropoint, Int8Quantizer.dequantize_zeropoint),
        'absmax_rowwise': (Int8Quantizer.quantize_absmax_rowwise, Int8Quantizer.dequantize_absmax_rowwise),
        'absmax_vectorwise': (Int8Quantizer.quantize_absmax_vectorwise, Int8Quantizer.dequantize_absmax_vectorwise),
        'zeropoint_vectorwise': (Int8Quantizer.quantize_zeropoint_vectorwise, Int8Quantizer.dequantize_zeropoint_vectorwise),
        'absmax_rowwise_decomposition': (Int8Quantizer.quantize_absmax_rowwise_decomposition, Int8Quantizer.dequantize_absmax_rowwise_decomposition),
    }
    
    def __init__(self, original_linear: nn.Linear, method: str = 'absmax_rowwise', vector_size: int = 64):
        super().__init__()
        self.method = method
        self.vector_size = vector_size
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.has_bias = original_linear.bias is not None
        
        self.quantize_fn, self.dequantize_fn = self.METHODS[method]
        
        # Квантизуем веса на CPU для экономии памяти
        weight = original_linear.weight.data.float().cpu()
        
        if 'vectorwise' in method:
            self.q_weight = self.quantize_fn(weight, vector_size)
        else:
            self.q_weight = self.quantize_fn(weight)
        
        # Удаляем временный тензор
        del weight
        
        if self.has_bias:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
        self.register_buffer('q_data', self.q_weight.data)
        self.register_buffer('q_scale', self.q_weight.scale)
        
        if self.q_weight.zero_point is not None:
            self.register_buffer('q_zero_point', self.q_weight.zero_point)
        else:
            self.register_buffer('q_zero_point', None)
            
        if self.q_weight.outliers is not None:
            self.register_buffer('q_outliers', self.q_weight.outliers)
            self.register_buffer('q_outlier_indices', self.q_weight.outlier_indices)
        else:
            self.register_buffer('q_outliers', None)
            self.register_buffer('q_outlier_indices', None)
    
    def _rebuild_quantized_tensor(self) -> QuantizedTensor:
        return QuantizedTensor(
            data=self.q_data, scale=self.q_scale, zero_point=self.q_zero_point,
            outliers=self.q_outliers, outlier_indices=self.q_outlier_indices,
            original_shape=self.q_weight.original_shape, original_numel=self.q_weight.original_numel,
            vector_size=self.q_weight.vector_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_tensor = self._rebuild_quantized_tensor()
        weight = self.dequantize_fn(q_tensor).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, weight, self.bias)
    
    def memory_bytes(self) -> int:
        total = self.q_data.numel()
        total += self.q_scale.numel() * 4
        if self.q_zero_point is not None:
            total += self.q_zero_point.numel() * 4
        if self.q_outliers is not None:
            total += self.q_outliers.numel() * 4
            total += self.q_outlier_indices.numel() * 8
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total


class ModelQuantizer:
    """Квантизатор моделей с экономией памяти"""
    
    AVAILABLE_METHODS = [
        'absmax', 'zeropoint', 'absmax_rowwise',
        'absmax_vectorwise', 'zeropoint_vectorwise',
        'absmax_rowwise_decomposition'
    ]
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.quantization_stats: Dict[str, Dict] = {}
        self.eval_results: Dict[str, Dict] = {}
        
    def load_model(self, model_name: str, dtype=torch.float16):
        """Загрузка модели"""
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _reload_model(self, dtype=torch.float16):
        """Перезагрузка модели для следующего метода"""
        print("Reloading model...")
        
        # Удаляем старую модель
        if self.model is not None:
            del self.model
            clear_memory()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        self.model.eval()
    
    def _get_module_by_name(self, model, name):
        """Получить модуль по полному имени"""
        parts = name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def _set_module_by_name(self, model, name, new_module):
        """Установить модуль по полному имени"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    def quantize_model_inplace(
        self, 
        method: str = 'absmax_rowwise',
        vector_size: int = 64,
        target_layers: Optional[List[str]] = None
    ) -> Dict:
        """
        Квантизация модели IN-PLACE с освобождением памяти после каждого слоя
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print(f"\n{'='*60}")
        print(f"Quantizing with method: {method} (memory-efficient)")
        print(f"{'='*60}")
        
        stats = {
            'method': method,
            'layers_quantized': 0,
            'original_memory_mb': 0,
            'quantized_memory_mb': 0,
        }
        
        # Собираем список Linear слоёв
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if target_layers is not None:
                    should_quantize = any(t in name for t in target_layers)
                else:
                    should_quantize = True
                if should_quantize:
                    linear_layers.append(name)
        
        print(f"Found {len(linear_layers)} layers to quantize")
        
        # Получаем устройство модели
        device = next(self.model.parameters()).device
        
        # Квантизуем слои по одному
        for layer_name in tqdm(linear_layers, desc=f"Quantizing ({method})"):
            try:
                original_linear = self._get_module_by_name(self.model, layer_name)
                
                # Считаем оригинальную память
                orig_memory = original_linear.weight.numel() * 2
                if original_linear.bias is not None:
                    orig_memory += original_linear.bias.numel() * 2
                
                # Перемещаем на CPU для квантизации (экономит GPU память)
                original_linear_cpu = nn.Linear(
                    original_linear.in_features,
                    original_linear.out_features,
                    bias=original_linear.bias is not None
                )
                original_linear_cpu.weight.data = original_linear.weight.data.cpu()
                if original_linear.bias is not None:
                    original_linear_cpu.bias.data = original_linear.bias.data.cpu()
                
                # Удаляем оригинальный слой с GPU
                del original_linear
                clear_memory()
                
                # Квантизуем на CPU
                quantized_linear = QuantizedLinear(
                    original_linear_cpu, method=method, vector_size=vector_size
                )
                
                # Удаляем CPU копию
                del original_linear_cpu
                clear_memory()
                
                # Перемещаем квантизованный слой на GPU
                quantized_linear = quantized_linear.to(device)
                
                # Заменяем слой в модели
                self._set_module_by_name(self.model, layer_name, quantized_linear)
                
                quant_memory = quantized_linear.memory_bytes()
                
                stats['layers_quantized'] += 1
                stats['original_memory_mb'] += orig_memory / (1024 * 1024)
                stats['quantized_memory_mb'] += quant_memory / (1024 * 1024)
                
                # Периодическая очистка памяти
                if stats['layers_quantized'] % 10 == 0:
                    clear_memory()
                    
            except Exception as e:
                print(f"Warning: Could not quantize {layer_name}: {e}")
        
        clear_memory()
        
        if stats['quantized_memory_mb'] > 0:
            stats['compression_ratio'] = stats['original_memory_mb'] / stats['quantized_memory_mb']
        else:
            stats['compression_ratio'] = 0
        
        print(f"\nLayers quantized: {stats['layers_quantized']}")
        print(f"Original memory: {stats['original_memory_mb']:.2f} MB")
        print(f"Quantized memory: {stats['quantized_memory_mb']:.2f} MB")
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        
        self.quantization_stats[method] = stats
        return stats
    
    def evaluate_perplexity(self, dataset_name="Salesforce/wikitext", 
                           name="wikitext-2-raw-v1", split="test", max_length=512):
        """Измерение perplexity"""
        print(f"Measuring perplexity on {dataset_name}...")
        
        ds = load_dataset(dataset_name, name, split=split, streaming=False)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        total_loss = 0.0
        total_tokens = 0
        batch_size = 16  # Уменьшил для экономии памяти
        
        with torch.no_grad():
            for i in tqdm(range(0, min(len(ds), 500), batch_size), desc="Perplexity"):  # Ограничил до 500
                batch_texts = [ds[j]['text'] for j in range(i, min(i+batch_size, len(ds)))]
                batch_texts = [text for text in batch_texts if text and text.strip()]
                if not batch_texts:
                    continue
                
                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", truncation=True,
                    max_length=max_length, padding=True
                ).to(device)
                
                labels = inputs["input_ids"].clone()
                labels[inputs["attention_mask"] == 0] = -100
                
                outputs = self.model(**inputs, labels=labels)
                
                batch_tokens = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * batch_tokens
                total_tokens += batch_tokens
                
                # Очистка после каждого батча
                del inputs, labels, outputs
                clear_memory()
        
        ppl = math.exp(total_loss / total_tokens)
        return ppl
    
    def run_all_methods(self, vector_size: int = 64):
        """
        Запуск квантизации и оценки для всех методов последовательно.
        Модель перезагружается для каждого метода.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Сначала оцениваем оригинальную модель
        print("\n" + "="*60)
        print("Evaluating ORIGINAL model")
        print("="*60)
        
        orig_ppl = self.evaluate_perplexity()
        orig_memory = sum(p.numel() * 2 for p in self.model.parameters()) / (1024*1024)
        
        self.eval_results['original'] = {
            'perplexity': orig_ppl,
            'memory_mb': orig_memory,
            'compression_ratio': 1.0,
            'ppl_degradation': 0.0
        }
        print(f"Original perplexity: {orig_ppl:.2f}")
        print(f"Original memory: {orig_memory:.2f} MB")
        
        # Теперь для каждого метода
        for method in self.AVAILABLE_METHODS:
            print("\n" + "="*60)
            print(f"Processing method: {method}")
            print("="*60)
            
            # Перезагружаем модель
            self._reload_model()
            
            # Квантизуем
            stats = self.quantize_model_inplace(method=method, vector_size=vector_size)
            
            # Оцениваем
            ppl = self.evaluate_perplexity()
            
            self.eval_results[method] = {
                'perplexity': ppl,
                'memory_mb': stats['quantized_memory_mb'],
                'compression_ratio': stats['compression_ratio'],
                'ppl_degradation': ppl - orig_ppl
            }
            
            print(f"\n{method}: PPL={ppl:.2f} (Δ={ppl-orig_ppl:+.2f})")
            
            # Очистка перед следующим методом
            clear_memory()
        
        return self.eval_results
    
    def print_comparison_table(self):
        """Печать таблицы сравнения"""
        print("\n" + "=" * 85)
        print("COMPARISON TABLE")
        print("=" * 85)
        print(f"{'Method':<35} {'PPL':<12} {'Memory MB':<12} {'Compress':<12} {'ΔPPL':<10}")
        print("-" * 85)
        
        for method, stats in self.eval_results.items():
            ppl = stats.get('perplexity', 0)
            mem = stats.get('memory_mb', 0)
            comp = stats.get('compression_ratio', 1.0)
            delta = stats.get('ppl_degradation', 0)
            print(f"{method:<35} {ppl:<12.2f} {mem:<12.2f} {comp:<12.2f}x {delta:<+10.2f}")
        
        print("=" * 85)


def main():
    quantizer = ModelQuantizer()
    
    # Загружаем модель
    quantizer.load_model("Qwen/Qwen3-4B")
    
    # Запускаем все методы последовательно с перезагрузкой
    quantizer.run_all_methods(vector_size=64)
    
    # Печатаем результаты
    quantizer.print_comparison_table()


if __name__ == "__main__":
    main()
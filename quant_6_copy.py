#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
from tqdm import tqdm
import gc
import math
import sys


OUTLIER_THRESHOLD = 6.0
MAX_LENGTH = 512
STRIDE = 256  # For sliding window perplexity


class QuantMethod(Enum):
    NONE = "fp16"
    ABSMAX = "absmax"
    ZEROPOINT = "zeropoint"


@dataclass
class QuantResult:
    x_q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    
    def dequantize(self) -> torch.Tensor:
        return (self.x_q.float() - self.zero_point) * self.scale


# ============================================================
# QUANTIZATION FUNCTIONS
# ============================================================

def quantize_absmax(x: torch.Tensor, dim: int = -1) -> QuantResult:
    """Symmetric quantization: [-max_abs, +max_abs] → [-127, +127]"""
    max_abs = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 127.0
    x_q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    zero_point = torch.zeros_like(scale)
    return QuantResult(x_q, scale, zero_point)


def quantize_zeropoint(x: torch.Tensor, dim: int = -1) -> QuantResult:
    """Asymmetric quantization: [min, max] → [-128, +127]"""
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)
    x_range = (x_max - x_min).clamp(min=1e-8)
    
    scale = x_range / 255.0
    zero_point = torch.round(-x_min / scale) - 128.0
    zero_point = zero_point.clamp(-128, 127)
    
    x_q = torch.round(x / scale) + zero_point
    x_q = x_q.clamp(-128, 127).to(torch.int8)
    
    return QuantResult(x_q, scale, zero_point)


def quantize(x: torch.Tensor, method: QuantMethod, dim: int = -1) -> QuantResult:
    if method == QuantMethod.ABSMAX:
        return quantize_absmax(x, dim)
    elif method == QuantMethod.ZEROPOINT:
        return quantize_zeropoint(x, dim)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# INT8 MATMUL
# ============================================================

def matmul_int8_absmax(x_q, w_q, scale_x, scale_w):
    out_int32 = torch.matmul(x_q.float(), w_q.float().T)
    scale_outer = scale_x * scale_w.T
    return out_int32 * scale_outer


def matmul_int8_zeropoint(x_q, w_q, scale_x, scale_w, zp_x, zp_w):
    K = x_q.shape[1]
    x_f, w_f = x_q.float(), w_q.float()
    
    main = torch.matmul(x_f, w_f.T)
    corr1 = zp_x * w_f.sum(dim=1, keepdim=True).T
    corr2 = x_f.sum(dim=1, keepdim=True) * zp_w.T
    corr3 = zp_x * zp_w.T * K
    
    result_int = main - corr1 - corr2 + corr3
    scale_outer = scale_x * scale_w.T
    
    return result_int * scale_outer


# ============================================================
# INT8 LINEAR LAYER
# ============================================================

class Int8LinearDynamic(nn.Module):
    """LLM.int8() dynamic mixed-precision linear layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.float16,
        quant_method: QuantMethod = QuantMethod.ABSMAX,
        threshold: float = OUTLIER_THRESHOLD,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_method = quant_method
        self.threshold = threshold
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_flat = x.view(-1, self.in_features)
        N = x_flat.shape[0]
        device = x.device
        
        # Dynamic outlier detection
        max_abs_per_feature = x_flat.abs().amax(dim=0)
        outlier_mask = max_abs_per_feature > self.threshold
        
        normal_idx = (~outlier_mask).nonzero(as_tuple=False).flatten()
        outlier_idx = outlier_mask.nonzero(as_tuple=False).flatten()
        
        out = torch.zeros(N, self.out_features, device=device, dtype=torch.float32)
        
        # Normal path: Int8
        if normal_idx.numel() > 0:
            x_normal = x_flat[:, normal_idx].float()
            w_normal = self.weight[:, normal_idx].float()
            
            xq = quantize(x_normal, self.quant_method, dim=1)
            wq = quantize(w_normal, self.quant_method, dim=1)
            
            if self.quant_method == QuantMethod.ABSMAX:
                out_normal = matmul_int8_absmax(xq.x_q, wq.x_q, xq.scale, wq.scale)
            else:
                out_normal = matmul_int8_zeropoint(
                    xq.x_q, wq.x_q, xq.scale, wq.scale, xq.zero_point, wq.zero_point
                )
            out = out + out_normal
        
        # Outlier path: FP16
        if outlier_idx.numel() > 0:
            x_outlier = x_flat[:, outlier_idx]
            w_outlier = self.weight[:, outlier_idx]
            out_outlier = F.linear(x_outlier, w_outlier, None)
            out = out + out_outlier.float()
        
        if self.bias is not None:
            out = out + self.bias.float()
        
        return out.view(*orig_shape[:-1], self.out_features).to(orig_dtype)


# ============================================================
# MODEL CONVERSION
# ============================================================

def convert_model(
    model: nn.Module,
    quant_method: QuantMethod,
    threshold: float = OUTLIER_THRESHOLD,
    skip_patterns: tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> nn.Module:
    """Convert Linear layers to Int8LinearDynamic"""
    if quant_method == QuantMethod.NONE:
        return model
    
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            if any(p in name for p in skip_patterns):
                continue
            
            new_layer = Int8LinearDynamic(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
                quant_method=quant_method,
                threshold=threshold,
            )
            
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias)
            
            setattr(model, name, new_layer)
        else:
            convert_model(child, quant_method, threshold, skip_patterns)
    
    return model


# ============================================================
# WIKITEXT PERPLEXITY COMPUTATION
# ============================================================

def load_wikitext(
    dataset_name: str = "wikitext",
    config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> str:
    """Load WikiText dataset and concatenate all text"""
    print(f"📥 Loading {dataset_name}/{config} [{split}]...")
    dataset = load_dataset(dataset_name, config, split=split)
    
    # Concatenate all text with newlines
    full_text = "\n\n".join([item["text"] for item in dataset if item["text"].strip()])
    
    print(f"   Total characters: {len(full_text):,}")
    return full_text


def compute_perplexity_wikitext(
    model: nn.Module,
    tokenizer,
    text: str,
    max_length: int = MAX_LENGTH,
    stride: int = STRIDE,
    device: str = None,
) -> Dict:
    """
    Compute perplexity using sliding window approach.
    
    This handles long texts by:
    1. Tokenizing the full text
    2. Using sliding window with stride
    3. Only counting loss for non-overlapping tokens
    
    Returns dict with perplexity and statistics.
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize full text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    seq_len = input_ids.size(1)
    print(f"   Total tokens: {seq_len:,}")
    
    nlls = []  # Negative log likelihoods
    prev_end_loc = 0
    
    # Progress bar
    num_windows = (seq_len - 1) // stride + 1
    pbar = tqdm(range(0, seq_len, stride), desc="   Computing PPL", total=num_windows)
    
    with torch.no_grad():
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # Number of tokens to predict
            
            input_chunk = input_ids[:, begin_loc:end_loc]
            target_chunk = input_chunk.clone()
            
            # Mask tokens we've already computed loss for
            # Only compute loss for the last `trg_len` tokens
            target_chunk[:, :-trg_len] = -100
            
            outputs = model(input_chunk, labels=target_chunk)
            
            # Negative log likelihood for this chunk
            # outputs.loss is averaged over non-masked tokens
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            
            # Update progress bar
            current_ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc).item()
            pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})
            
            if end_loc == seq_len:
                break
    
    # Compute final perplexity
    total_nll = torch.stack(nlls).sum()
    total_tokens = prev_end_loc
    perplexity = torch.exp(total_nll / total_tokens).item()
    
    return {
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_windows": len(nlls),
        "avg_nll": (total_nll / total_tokens).item(),
    }


def compute_perplexity_simple(
    model: nn.Module,
    tokenizer,
    text: str,
    max_length: int = MAX_LENGTH,
    max_samples: int = 200,
    device: str = None,
) -> Dict:
    """
    Simpler perplexity computation: split text into chunks.
    Faster but slightly less accurate than sliding window.
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # Remove batch dim
    
    seq_len = len(input_ids)
    print(f"   Total tokens: {seq_len:,}")
    
    # Split into chunks
    chunks = []
    for i in range(0, seq_len - 1, max_length - 1):
        chunk = input_ids[i : i + max_length]
        if len(chunk) > 10:  # Skip very short chunks
            chunks.append(chunk)
    
    if max_samples and len(chunks) > max_samples:
        # Sample evenly
        indices = torch.linspace(0, len(chunks) - 1, max_samples).long().tolist()
        chunks = [chunks[i] for i in indices]
    
    print(f"   Processing {len(chunks)} chunks...")
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for chunk in tqdm(chunks, desc="   Computing PPL"):
            input_chunk = chunk.unsqueeze(0).to(device)
            
            outputs = model(input_chunk, labels=input_chunk)
            
            n_tokens = len(chunk) - 1  # Exclude first token (no prediction)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_chunks": len(chunks),
        "avg_loss": avg_loss,
    }

def clear_memory():
    """Агрессивная очистка памяти"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def evaluate_perplexity(model, tokenizer, dataset_name="Salesforce/wikitext", 
                           name="wikitext-2-raw-v1", split="test", max_length=512):
        """Измерение perplexity"""
        print(f"Measuring perplexity on {dataset_name}...")
        
        ds = load_dataset(dataset_name, name, split=split, streaming=False)
        
        model.eval()
        device = next(model.parameters()).device
        
        total_loss = 0.0
        total_tokens = 0
        batch_size = 16  # Уменьшил для экономии памяти
        
        with torch.no_grad():
            for i in tqdm(range(0, len(ds), 500), desc="Perplexity"): 
                batch_texts = [ds[j]['text'] for j in range(i, min(i+batch_size, len(ds)))]
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


# ============================================================
# MAIN BENCHMARK
# ============================================================

def benchmark_wikitext(size=0.6):
    """Full benchmark on WikiText-2"""
    print("=" * 70)
    print("WIKITEXT-2 PERPLEXITY BENCHMARK")
    print("FP16 vs Absmax (Symmetric) vs ZeroPoint (Asymmetric)")
    print("=" * 70)

    wikitext = load_wikitext("wikitext", "wikitext-2-raw-v1", "test")

    #sizes = [0.6, 1.7, 4, 8]
    
    print(f"Model size: {size}")
    #for size in tqdm(sizes):
    MODEL_NAME = f"Qwen/Qwen3-{size}B"
    print(MODEL_NAME)

    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load WikiText
    # wikitext = load_wikitext("wikitext", "wikitext-2-raw-v1", "test")
    
    results = {}


    for method in [QuantMethod.NONE, QuantMethod.ABSMAX, QuantMethod.ZEROPOINT]:
        print(f"\n{'='*60}")
        print(f"📊 Method: {method.value.upper()}")
        print("=" * 60)
        
        # Load model
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Convert if needed
        if method != QuantMethod.NONE:
            print(f"🔄 Converting to {method.value}...")
            convert_model(model, method)
        
        model.eval()
        
        # Compute perplexity
        print("🧮 Computing perplexity...")
        result = evaluate_perplexity(
            model, 
            tokenizer,
            dataset_name="allenai/c4",
            name="realnewslike",
            split="validation"

        )
        
        results[method] = result
        print(f"✅ {method.value.upper()} Perplexity: {result['perplexity']:.2f}")
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Print summary
        print_results_summary(results)
        
        # return results


def print_results_summary(results: Dict):
    """Print formatted results table"""
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY - WikiText-2 Test Set")
    print("=" * 70)
    
    baseline_ppl = results[QuantMethod.NONE]["perplexity"]
    
    print(f"\n{'Method':<20} {'Perplexity':<12} {'Tokens':<12} {'Degradation':<15} {'Status'}")
    print("-" * 75)
    
    for method, result in results.items():
        ppl = result["perplexity"]
        tokens = result["total_tokens"]
        
        if method == QuantMethod.NONE:
            deg_str = "—"
            status = "Baseline"
        else:
            deg = ((ppl / baseline_ppl) - 1) * 100
            deg_str = f"{deg:+.2f}%"
            
            if abs(deg) < 0.5:
                status = "🎉 Excellent (<0.5%)"
            elif abs(deg) < 1.0:
                status = "✅ Great (<1%)"
            elif deg < 2.0:
                status = "👍 Good (<2%)"
            elif deg < 5.0:
                status = "⚠️ Acceptable (<5%)"
            else:
                status = "❌ High degradation"
        
        print(f"{method.value.upper():<20} {ppl:<12.2f} {tokens:<12,} {deg_str:<15} {status}")
    
    print("\n" + "=" * 70)


def quick_test():
    """Quick test with fewer samples for debugging"""
    print("=" * 70)
    print("QUICK TEST (subset of WikiText-2)")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load just a portion
    wikitext = load_wikitext("wikitext", "wikitext-2-raw-v1", "test")
    wikitext = wikitext[:50000]  # First 50k chars
    
    results = {}
    
    for method in [QuantMethod.NONE, QuantMethod.ABSMAX, QuantMethod.ZEROPOINT]:
        print(f"\n🔄 Testing {method.value.upper()}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if method != QuantMethod.NONE:
            convert_model(model, method)
        
        model.eval()
        
        result = compute_perplexity_simple(
            model, tokenizer, wikitext, 
            max_length=256, max_samples=50
        )
        
        results[method] = result
        print(f"   PPL: {result['perplexity']:.2f}")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    print_results_summary(results)


def test_single_method(method: QuantMethod):
    """Test a single quantization method"""
    print(f"\n{'='*60}")
    print(f"Testing: {method.value.upper()}")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    wikitext = load_wikitext("wikitext", "wikitext-2-raw-v1", "test")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if method != QuantMethod.NONE:
        convert_model(model, method)
    
    model.eval()
    
    result = compute_perplexity_wikitext(model, tokenizer, wikitext)
    
    print(f"\n✅ Results for {method.value.upper()}:")
    print(f"   Perplexity: {result['perplexity']:.4f}")
    print(f"   Total tokens: {result['total_tokens']:,}")
    print(f"   Avg NLL: {result['avg_nll']:.4f}")
    
    return result


# ============================================================
# ADDITIONAL UTILITIES
# ============================================================

def compare_on_c4():
    """Optional: Compare on C4 dataset"""
    print("=" * 70)
    print("C4 PERPLEXITY BENCHMARK")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load C4
    print("📥 Loading C4 dataset...")
    dataset = load_dataset(
        "allenai/c4", 
        "realnewslike", 
        split="validation",
        streaming=True
    )
    
    # Take subset
    texts = []
    for i, item in enumerate(dataset):
        if i >= 100:
            break
        texts.append(item["text"])
    
    full_text = "\n\n".join(texts)
    print(f"   Loaded {len(texts)} documents, {len(full_text):,} chars")
    
    results = {}
    
    for method in [QuantMethod.NONE, QuantMethod.ABSMAX, QuantMethod.ZEROPOINT]:
        print(f"\n🔄 Testing {method.value.upper()}...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if method != QuantMethod.NONE:
            convert_model(model, method)
        
        model.eval()
        
        result = compute_perplexity_simple(
            model, tokenizer, full_text,
            max_length=512, max_samples=100
        )
        
        results[method] = result
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    print_results_summary(results)
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WikiText Perplexity Benchmark")
    parser.add_argument(
        "--mode", 
        choices=["full", "quick", "single", "c4"],
        default="full",
        help="Benchmark mode"
    )
    parser.add_argument(
        "--method",
        choices=["fp16", "absmax", "zeropoint"],
        default="fp16",
        help="Method for single mode"
    )
    parser.add_argument(
        "--size",
        choices=["0.6", "1.7", "4", "8"],
        default="0.6",
        help="qwen model size"
    )
    
    args = parser.parse_args()
    size = args.size
    if args.mode == "full":
        benchmark_wikitext(size)
    elif args.mode == "quick":
        quick_test()
    elif args.mode == "single":
        method_map = {
            "fp16": QuantMethod.NONE,
            "absmax": QuantMethod.ABSMAX,
            "zeropoint": QuantMethod.ZEROPOINT,
        }
        test_single_method(method_map[args.method])
    elif args.mode == "c4":
        compare_on_c4()


if __name__ == "__main__":
    main()
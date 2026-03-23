#!/usr/bin/env python3
"""
추론 백엔드 등록소 — evaluate_sqa, run_inference 등에서 모델에 무관하게 사용.

인터페이스: inference(audio_path, prompt, max_new_tokens), get_next_token_logits(audio_path, text_input, answer_suffix), .processor
모델별 구현은 Qwen_2_Audio_inference.py, Llama_*_inference.py 등에 두고 여기서만 등록.
"""

import os
from typing import Any, Callable, Dict

# 백엔드 이름 -> (model_path, **kwargs) -> backend 인스턴스
_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str):
    """백엔드 등록 데코레이터. 사용: @register("qwen") def get_qwen(model_path, **kwargs): ..."""

    def decorator(fn: Callable[..., Any]):
        _REGISTRY[name.lower()] = fn
        return fn

    return decorator


def get_backend(backend_name: str, model_path: str = None, **kwargs) -> Any:
    """
    백엔드 이름과 모델 경로로 추론 백엔드 인스턴스 반환.
    model_path가 None이거나 비어 있으면 백엔드별 기본 모델 사용 (HuggingFace cache/다운로드).
    """
    name = (backend_name or "qwen").lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"지원하지 않는 백엔드: '{backend_name}'. 사용 가능: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](model_path or "", **kwargs)


def list_backends():
    """등록된 백엔드 이름 목록."""
    return list(_REGISTRY.keys())


# --- Qwen2-Audio 등록 ---
@register("qwen")
def _get_qwen(model_path: str, **kwargs):
    from Qwen_2_Audio_inference import Qwen2AudioModel
    return Qwen2AudioModel(model_path, **kwargs)


# --- Qwen3-Omni 등록 ---
@register("qwen3")
def _get_qwen3(model_path: str, **kwargs):
    from Qwen3_Omni_inference import Qwen3OmniModel
    return Qwen3OmniModel(model_path, **kwargs)


# --- Qwen2-Audio vLLM 등록 ---
@register("qwen_vllm")
def _get_qwen_vllm(model_path: str, **kwargs):
    from Qwen2_Audio_vllm_inference import Qwen2AudioVLLMModel
    return Qwen2AudioVLLMModel(model_path, **kwargs)


# --- Qwen3-Omni vLLM 등록 (ASR/SQA/Translation/Instruct 모두 지원, SQA는 generate 후 파싱) ---
@register("qwen3_vllm")
def _get_qwen3_vllm(model_path: str, **kwargs):
    from Qwem3_Omni_vllm_inference import Qwen3OmniVLLMModel
    return Qwen3OmniVLLMModel(model_path, **kwargs)


# --- Llama-3.1-8B-Omni 등록 ---
@register("llama3_omni")
def _get_llama3_omni(model_path: str, **kwargs):
    import importlib.util
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "llama3_omni_inference",
        os.path.join(_src_dir, "Llama3.1-8B-Omni_inference.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.Llama3OmniModel(model_path, **kwargs)


# --- HyperCLOVAX 등록 ---
@register("hyperclovax")
def _get_hyperclovax(model_path: str, **kwargs):
    from hyperclovaX_inference import HyperCLOVAXModel
    return HyperCLOVAXModel(model_path, **kwargs)


# --- Gemma-3n vLLM 등록 ---
@register("gemma3n_vllm")
def _get_gemma3n_vllm(model_path: str, **kwargs):
    from Gemma3n_vllm_inference import Gemma3nVLLMModel
    return Gemma3nVLLMModel(model_path, **kwargs)


@register("gpt_realtime_mini")
def _get_gpt_realtime_mini(model_path: str, **kwargs):
    from GPT_Realtime_inference import GPTRealtimeMiniModel
    return GPTRealtimeMiniModel(model_path, **kwargs)


# --- Voxtral Mini 3B 등록 ---
@register("voxtral3b")
def _get_voxtral3b(model_path: str, **kwargs):
    from voxtral_3B_inference import VoxtralMini3BModel
    return VoxtralMini3BModel(model_path, **kwargs)


# --- Voxtral Mini 3B vLLM 등록 (vLLM >= 0.10.0, tokenizer_mode/config_format/load_format mistral) ---
@register("voxtral3b_vllm")
def _get_voxtral3b_vllm(model_path: str, **kwargs):
    from Voxtral_vllm_inference import VoxtralVLLMModel
    return VoxtralVLLMModel(model_path, **kwargs)


# --- Gemini Flash / Flash-Lite 등록 ---
@register("gemini_flash")
def _get_gemini_flash(model_path: str, **kwargs):
    from Gemini_flash_inference import GeminiFlashModel
    return GeminiFlashModel(model_path, **kwargs)


# --- Kimi-Audio 등록 ---
@register("kimi")
@register("kimi_audio")
def _get_kimi_audio(model_path: str, **kwargs):
    import importlib.util

    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "kimi_audio_inference",
        os.path.join(_src_dir, "Kimi-Audio_inference.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.KimiAudioModel(model_path, **kwargs)

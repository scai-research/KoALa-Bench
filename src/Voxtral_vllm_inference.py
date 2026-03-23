#!/usr/bin/env python3
"""
Voxtral Mini 3B — vLLM 백엔드 (mistral_common + vLLM 공식 지원)

vLLM >= 0.10.0 에서 Voxtral 공식 지원.
- tokenizer_mode=mistral, config_format=mistral, load_format=mistral
- mistral_common으로 UserMessage(AudioChunk + TextChunk) → encode_chat_completion → prompt_token_ids + multi_modal_data

지원 task: asr, sqa, instruct, LSQA (backends 인터페이스 호환)
"""

import os
import sys
import re
import json
import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders import get_dataloader, list_available_tasks


# --- SQA 헬퍼 (Gemma3n/Voxtral 공통) ---
SQA_ANSWER_SUFFIX = "\n답: "


def _parse_choice_letters(choices_ko: str) -> List[str]:
    if not (choices_ko or "").strip():
        return ["A", "B", "C"]
    letters = []
    for m in re.finditer(r"\(([A-Z])\)", (choices_ko or "").strip(), re.IGNORECASE):
        letters.append(m.group(1).upper())
    seen = set()
    out = []
    for L in letters:
        if L not in seen:
            seen.add(L)
            out.append(L)
    return out if out else ["A", "B", "C"]


def _normalize_gt_to_letter(answer_ko: str) -> Optional[str]:
    raw = (answer_ko or "").strip()
    if not raw:
        return None
    m = re.match(r"^\(([A-Z])\)", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if raw.upper() in ("A", "B", "C", "D", "E"):
        return raw.upper()
    return None


def _parse_sqa_prediction_from_text(generated_text: str, choice_letters: List[str]) -> str:
    text = (generated_text or "").strip()
    if not text:
        return choice_letters[0]
    for L in choice_letters:
        if f"({L})" in text or f"（{L}）" in text:
            return L
    for L in choice_letters:
        if re.search(rf"\b{L}\b", text, re.IGNORECASE):
            return L
    return choice_letters[0]


def save_jsonl(data: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _audio_path_for_segment(
    audio_path: str,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    """offset/duration 적용 시 해당 구간만 임시 wav로 저장. 반환: (사용할_경로, 정리할_임시경로 또는 None)."""
    if (offset is None or offset == 0.0) and (duration is None or duration <= 0):
        return (audio_path, None)
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        return (audio_path, None)
    try:
        kwargs = {"sr": 16000, "offset": float(offset)}
        if duration is not None and duration > 0:
            kwargs["duration"] = float(duration)
        audio, sr = librosa.load(audio_path, **kwargs)
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(temp_path, audio, sr)
        return (temp_path, temp_path)
    except Exception as e:
        print(f"[Voxtral vLLM] 오디오 구간 로드 실패 [{audio_path}]: {e}")
        return (audio_path, None)


class VoxtralVLLMModel:
    """
    Voxtral Mini 3B — vLLM 백엔드.
    mistral_common으로 채팅 인코딩 후 prompt_token_ids + multi_modal_data 로 generate.
    """

    DEFAULT_MODEL_PATH = "mistralai/Voxtral-Mini-3B-2507"

    def __init__(
        self,
        model_path: str = None,
        tensor_parallel_size: int = 1,
        max_new_tokens_default: int = 512,
        **kwargs,
    ):
        from vllm import LLM, SamplingParams
        from mistral_common.audio import Audio
        from mistral_common.protocol.instruct.chunk import (
            AudioChunk,
            RawAudio,
            TextChunk,
        )
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self.model_path = (model_path or "").strip() or self.DEFAULT_MODEL_PATH
        self.tensor_parallel_size = tensor_parallel_size
        self.max_new_tokens_default = max_new_tokens_default
        self._Audio = Audio
        self._AudioChunk = AudioChunk
        self._RawAudio = RawAudio
        self._TextChunk = TextChunk
        self._UserMessage = UserMessage
        self._ChatCompletionRequest = ChatCompletionRequest

        print(f"Voxtral vLLM: MistralTokenizer 로딩 — {self.model_path}")
        self.tokenizer = MistralTokenizer.from_hf_hub(self.model_path)

        print(f"Voxtral vLLM: LLM 로딩 — {self.model_path} (tp={tensor_parallel_size})")
        # LSQA는 긴 오디오+질문이라 256이면 부족 → 기본 2048. ASR만 돌릴 땐 VOXTRAL_VLLM_MAX_MODEL_LEN=256 설정으로 메모리 절약.
        _max_len = int(os.environ.get("VOXTRAL_VLLM_MAX_MODEL_LEN", "2048"))
        # enforce_eager=True 로 OOM(std::bad_alloc) 완화
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"audio": 1},
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=_max_len,
            max_num_batched_tokens=_max_len,
            enforce_eager=True,
        )
        self.SamplingParams = SamplingParams
        print("Voxtral vLLM 로딩 완료")

    @property
    def processor(self):
        return self.tokenizer

    def _encode_audio_prompt(
        self,
        audio_path: str,
        prompt: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[Tuple[Optional[List[int]], Optional[Dict]], Optional[str]]:
        """
        (audio_path, prompt) → (prompt_token_ids, multi_modal_data).
        offset/duration 사용 시 임시 파일 생성; 반환값 세 번째는 정리할 임시 경로(없으면 None).
        """
        path_to_use, temp_path = _audio_path_for_segment(audio_path, offset=offset, duration=duration)
        try:
            audio = self._Audio.from_file(path_to_use, strict=False)
        except Exception as e:
            print(f"[Voxtral vLLM] Audio.from_file 실패 [{path_to_use}]: {e}")
            return (None, None), temp_path

        audio_chunk = self._AudioChunk(input_audio=self._RawAudio.from_audio(audio))
        text_chunk = self._TextChunk(text=prompt or "")
        messages = [self._UserMessage(content=[audio_chunk, text_chunk])]
        req = self._ChatCompletionRequest(messages=messages, model=self.model_path)
        try:
            tokens = self.tokenizer.encode_chat_completion(req)
        except Exception as e:
            print(f"[Voxtral vLLM] encode_chat_completion 실패: {e}")
            return (None, None), temp_path

        prompt_ids = tokens.tokens
        if not tokens.audios:
            return (None, None), temp_path
        audios_and_sr = [(au.audio_array, au.sampling_rate) for au in tokens.audios]
        multi_modal_data = {"audio": audios_and_sr}
        return (prompt_ids, multi_modal_data), temp_path

    def inference(
        self,
        audio_path: str,
        prompt: str,
        max_new_tokens: int = 256,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> str:
        if not os.path.exists(audio_path):
            print(f"오디오 파일 없음: {audio_path}")
            return ""
        (payload, temp_path) = self._encode_audio_prompt(audio_path, prompt, offset=offset, duration=duration)
        if payload[0] is None:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            return ""
        prompt_ids, multi_modal_data = payload
        sampling_params = self.SamplingParams(max_tokens=max_new_tokens, temperature=0)
        inputs = [{"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}]
        try:
            outputs = self.llm.generate(inputs, sampling_params)
            if outputs and outputs[0].outputs:
                text = outputs[0].outputs[0].text.strip()
            else:
                text = ""
        finally:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        return text

    def inference_batch(
        self,
        items: List[Dict],
        max_new_tokens: int = 256,
        return_first_logits: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[Optional[object]]]]:
        if not items:
            return ([], []) if return_first_logits else []
        results = [""] * len(items)
        sampling_params = self.SamplingParams(max_tokens=max_new_tokens, temperature=0)
        for i, item in enumerate(items):
            audio_path = item.get("audio_path", "")
            prompt = item.get("prompt", item.get("text_input", ""))
            offset = item.get("offset", 0.0)
            duration = item.get("duration", None)
            if not os.path.exists(audio_path):
                continue
            (payload, temp_path) = self._encode_audio_prompt(audio_path, prompt, offset=offset, duration=duration)
            if payload[0] is None:
                if temp_path and os.path.isfile(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                continue
            prompt_ids, multi_modal_data = payload
            try:
                outputs = self.llm.generate(
                    [{"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}],
                    sampling_params,
                )
                if outputs and outputs[0].outputs:
                    results[i] = outputs[0].outputs[0].text.strip()
            finally:
                if temp_path and os.path.isfile(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
        if return_first_logits:
            return (results, [None] * len(items))
        return results

    def get_next_token_logits(
        self,
        audio_path: str,
        text_input: str,
        answer_suffix: str = "\n답: ",
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Optional[Dict[int, float]]:
        """SQA/LSQA: 답 suffix 다음 1토큰에 대한 logprobs (token_id -> logprob)."""
        if not os.path.exists(audio_path):
            return None
        prompt = (text_input or "").strip() + (answer_suffix or "")
        (payload, temp_path) = self._encode_audio_prompt(audio_path, prompt, offset=offset, duration=duration)
        if payload[0] is None:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            return None
        prompt_ids, multi_modal_data = payload
        sampling_params = self.SamplingParams(max_tokens=1, temperature=0, logprobs=100)
        try:
            outputs = self.llm.generate(
                [{"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}],
                sampling_params,
            )
        except Exception:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            return None
        if temp_path and os.path.isfile(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        if not outputs or not outputs[0].outputs:
            return None
        logprobs_pos0 = outputs[0].outputs[0].logprobs
        if not logprobs_pos0 or len(logprobs_pos0) == 0:
            return None
        first = logprobs_pos0[0]
        if hasattr(first, "items"):
            return {tid: lp.logprob for tid, lp in first.items()}
        return None

    def get_next_token_logits_batch(
        self,
        items: List[Dict],
        answer_suffix: str = "\n답: ",
    ) -> List[Optional[Dict[int, float]]]:
        out: List[Optional[Dict[int, float]]] = []
        for item in items:
            out.append(
                self.get_next_token_logits(
                    audio_path=item.get("audio_path", ""),
                    text_input=item.get("text_input", ""),
                    answer_suffix=answer_suffix,
                    offset=item.get("offset", 0.0),
                    duration=item.get("duration", None),
                )
            )
        return out


def run_inference(
    task: str,
    input_jsonl: str,
    output_jsonl: str,
    model_path: str,
    custom_prompt: str = None,
    max_samples: int = None,
    max_new_tokens: int = 256,
    base_dir: str = None,
    batch_size: int = 8,
    tensor_parallel_size: int = 1,
) -> Dict:
    base_dir = Path(base_dir).resolve() if base_dir else None
    print(f"\n[Task: {task}] (Voxtral vLLM)")
    print(f"입력 파일: {input_jsonl}")

    loader_kwargs = {"max_samples": max_samples, "custom_prompt": custom_prompt}
    if task == "instruct" and base_dir:
        loader_kwargs["base_dir"] = str(base_dir)
    dataloader = get_dataloader(task=task, jsonl_path=input_jsonl, **loader_kwargs)

    model = VoxtralVLLMModel(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_new_tokens_default=max_new_tokens,
    )

    start_time = time.time()
    results: List[Dict] = []
    success_count = 0
    fail_count = 0

    all_items = []
    for item in dataloader:
        audio_path = item["audio_path"]
        if base_dir and not os.path.isabs(audio_path):
            item["audio_path"] = str(base_dir / audio_path)
        all_items.append(item)

    if task == "sqa":
        all_inputs = []
        all_meta = []
        for item in all_items:
            audio_path = item["audio_path"]
            index = item["index"]
            text_input = item.get("text_input", "")
            choices_ko = item.get("choices_ko", "")
            answer_ko = item.get("answer_ko", "")
            choice_letters = _parse_choice_letters(choices_ko)
            gt_letter = _normalize_gt_to_letter(answer_ko)
            if not os.path.exists(audio_path):
                results.append({"index": index, "prediction": "", "answer_ko": answer_ko, "correct": False, "note": "audio_missing"})
                fail_count += 1
                continue
            prompt = (text_input or "").strip() + SQA_ANSWER_SUFFIX
            all_inputs.append({"audio_path": audio_path, "prompt": prompt})
            all_meta.append((index, answer_ko, choice_letters, gt_letter))
        if all_inputs:
            generated_list = model.inference_batch(all_inputs, max_new_tokens=max_new_tokens)
            for i, meta in enumerate(all_meta):
                index, answer_ko, choice_letters, gt_letter = meta
                pred_letter = _parse_sqa_prediction_from_text(generated_list[i], choice_letters)
                is_correct = gt_letter is not None and pred_letter == gt_letter
                if is_correct:
                    success_count += 1
                else:
                    fail_count += 1
                results.append({"index": index, "prediction": f"({pred_letter})", "answer_ko": answer_ko, "correct": is_correct})
    else:
        all_inputs = []
        all_meta = []
        missing_results = []
        for item in all_items:
            audio_path = item["audio_path"]
            index = item["index"]
            prompt = item.get("text_input", item.get("prompt", ""))
            if not os.path.exists(audio_path):
                missing_results.append({"index": index, "prediction": ""})
                fail_count += 1
                continue
            all_inputs.append({"audio_path": audio_path, "prompt": prompt})
            all_meta.append(index)
        if all_inputs:
            generated_list = model.inference_batch(all_inputs, max_new_tokens=max_new_tokens)
            for i, index in enumerate(all_meta):
                generated = generated_list[i]
                results.append({"index": index, "prediction": generated})
                if generated:
                    success_count += 1
                else:
                    fail_count += 1
        results.extend(missing_results)

    elapsed_time = time.time() - start_time
    save_jsonl(results, output_jsonl)

    stats = {
        "task": task,
        "input_file": input_jsonl,
        "output_file": output_jsonl,
        "model": model_path,
        "total_samples": len(results),
        "success": success_count,
        "fail": fail_count,
        "elapsed_time_seconds": elapsed_time,
        "avg_time_per_sample": elapsed_time / len(results) if results else 0,
        "timestamp": datetime.now().isoformat(),
    }
    print(f"\n추론 완료 [{task}] (Voxtral vLLM) — 성공: {success_count}, 실패: {fail_count}, 소요: {elapsed_time:.1f}초")
    print(f"출력: {output_jsonl}")
    return stats


def main():
    available_tasks = list_available_tasks()
    parser = argparse.ArgumentParser(description="Voxtral Mini 3B vLLM 추론")
    parser.add_argument("--task", "-t", required=True, choices=available_tasks)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--model", "-m", default=VoxtralVLLMModel.DEFAULT_MODEL_PATH)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--prompt", "-p", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    args = parser.parse_args()

    run_inference(
        task=args.task,
        input_jsonl=args.input,
        output_jsonl=args.output,
        model_path=args.model,
        custom_prompt=args.prompt,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()

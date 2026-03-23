#!/usr/bin/env python3
"""
Gemma-3n vLLM 로컬 추론 스크립트

vLLM의 LLM.generate()로 직접 배치 추론. 서버 불필요.

입력: JSONL 파일 (task별 dataloader 사용)
출력: JSONL 파일 (index, prediction [, task별 필드])

지원 task: asr, sqa, instruct
지원 모델: google/gemma-3n-E2B-it, google/gemma-3n-E4B-it
"""

import os
import sys
import re
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import librosa
from tqdm import tqdm

# dataloaders 모듈 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders import get_dataloader, list_available_tasks


# --- SQA용 헬퍼 ---
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
    """generate 출력에서 선택지 글자 추출."""
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
    """JSONL 파일 저장"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


TARGET_SR = 16000


def load_audio_for_vllm(
    audio_path: str,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Optional[tuple]:
    """오디오 파일을 vLLM 형식으로 로드 (offset/duration 지원)
    
    Args:
        audio_path: 오디오 파일 경로
        offset: 오디오 시작 시점(초). Translation 구간 번역용
        duration: 로드할 길이(초). None이면 끝까지
    
    Returns:
        (audio_array, sample_rate) 튜플 또는 실패 시 None
    """
    try:
        kwargs = {"sr": TARGET_SR, "offset": offset}
        if duration is not None:
            kwargs["duration"] = duration
        audio, sr = librosa.load(audio_path, **kwargs)
        return (audio.astype(np.float32), sr)
    except Exception as e:
        print(f"오디오 로드 실패 [{audio_path}]: {e}")
        return None


class Gemma3nVLLMModel:
    """
    vLLM 로컬 기반 Gemma-3n 백엔드.
    서버 없이 직접 LLM.generate() 사용.
    """

    DEFAULT_MODEL_PATH = "google/gemma-3n-E4B-it"

    def __init__(
        self,
        model_path: str = None,
        tensor_parallel_size: int = 1,
        max_new_tokens_default: int = 512,
        **kwargs
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        
        self.model_path = (model_path or "").strip() or self.DEFAULT_MODEL_PATH
        self.tensor_parallel_size = tensor_parallel_size
        self.max_new_tokens_default = max_new_tokens_default
        
        # Processor 로딩 (chat template 적용용)
        print(f"Processor 로딩 중: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        print(f"vLLM 모델 로딩 중: {self.model_path} (tp={tensor_parallel_size})")
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"audio": 1},
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams
        print("vLLM 모델 로딩 완료")

    def _build_prompt(self, text: str, audio_path: str = None) -> str:
        """
        Gemma-3n 프롬프트 형식.

        사용자 제안대로, 모델 카드에서 사용하는 스페셜 토큰
        `<start_of_turn>`, `<end_of_turn>`, `<audio_soft_token>`을 직접 사용한다.
        오디오는 vLLM `multi_modal_data={"audio": ...}` 로 전달되므로,
        텍스트 프롬프트에서는 오디오 소프트 토큰만 넣어준다.
        """
        text = (text or "").strip()
        if audio_path:
            return (
                "<start_of_turn>user\n"
                f"<audio_soft_token>{text}"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        else:
            return (
                "<start_of_turn>user\n"
                f"{text}"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )

    def inference(
        self,
        audio_path: str,
        prompt: str,
        max_new_tokens: int = 256,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> str:
        """단일 오디오 추론 (offset/duration으로 구간 지정 가능)"""
        if not os.path.exists(audio_path):
            print(f"오디오 파일 없음: {audio_path}")
            return ""
        
        audio_data = load_audio_for_vllm(audio_path, offset=offset, duration=duration)
        if audio_data is None:
            return ""
        
        # chat template 적용된 프롬프트 생성
        prompt_text = self._build_prompt(prompt, audio_path)
        
        sampling_params = self.SamplingParams(max_tokens=max_new_tokens, temperature=0)
        
        inputs = [{
            "prompt": prompt_text,
            "multi_modal_data": {"audio": audio_data},
        }]
        
        outputs = self.llm.generate(inputs, sampling_params)
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text.strip()
        return ""

    def inference_batch(
        self,
        items: List[Dict],
        max_new_tokens: int = 256,
        return_first_logits: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[Optional[object]]]]:
        """배치 추론 - vLLM이 내부적으로 최적화 (offset/duration 지원)
        
        Args:
            items: 각 항목은 audio_path, prompt, (선택) offset, duration
            max_new_tokens: 최대 생성 토큰 수
            return_first_logits: True면 첫 토큰의 logits도 반환 (vLLM에서는 미지원)
        
        Returns:
            생성된 텍스트 리스트 (또는 return_first_logits=True면 튜플)
        """
        if not items:
            return ([], []) if return_first_logits else []
        
        sampling_params = self.SamplingParams(max_tokens=max_new_tokens, temperature=0)
        
        # 유효한 입력만 필터링
        valid_inputs = []
        valid_indices = []
        
        for i, item in enumerate(items):
            audio_path = item.get("audio_path", "")
            prompt = item.get("prompt", item.get("text_input", ""))
            offset = item.get("offset", 0.0)
            duration = item.get("duration", None)
            
            if not os.path.exists(audio_path):
                continue
            
            # offset/duration으로 오디오 구간 로드
            audio_data = load_audio_for_vllm(audio_path, offset=offset, duration=duration)
            if audio_data is None:
                continue
            
            # chat template 적용된 프롬프트 생성
            prompt_text = self._build_prompt(prompt, audio_path)
            
            valid_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"audio": audio_data},
            })
            valid_indices.append(i)
        
        # 결과 초기화
        results = [""] * len(items)
        
        # vLLM Gemma3n은 배치 내 오디오 feature 길이가 달라도 shape 검증에서 실패함.
        # 서로 다른 길이 오디오를 한 배치로 보내지 않도록 1건씩 generate 호출.
        if valid_inputs:
            for vi, idx in zip(valid_inputs, valid_indices):
                outputs = self.llm.generate([vi], sampling_params)
                if outputs and outputs[0].outputs:
                    results[idx] = outputs[0].outputs[0].text.strip()
        
        if return_first_logits:
            return (results, [None] * len(items))
        return results

    def get_next_token_logits(
        self, audio_path: str, text_input: str, answer_suffix: str = "\n답: "
    ) -> Optional[Dict[int, float]]:
        """SQA용: '답: ' 다음 1토큰 생성 시 vLLM logprobs 반환 (token_id -> logprob)."""
        if not os.path.exists(audio_path):
            return None
        audio_data = load_audio_for_vllm(audio_path)
        if audio_data is None:
            return None
        prompt_text = self._build_prompt((text_input or "").strip() + answer_suffix, audio_path)
        sampling_params = self.SamplingParams(max_tokens=1, temperature=0, logprobs=100)
        inputs = [{"prompt": prompt_text, "multi_modal_data": {"audio": audio_data}}]
        try:
            outputs = self.llm.generate(inputs, sampling_params)
        except Exception:
            return None
        if not outputs or not outputs[0].outputs:
            return None
        logprobs_pos0 = outputs[0].outputs[0].logprobs
        if not logprobs_pos0 or len(logprobs_pos0) == 0:
            return None
        # logprobs_pos0[0] = first generated token: dict[int, Logprob] or FlatLogprobs
        first = logprobs_pos0[0]
        if hasattr(first, "items"):
            return {tid: lp.logprob for tid, lp in first.items()}
        return None

    def get_next_token_logits_batch(
        self, items: List[Dict], answer_suffix: str = "\n답: "
    ) -> List[Optional[Dict[int, float]]]:
        """SQA용: 각 샘플에 대해 get_next_token_logits 호출 (오디오 1건씩 처리)."""
        out: List[Optional[Dict[int, float]]] = []
        for item in items:
            audio_path = item.get("audio_path", "")
            text_input = item.get("text_input", "")
            if not os.path.exists(audio_path):
                out.append(None)
                continue
            out.append(self.get_next_token_logits(audio_path, text_input, answer_suffix))
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
    batch_size: int = 8,  # deprecated: vLLM이 내부적으로 continuous batching 수행
    tensor_parallel_size: int = 1,
) -> Dict:
    """vLLM 로컬로 추론 실행 (vLLM continuous batching 사용)"""

    base_dir = Path(base_dir).resolve() if base_dir else None
    print(f"\n[Task: {task}] (vLLM Local - Continuous Batching)")
    print(f"입력 파일: {input_jsonl}")

    loader_kwargs = {"max_samples": max_samples, "custom_prompt": custom_prompt}
    if task == "instruct" and base_dir:
        loader_kwargs["base_dir"] = str(base_dir)
    dataloader = get_dataloader(task=task, jsonl_path=input_jsonl, **loader_kwargs)

    print(f"총 {len(dataloader)}개 샘플")

    # vLLM 모델 로딩
    model = Gemma3nVLLMModel(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_new_tokens_default=max_new_tokens,
    )

    start_time = time.time()
    results = []
    success_count = 0
    fail_count = 0

    # 데이터 준비 - 전체를 한 번에 처리 (vLLM이 내부적으로 continuous batching)
    all_items = []
    for item in dataloader:
        audio_path = item["audio_path"]
        if base_dir and not os.path.isabs(audio_path):
            item["audio_path"] = str(base_dir / audio_path)
        all_items.append(item)

    print(f"전체 {len(all_items)}개 샘플을 vLLM에 전달 (continuous batching)")

    if task == "sqa":
        # SQA: 전체 데이터 준비
        all_inputs = []
        all_meta = []
        missing_indices = set()
        
        for item in all_items:
            audio_path = item["audio_path"]
            index = item["index"]
            text_input = item.get("text_input", "")
            choices_ko = item.get("choices_ko", "")
            answer_ko = item.get("answer_ko", "")
            choice_letters = _parse_choice_letters(choices_ko)
            gt_letter = _normalize_gt_to_letter(answer_ko)

            if not os.path.exists(audio_path):
                results.append({
                    "index": index, "prediction": "",
                    "answer_ko": answer_ko,
                    "correct": False, "note": "audio_missing",
                })
                fail_count += 1
                missing_indices.add(index)
                continue

            prompt = (text_input or "").strip() + SQA_ANSWER_SUFFIX
            all_inputs.append({
                "audio_path": audio_path,
                "prompt": prompt,
            })
            all_meta.append((index, answer_ko, choice_letters, gt_letter))

        # vLLM이 전체를 한 번에 처리
        if all_inputs:
            print(f"vLLM 추론 시작: {len(all_inputs)}개 유효 샘플")
            generated_list = model.inference_batch(all_inputs, max_new_tokens)
            
            for i, meta in enumerate(all_meta):
                index, answer_ko, choice_letters, gt_letter = meta
                generated = generated_list[i]

                pred_letter = _parse_sqa_prediction_from_text(generated, choice_letters)
                is_correct = gt_letter is not None and pred_letter == gt_letter
                if is_correct:
                    success_count += 1
                else:
                    fail_count += 1

                results.append({
                    "index": index,
                    "prediction": f"({pred_letter})",
                    "answer_ko": answer_ko,
                    "correct": is_correct,
                })
    else:
        # ASR/Instruct/Translation: 전체 데이터 준비
        all_inputs = []
        all_meta = []
        missing_results = []
        
        for item in all_items:
            audio_path = item["audio_path"]
            index = item["index"]
            prompt = item.get("text_input", item.get("prompt", ""))

            if not os.path.exists(audio_path):
                print(f"경고: 오디오 파일 없음 - {audio_path}")
                missing_results.append({"index": index, "prediction": ""})
                fail_count += 1
                continue

            all_inputs.append({
                "audio_path": audio_path,
                "prompt": prompt,
            })
            all_meta.append(index)

        # vLLM이 전체를 한 번에 처리
        if all_inputs:
            print(f"vLLM 추론 시작: {len(all_inputs)}개 유효 샘플")
            generated_list = model.inference_batch(all_inputs, max_new_tokens)

            for i, index in enumerate(all_meta):
                generated = generated_list[i]
                results.append({"index": index, "prediction": generated})
                if generated:
                    success_count += 1
                else:
                    fail_count += 1
        
        # 누락된 결과 추가
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

    print("\n" + "=" * 60)
    print(f"추론 완료 [{task}] (vLLM Local)")
    print("=" * 60)
    print(f"총 샘플: {len(results)}")
    print(f"성공: {success_count}, 실패: {fail_count}")
    if task == "sqa":
        acc = success_count / len(results) * 100 if results else 0
        print(f"정확도: {acc:.2f}%")
    if results:
        print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/len(results):.2f}초/샘플)")
    print(f"출력 파일: {output_jsonl}")
    print("=" * 60)

    return stats


def main():
    available_tasks = list_available_tasks()

    parser = argparse.ArgumentParser(
        description='Gemma-3n vLLM 로컬 추론 (서버 불필요, continuous batching)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
지원 task: {', '.join(available_tasks)}

예시:
  python Gemma3n_vllm_inference.py -t asr -i ./input.jsonl -o ./output.jsonl -m google/gemma-3n-E2B-it -tp 1
  python Gemma3n_vllm_inference.py -t asr -i ./input.jsonl -o ./output.jsonl -m google/gemma-3n-E4B-it -tp 1

참고: vLLM이 내부적으로 continuous batching을 수행하므로 --batch-size는 무시됩니다.
        """,
    )
    parser.add_argument('--task', '-t', required=True, choices=available_tasks)
    parser.add_argument('--input', '-i', required=True, help='입력 JSONL 파일 경로')
    parser.add_argument('--output', '-o', required=True, help='출력 JSONL 파일 경로')
    parser.add_argument('--model', '-m', default='google/gemma-3n-E2B-it', help='모델 경로 (google/gemma-3n-E2B-it 또는 google/gemma-3n-E4B-it)')
    parser.add_argument('--base-dir', default=None, help='오디오 상대 경로 기준 디렉토리')
    parser.add_argument('--prompt', '-p', default=None, help='커스텀 프롬프트')
    parser.add_argument('--max-samples', type=int, default=None, help='최대 샘플 수')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='최대 생성 토큰 수')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='(deprecated) vLLM continuous batching 사용')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1, help='GPU 수')

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


if __name__ == '__main__':
    main()

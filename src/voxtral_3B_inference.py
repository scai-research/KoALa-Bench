#!/usr/bin/env python3
"""
Voxtral Mini 3B (mistralai/Voxtral-Mini-3B-2507) 전용 추론 스크립트

입력: JSONL 파일 (task별 dataloader 사용)
출력: JSONL 파일 (index, prediction [, task별 필드])

지원 task: asr, sqa, instruct

참고: Voxtral 공식 사용 예시는 Hugging Face 모델 카드 참조
  - `mistralai/Voxtral-Mini-3B-2507`
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
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
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
    """Voxtral generate 출력에서 선택지 글자 추출."""
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class VoxtralMini3BModel:
    """
    Voxtral Mini 3B 모델 래퍼 (Transformers 기반)

    - HF 모델 카드: mistralai/Voxtral-Mini-3B-2507
      참고: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507

    - Audio Instruct 예시를 기반으로 구현
      (conversation + processor.apply_chat_template → model.generate)
    """

    DEFAULT_MODEL_PATH = "mistralai/Voxtral-Mini-3B-2507"

    def __init__(self, model_path: str = None, device: str = "cuda", **kwargs):
        """
        Args:
            model_path: 모델 경로 (로컬 또는 HuggingFace). 비우면 DEFAULT_MODEL_PATH 사용.
            device: 디바이스 (cuda/cpu)
            **kwargs: tensor_parallel_size 등 호출부에서 넘기는 추가 인자 (무시)
        """
        from transformers import VoxtralForConditionalGeneration, AutoProcessor

        if not (model_path and model_path.strip()):
            model_path = self.DEFAULT_MODEL_PATH
        model_path = model_path.strip()

        print(f"Voxtral 모델 로딩 중: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)

        # bf16 권장 (모델 카드 참고), 여의치 않으면 float16/auto 사용
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        self.device = device
        print("Voxtral 모델 로딩 완료")

    @property
    def tokenizer(self):
        # 인터페이스 호환용 (다른 backend 들이 model.tokenizer 에 접근하는 경우 대비)
        return getattr(self.processor, "tokenizer", None)

    def _build_conversation(self, audio_path: str, prompt: str):
        """
        Voxtral Audio Instruct 형식의 대화 구성.
        HF 예시와 동일하게 content 안에 type=audio/path 및 type=text 사용.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": audio_path,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        return conversation

    def _get_audio_path_segment(
        self,
        audio_path: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        offset/duration 이 있으면 해당 구간만 잘라 임시 wav로 저장해 그 경로 반환.
        없으면 원본 경로 반환. 반환: (사용할_경로, 정리할_임시경로 또는 None)
        """
        if (offset is None or offset == 0.0) and (duration is None or duration <= 0):
            return (audio_path, None)
        try:
            import librosa
            import soundfile as sf
        except ImportError as e:
            print(f"[Voxtral] offset/duration 사용을 위해 librosa, soundfile 필요: {e}")
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
            print(f"[Voxtral] 오디오 구간 로드 실패 [{audio_path}]: {e}")
            return (audio_path, None)

    def _apply_chat_template(self, conversation_or_list, **kwargs):
        """
        apply_chat_template 호출. 멀티모달 processor는 문자열 또는 tensor dict를 반환할 수 있으므로
        tokenize=True, return_dict=True, return_tensors="pt" 로 항상 모델 입력용 dict 수신.
        MistralCommonTokenizer는 add_generation_prompt 미지원 → 실패 시 해당 인자 없이 재시도.
        """
        base_kwargs = dict(
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        base_kwargs.update(kwargs)

        try:
            return self.processor.apply_chat_template(
                conversation_or_list,
                add_generation_prompt=True,
                **base_kwargs,
            )
        except Exception as e:
            msg = str(e).lower()
            if "add_generation_prompt" in msg or "not supported" in msg:
                return self.processor.apply_chat_template(
                    conversation_or_list,
                    **base_kwargs,
                )
            raise

    def _prepare_inputs(self, conversation):
        """
        conversation → processor.apply_chat_template → 모델 입력 텐서.
        HF 모델 카드와 동일한 방식.
        """
        inputs = self._apply_chat_template(conversation)
        # dict of tensors 로 가정하고 모델 디바이스로 이동
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        return inputs

    def inference(
        self,
        audio_path: str,
        prompt: str,
        max_new_tokens: int = 512,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> str:
        """
        단일 오디오 추론 (ASR/Translation/Instruct 공통)

        Args:
            audio_path: 오디오 파일 경로 (로컬 경로)
            prompt: 텍스트 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            offset: 오디오 시작 시점(초). Translation 구간 번역용
            duration: 로드할 길이(초). None이면 끝까지

        Returns:
            모델 응답 텍스트
        """
        if not os.path.exists(audio_path):
            print(f"오디오 파일 없음: {audio_path}")
            return ""

        path_to_use, temp_path = self._get_audio_path_segment(audio_path, offset=offset, duration=duration)
        try:
            conversation = self._build_conversation(path_to_use, prompt or "")
            try:
                inputs = self._prepare_inputs(conversation)
            except Exception as e:
                print(f"Voxtral 입력 준비 실패 [{audio_path}]: {e}")
                return ""

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[:, input_len:]
            response = self.processor.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return response.strip()
        finally:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def inference_batch(
        self,
        items: List[Dict],
        max_new_tokens: int = 512,
        return_first_logits: bool = False,
    ):
        """
        배치 추론 — Voxtral chat_template 가 리스트 입력을 지원하므로
        한 번에 처리 (오디오 파일이 모두 로컬에 있다고 가정).

        Args:
            items: 각 항목은 audio_path, prompt
            return_first_logits: True면 (texts, logits_list) 반환. Voxtral은 logits 미지원 → (texts, [None]*n)
        Returns:
            생성 텍스트 리스트 또는 (리스트, logits 리스트)
        """
        if not items:
            return ([], []) if return_first_logits else []

        temp_paths: List[Optional[str]] = []
        conversations = []
        valid_indices = []
        for idx, it in enumerate(items):
            audio_path = it.get("audio_path", "")
            prompt = it.get("prompt", it.get("text_input", ""))
            offset = it.get("offset", 0.0)
            duration = it.get("duration")
            if not os.path.exists(audio_path):
                continue
            path_to_use, temp_path = self._get_audio_path_segment(audio_path, offset=offset, duration=duration)
            if temp_path:
                temp_paths.append(temp_path)
            conversations.append(self._build_conversation(path_to_use, prompt or ""))
            valid_indices.append(idx)

        results = [""] * len(items)
        if not conversations:
            for p in temp_paths:
                if p and os.path.isfile(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
            return (results, [None] * len(results)) if return_first_logits else results

        try:
            inputs = self._apply_chat_template(conversations)
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        except Exception as e:
            print(f"Voxtral 배치 입력 준비 실패: {e}")
            for p in temp_paths:
                if p and os.path.isfile(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
            for idx, it in enumerate(items):
                audio_path = it.get("audio_path", "")
                prompt = it.get("prompt", it.get("text_input", ""))
                offset = it.get("offset", 0.0)
                duration = it.get("duration")
                if os.path.exists(audio_path):
                    results[idx] = self.inference(
                        audio_path=audio_path,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        offset=offset,
                        duration=duration,
                    )
            return (results, [None] * len(results)) if return_first_logits else results

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            input_len = int(attention_mask[0].sum().item())
        else:
            input_len = inputs["input_ids"].shape[1]

        gen_ids = outputs[:, input_len:]
        decoded = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for i, idx in enumerate(valid_indices):
            results[idx] = decoded[i]

        for p in temp_paths:
            if p and os.path.isfile(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass

        if return_first_logits:
            return (results, [None] * len(results))
        return results

    def get_next_token_logits(
        self,
        audio_path: str,
        text_input: str,
        answer_suffix: str = "\n답: ",
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """
        SQA/LSQA 등에서 사용:
        audio + (text_input + answer_suffix) 까지 인코딩한 뒤,
        마지막 위치의 next-token logits 반환. (생성 없이 forward 1회)

        Returns:
            (1, vocab_size) float tensor 또는 실패 시 None
        """
        if not os.path.exists(audio_path):
            return None

        path_to_use, temp_path = self._get_audio_path_segment(
            audio_path, offset=offset, duration=duration
        )
        try:
            prompt = (text_input or "").strip() + (answer_suffix or "")
            conversation = self._build_conversation(path_to_use, prompt)
            inputs = self._prepare_inputs(conversation)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = getattr(outputs, "logits", None)
            if logits is None:
                return None

            # (batch=1, seq_len, vocab)
            attn = inputs.get("attention_mask", None)
            if attn is not None:
                pos = int(attn[0].sum().item()) - 1
                pos = max(pos, 0)
                next_logits = logits[0, pos : pos + 1, :].float()  # (1, vocab)
            else:
                next_logits = logits[:, -1, :].float()  # (1, vocab)
            return next_logits
        except Exception as e:
            print(f"[Voxtral] get_next_token_logits 실패 [{audio_path}]: {e}")
            return None
        finally:
            if temp_path and os.path.isfile(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def get_next_token_logits_batch(
        self,
        items: List[Dict],
        answer_suffix: str = "\n답: ",
    ) -> List[Optional[torch.Tensor]]:
        """
        배치 next-token logits:
        여러 샘플을 한 번의 forward로 처리하고, 각 샘플의 (1, vocab) logits를 반환.
        오디오 누락/오류 시 해당 샘플은 None.
        """
        if not items:
            return []

        out: List[Optional[torch.Tensor]] = [None] * len(items)

        temp_paths: List[Optional[str]] = []
        conversations = []
        valid_indices = []
        for idx, it in enumerate(items):
            audio_path = it.get("audio_path", "")
            if not os.path.exists(audio_path):
                continue
            text_input = it.get("text_input", "")
            offset = it.get("offset", 0.0)
            duration = it.get("duration", None)
            path_to_use, temp_path = self._get_audio_path_segment(
                audio_path, offset=offset, duration=duration
            )
            if temp_path:
                temp_paths.append(temp_path)
            prompt = (text_input or "").strip() + (answer_suffix or "")
            conversations.append(self._build_conversation(path_to_use, prompt))
            valid_indices.append(idx)

        if not conversations:
            for p in temp_paths:
                if p and os.path.isfile(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
            return out

        try:
            inputs = self._apply_chat_template(conversations)
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                return out

            attn = inputs.get("attention_mask", None)
            for bi, idx in enumerate(valid_indices):
                if attn is not None:
                    pos = int(attn[bi].sum().item()) - 1
                    pos = max(pos, 0)
                    out[idx] = logits[bi, pos : pos + 1, :].float()
                else:
                    out[idx] = logits[bi : bi + 1, -1, :].float()
            return out
        except Exception as e:
            print(f"[Voxtral] get_next_token_logits_batch 실패: {e}")
            # 폴백: 단건 호출
            for idx, it in enumerate(items):
                audio_path = it.get("audio_path", "")
                if not os.path.exists(audio_path):
                    continue
                out[idx] = self.get_next_token_logits(
                    audio_path=audio_path,
                    text_input=it.get("text_input", ""),
                    answer_suffix=answer_suffix,
                    offset=it.get("offset", 0.0),
                    duration=it.get("duration", None),
                )
            return out
        finally:
            for p in temp_paths:
                if p and os.path.isfile(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass


def run_inference(
    task: str,
    input_jsonl: str,
    output_jsonl: str,
    model_path: str,
    custom_prompt: str = None,
    max_samples: int = None,
    max_new_tokens: int = 256,
    base_dir: str = None,
    batch_size: int = 4,
) -> Dict:
    """
    Voxtral Mini 3B 배치 추론 실행 (asr / sqa / instruct 공통)

    - Qwen2-Audio 추론 스크립트와 동일한 인터페이스를 제공
    - SQA의 경우 generate 텍스트를 파싱하여 선택지 예측
    """
    base_dir = Path(base_dir).resolve() if base_dir else None
    print(f"\n[Task: {task}] (Voxtral Mini 3B)")
    print(f"입력 파일: {input_jsonl}")

    loader_kwargs = {"max_samples": max_samples, "custom_prompt": custom_prompt}
    if task == "instruct" and base_dir:
        loader_kwargs["base_dir"] = str(base_dir)
    dataloader = get_dataloader(task=task, jsonl_path=input_jsonl, **loader_kwargs)

    all_items = list(dataloader)
    total_samples = len(all_items)
    print(f"총 {total_samples}개 샘플, 배치 크기: {batch_size}")

    model = VoxtralMini3BModel(model_path)

    start_time = time.time()
    results: List[Dict] = []
    success_count = 0
    fail_count = 0

    num_batches = (total_samples + batch_size - 1) // batch_size if batch_size > 0 else 1

    for batch_idx in tqdm(range(num_batches), desc=f"Voxtral 배치 추론 (bs={batch_size})"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        batch_items = all_items[batch_start:batch_end]

        processed_items = []
        for item in batch_items:
            audio_path = item["audio_path"]
            if base_dir and not os.path.isabs(audio_path):
                audio_path = str(base_dir / audio_path)
            it = dict(item)
            it["audio_path"] = audio_path
            processed_items.append(it)

        if task == "sqa":
            # SQA: Voxtral은 logits API가 없어 텍스트 generate 기반으로 선택지 예측
            inference_items = [
                {
                    "audio_path": it["audio_path"],
                    "prompt": (it.get("text_input", "") or "").strip() + SQA_ANSWER_SUFFIX,
                }
                for it in processed_items
            ]
            generated_list = model.inference_batch(inference_items, max_new_tokens)

            for it, generated in zip(processed_items, generated_list):
                index = it["index"]
                choices_ko = it.get("choices_ko", "")
                answer_ko = it.get("answer_ko", "")
                choice_letters = _parse_choice_letters(choices_ko)
                gt_letter = _normalize_gt_to_letter(answer_ko)

                pred_letter = _parse_sqa_prediction_from_text(generated, choice_letters)
                is_correct = gt_letter is not None and pred_letter == gt_letter
                if is_correct:
                    success_count += 1
                else:
                    fail_count += 1

                results.append(
                    {
                        "index": index,
                        "prediction": f"({pred_letter})",
                        "answer_ko": answer_ko,
                        "correct": is_correct,
                    }
                )
        else:
            # ASR / Instruct
            inference_items = [
                {
                    "audio_path": it["audio_path"],
                    "prompt": it.get("text_input", it.get("prompt", "")),
                }
                for it in processed_items
            ]
            predictions = model.inference_batch(inference_items, max_new_tokens)

            for it, prediction in zip(processed_items, predictions):
                index = it["index"]
                if prediction:
                    success_count += 1
                else:
                    fail_count += 1
                results.append({"index": index, "prediction": prediction})

    elapsed_time = time.time() - start_time

    # 결과 저장
    save_jsonl(results, output_jsonl)

    # 통계
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
    print(f"추론 완료 [{task}] (Voxtral Mini 3B)")
    print("=" * 60)
    print(f"총 샘플: {len(results)}")
    print(f"성공: {success_count}, 실패: {fail_count}")
    if task == "sqa" and results:
        acc = success_count / len(results) * 100
        print(f"정확도: {acc:.2f}%")
    if results:
        print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/len(results):.2f}초/샘플)")
    print(f"출력 파일: {output_jsonl}")
    print("=" * 60)

    return stats


def main():
    available_tasks = list_available_tasks()

    parser = argparse.ArgumentParser(
        description="Voxtral Mini 3B 추론 (task별 dataloader)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
지원 task: {', '.join(available_tasks)}

예시:
  # ASR
  python voxtral_3B_inference.py -t asr -i ./ASR/output/test.jsonl -o ./ASR/results/voxtral_out.jsonl -m mistralai/Voxtral-Mini-3B-2507

  # SQA
  python voxtral_3B_inference.py -t sqa -i ./SQA/click_final.jsonl -o ./SQA/results/voxtral_out.jsonl -m mistralai/Voxtral-Mini-3B-2507 --base-dir /path/to/audio/root

  # Instruct
  python voxtral_3B_inference.py -t instruct -i ./Instruct/kudge.jsonl -o ./Instruct/voxtral_out.jsonl -m mistralai/Voxtral-Mini-3B-2507
        """,
    )

    parser.add_argument(
        "--task",
        "-t",
        required=True,
        choices=available_tasks,
        help=f"Task 이름 ({', '.join(available_tasks)})",
    )
    parser.add_argument("--input", "-i", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("--output", "-o", required=True, help="출력 JSONL 파일 경로")
    parser.add_argument(
        "--model",
        "-m",
        default="mistralai/Voxtral-Mini-3B-2507",
        help="모델 경로 (로컬 또는 HuggingFace)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="오디오 상대 경로 기준 디렉토리 (sqa/instruct 등)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=None,
        help="커스텀 프롬프트 (미지정 시 dataloader 기본값)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="최대 샘플 수 (테스트용)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="최대 생성 토큰 수",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="배치 크기 (기본: 4)",
    )

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
    )


if __name__ == "__main__":
    main()


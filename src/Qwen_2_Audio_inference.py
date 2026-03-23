#!/usr/bin/env python3
"""
Qwen2-Audio 전용 추론 스크립트

입력: JSONL 파일 (task별 dataloader 사용)
출력: JSONL 파일 (index, prediction [, task별 필드])

지원 task: asr, sqa, instruct
다른 모델(Llama 등)은 모델마다 동일 인터페이스로 별도 스크립트를 만들어 사용하면 됨.
"""

import os
import sys
import re
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
import librosa

# dataloaders 모듈 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders import get_dataloader, list_available_tasks


# --- SQA용 헬퍼 (run_inference task=sqa 에서 사용) ---
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


def _predict_choice_from_logits(processor, next_logits: torch.Tensor, choice_letters: List[str]) -> str:
    tokenizer = processor.tokenizer
    candidates = []
    for L in choice_letters:
        for s in [L, f"({L})", f" ({L})", f" {L}"]:
            tok = tokenizer.encode(s, add_special_tokens=False)
            if len(tok) == 1:
                candidates.append((L, tok[0]))
                break
        else:
            tok = tokenizer.encode(L, add_special_tokens=False)
            if tok:
                candidates.append((L, tok[0]))
    if not candidates:
        return choice_letters[0]
    logprobs = F.log_softmax(next_logits[0], dim=-1)
    best_letter, best_lp = choice_letters[0], -1e9
    for L, tid in candidates:
        lp = logprobs[tid].item()
        if lp > best_lp:
            best_lp, best_letter = lp, L
    return best_letter


def save_jsonl(data: List[Dict], output_path: str):
    """JSONL 파일 저장"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


class Qwen2AudioModel:
    """Qwen2-Audio 모델 래퍼"""
    
    DEFAULT_MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"

    def __init__(self, model_path: str = None, device: str = "cuda", **kwargs):
        """
        Args:
            model_path: 모델 경로 (로컬 또는 HuggingFace). 비우면 DEFAULT_MODEL_PATH 사용.
            device: 디바이스 (cuda/cpu)
            **kwargs: tensor_parallel_size 등 호출부에서 넘기는 추가 인자 (무시)
        """
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

        if not (model_path and model_path.strip()):
            model_path = self.DEFAULT_MODEL_PATH
        model_path = model_path.strip()

        print(f"모델 로딩 중: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path)
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
            print("flash_attn 미설치 → attn_implementation=sdpa 사용")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation=attn_impl,
        )
        self.model.eval()
        self.device = device
        self.target_sr = self.processor.feature_extractor.sampling_rate
        print(f"모델 로딩 완료 (샘플레이트: {self.target_sr}Hz)")
    
    def load_audio(
        self,
        audio_path: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """오디오 파일 로드 (Translation 등에서 offset/duration 지원)"""
        try:
            kwargs = {"sr": self.target_sr, "offset": offset}
            if duration is not None:
                kwargs["duration"] = duration
            audio, sr = librosa.load(audio_path, **kwargs)
            return audio
        except Exception as e:
            print(f"오디오 로드 실패 [{audio_path}]: {e}")
            return None

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
            audio_path: 오디오 파일 경로
            prompt: 텍스트 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            offset: 오디오 시작 시점(초). Translation 구간 번역용
            duration: 로드할 길이(초). None이면 끝까지
        
        Returns:
            모델 응답 텍스트
        """
        audio = self.load_audio(audio_path, offset=offset, duration=duration)
        if audio is None:
            return ""
        
        # 대화 구성
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]}
        ]
        
        # 텍스트 템플릿 적용
        text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # 입력 처리 (Qwen2AudioProcessor: audio 단수형 사용!)
        inputs = self.processor(
            text=text, 
            audio=[audio],  # audios가 아닌 audio (단수형)!
            return_tensors="pt", 
            padding=True
        )
        
        # GPU로 이동
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # 입력 부분 제거
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        
        # 디코딩
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()

    def inference_batch(
        self,
        items: List[Dict],
        max_new_tokens: int = 256,
        return_first_logits: bool = False,
    ):
        """
        배치 추론 (ASR/Translation/Instruct 공통). for 문 없이 한 배치씩 processor + generate 1회.

        Args:
            items: 각 항목은 audio_path, prompt, (선택) offset, duration
            return_first_logits: True면 첫 토큰의 logits도 반환 (SQA용)
        Returns:
            return_first_logits=False: 각 샘플에 대한 생성 텍스트 리스트
            return_first_logits=True: (생성 텍스트 리스트, 첫 토큰 logits 리스트) 튜플
        """
        if not items:
            if return_first_logits:
                return [], []
            return []
        from concurrent.futures import ThreadPoolExecutor
        
        def load_audio_item(args):
            j, it = args
            audio = self.load_audio(
                it["audio_path"],
                offset=it.get("offset", 0.0),
                duration=it.get("duration"),
            )
            return j, audio
        
        # 병렬로 오디오 로딩
        with ThreadPoolExecutor(max_workers=min(32, len(items))) as executor:
            results = list(executor.map(load_audio_item, enumerate(items)))
        
        audios = []
        valid_indices = []
        for j, audio in results:
            if audio is not None:
                audios.append(audio)
                valid_indices.append(j)
        if not audios:
            if return_first_logits:
                return [""] * len(items), [None] * len(items)
            return [""] * len(items)
        conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": items[j]["audio_path"]},
                    {"type": "text", "text": items[j].get("prompt", "")},
                ]},
            ]
            for j in valid_indices
        ]
        texts = [
            self.processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
            for c in conversations
        ]
        inputs = self.processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=return_first_logits,
                return_dict_in_generate=return_first_logits,
            )
        
        # generation 결과 추출
        if return_first_logits:
            if hasattr(gen_out, "sequences"):
                generate_ids = gen_out.sequences
                first_token_scores = gen_out.scores[0] if hasattr(gen_out, "scores") and gen_out.scores else None
            else:
                generate_ids = gen_out
                first_token_scores = None
        else:
            generate_ids = gen_out
            first_token_scores = None
        
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        responses = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        out = [""] * len(items)
        for idx, resp in zip(valid_indices, responses):
            out[idx] = resp.strip()
        
        if return_first_logits:
            logits_out = [None] * len(items)
            if first_token_scores is not None:
                for i, idx in enumerate(valid_indices):
                    logits_out[idx] = first_token_scores[i].unsqueeze(0).float()  # (1, vocab_size)
            return out, logits_out
        return out

    def get_next_token_logits(
        self, audio_path: str, text_input: str, answer_suffix: str = "\n답: "
    ) -> Optional[torch.Tensor]:
        """
        SQA 등에서 사용: audio + text_input + answer_suffix 까지 인코딩한 뒤,
        마지막 위치의 next-token logits 반환. (생성 없이 forward 1회)
        """
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
        prompt = (text_input or "").strip() + answer_suffix
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, audio=[audio], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        next_logits = logits[:, -1, :].float()
        return next_logits

    def get_next_token_logits_batch(
        self, items: List[Dict], answer_suffix: str = "\n답: "
    ) -> List[Optional[torch.Tensor]]:
        """
        SQA 배치 처리: 여러 샘플의 next-token logits를 한 번의 forward로 반환.

        Args:
            items: 각 항목은 audio_path, text_input
        Returns:
            각 샘플의 next_logits 리스트 (로드 실패 시 None)
        """
        if not items:
            return []
        from concurrent.futures import ThreadPoolExecutor
        
        def load_audio_item(args):
            j, it = args
            audio = self.load_audio(it["audio_path"])
            return j, audio
        
        # 병렬로 오디오 로딩
        with ThreadPoolExecutor(max_workers=min(32, len(items))) as executor:
            results = list(executor.map(load_audio_item, enumerate(items)))
        
        audios = []
        valid_indices = []
        for j, audio in results:
            if audio is not None:
                audios.append(audio)
                valid_indices.append(j)
        if not audios:
            return [None] * len(items)
        conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": items[j]["audio_path"]},
                    {"type": "text", "text": (items[j].get("text_input", "") or "").strip() + answer_suffix},
                ]},
            ]
            for j in valid_indices
        ]
        texts = [
            self.processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
            for c in conversations
        ]
        inputs = self.processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab)
        next_logits_batch = logits[:, -1, :].float()  # (batch, vocab)
        out: List[Optional[torch.Tensor]] = [None] * len(items)
        for i, j in enumerate(valid_indices):
            out[j] = next_logits_batch[i:i+1]  # keep dim (1, vocab)
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
    batch_size: int = 4,
) -> Dict:
    """
    배치 추론 실행 (asr / sqa / instruct 공통) - 진짜 배치 처리 (for 문 없이)
    
    Args:
        task: task 이름 ('asr', 'sqa', 'instruct')
        input_jsonl: 입력 JSONL 파일
        output_jsonl: 출력 JSONL 파일
        model_path: 모델 경로
        custom_prompt: 커스텀 프롬프트 (None이면 dataloader 기본값)
        max_samples: 최대 샘플 수
        max_new_tokens: 최대 생성 토큰 수
        base_dir: 오디오 경로가 상대일 때 기준 디렉토리 (sqa/instruct 등)
        batch_size: 배치 크기 (GPU 메모리에 맞게 조절)
    
    Returns:
        실행 통계
    """
    base_dir = Path(base_dir).resolve() if base_dir else None
    print(f"\n[Task: {task}]")
    print(f"입력 파일: {input_jsonl}")
    
    loader_kwargs = {"max_samples": max_samples, "custom_prompt": custom_prompt}
    if task == "instruct" and base_dir:
        loader_kwargs["base_dir"] = str(base_dir)
    dataloader = get_dataloader(task=task, jsonl_path=input_jsonl, **loader_kwargs)
    
    # dataloader를 리스트로 변환 (배치 슬라이싱을 위해)
    all_items = list(dataloader)
    total_samples = len(all_items)
    print(f"총 {total_samples}개 샘플, 배치 크기: {batch_size}")
    
    model = Qwen2AudioModel(model_path)
    
    start_time = time.time()
    results = []
    success_count = 0
    fail_count = 0
    
    # 배치 단위로 처리 (tqdm은 배치 수 기준)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"배치 추론 (bs={batch_size})"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_samples)
        batch_items = all_items[batch_start:batch_end]
        
        # 경로 전처리 및 존재 확인
        processed_items = []
        for item in batch_items:
            audio_path = item["audio_path"]
            if base_dir and not os.path.isabs(audio_path):
                audio_path = str(base_dir / audio_path)
            item = dict(item)  # 원본 수정 방지
            item["audio_path"] = audio_path
            processed_items.append(item)
        
        if task == "sqa":
            # SQA 배치 처리
            sqa_batch_items = [
                {"audio_path": it["audio_path"], "text_input": it.get("text_input", "")}
                for it in processed_items
            ]
            logits_list = model.get_next_token_logits_batch(sqa_batch_items, SQA_ANSWER_SUFFIX)
            
            for it, next_logits in zip(processed_items, logits_list):
                index = it["index"]
                choices_ko = it.get("choices_ko", "")
                answer_ko = it.get("answer_ko", "")
                choice_letters = _parse_choice_letters(choices_ko)
                gt_letter = _normalize_gt_to_letter(answer_ko)
                
                if next_logits is None:
                    pred_letter = choice_letters[0]
                    results.append({
                        "index": index,
                        "prediction": f"({pred_letter})",
                        "answer_ko": answer_ko,
                        "correct": False,
                        "note": "audio_load_failed",
                    })
                    fail_count += 1
                else:
                    pred_letter = _predict_choice_from_logits(model.processor, next_logits, choice_letters)
                    is_correct = gt_letter is not None and pred_letter == gt_letter
                    if is_correct:
                        success_count += 1
                    results.append({
                        "index": index,
                        "prediction": f"({pred_letter})",
                        "answer_ko": answer_ko,
                        "correct": is_correct,
                    })
        else:
            # ASR / Instruct 배치 처리
            inference_items = [
                {
                    "audio_path": it["audio_path"],
                    "prompt": it.get("text_input", it.get("prompt", "")),
                    "offset": it.get("offset", 0.0),
                    "duration": it.get("duration"),
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
        "timestamp": datetime.now().isoformat()
    }
    
    # 결과 출력
    print("\n" + "="*60)
    print(f"추론 완료 [{task}]")
    print("="*60)
    print(f"총 샘플: {len(results)}")
    print(f"성공: {success_count}, 실패: {fail_count}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/len(results):.2f}초/샘플)")
    print(f"출력 파일: {output_jsonl}")
    print("="*60)
    
    return stats


def main():
    available_tasks = list_available_tasks()
    
    parser = argparse.ArgumentParser(
        description='Qwen2-Audio 추론 (task별 dataloader)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
지원 task: {', '.join(available_tasks)}

예시:
  # 기본 배치 처리 (batch_size=4)
  python Qwen_2_Audio_inference.py -t asr -i ./ASR/output/clovacall_test.jsonl -o ./ASR/results/out.jsonl -m /path/to/Qwen2-Audio-7B

  # 배치 크기 조절 (GPU 메모리에 맞게)
  python Qwen_2_Audio_inference.py -t asr -i ./ASR/output/clovacall_test.jsonl -o ./ASR/results/out.jsonl -m /path/to/Qwen2-Audio-7B --batch-size 8

  # SQA 배치 처리
  python Qwen_2_Audio_inference.py -t sqa -i ./SQA/click_final.jsonl -o ./SQA/results/out.jsonl -m /path/to/model --base-dir /path/to/audio/root -b 4

  # Instruct 배치 처리
  python Qwen_2_Audio_inference.py -t instruct -i ./Instruct/kudge_pairwise.jsonl -o ./Instruct/out.jsonl -m /path/to/model -b 4
        """
    )
    
    parser.add_argument('--task', '-t', required=True, choices=available_tasks,
                        help=f'Task 이름 ({", ".join(available_tasks)})')
    parser.add_argument('--input', '-i', required=True, help='입력 JSONL 파일 경로')
    parser.add_argument('--output', '-o', required=True, help='출력 JSONL 파일 경로')
    parser.add_argument('--model', '-m', required=True, help='모델 경로 (로컬 또는 HuggingFace)')
    parser.add_argument('--base-dir', default=None,
                        help='오디오 상대 경로 기준 디렉토리 (sqa/instruct 등)')
    parser.add_argument('--prompt', '-p', default=None,
                        help='커스텀 프롬프트 (미지정 시 dataloader 기본값)')
    parser.add_argument('--max-samples', type=int, default=None, help='최대 샘플 수 (테스트용)')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='최대 생성 토큰 수')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='배치 크기 (기본: 4)')
    
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


if __name__ == '__main__':
    main()

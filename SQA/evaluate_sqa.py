#!/usr/bin/env python3
"""
SQA (Spoken QA) 평가 스크립트 — Logit 기반 (LM harness 스타일)

- 음성 + 텍스트(prompt + choices_ko) 입력 후, "답: " 다음 토큰의 logprob로 (A)/(B)/(C) 선택
- 생성 문장 파싱 없이 선택지 토큰 확률만 사용 → "a person..." 같은 이상 출력 영향 없음
- 모델 무관: backends.get_backend(backend_name, model_path) 로 Qwen/Llama 등 동일 스크립트 사용
"""

import os
import sys
import json
import re
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Ko-Speech-Eval/src 기준
_KO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_KO_SRC) not in sys.path:
    sys.path.insert(0, str(_KO_SRC))
from dataloaders import get_dataloader
from backends import get_backend, list_backends


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_choice_letters(choices_ko: str) -> List[str]:
    """choices_ko에서 선택지 문자만 추출. 예: '(A) ...\\n(B) ...' -> ['A', 'B', 'C']"""
    if not (choices_ko or "").strip():
        return []
    letters = []
    for m in re.finditer(r"\(([A-Z])\)", (choices_ko or "").strip(), re.IGNORECASE):
        letters.append(m.group(1).upper())
    # 순서 유지, 중복 제거
    seen = set()
    out = []
    for L in letters:
        if L not in seen:
            seen.add(L)
            out.append(L)
    return out if out else ["A", "B", "C"]


def normalize_gt_to_letter(answer_ko: str) -> Optional[str]:
    """answer_ko를 선택지 문자 하나로 정규화. (A), (B), (A) 아니요 등 -> A, B, A"""
    raw = (answer_ko or "").strip()
    if not raw:
        return None
    m = re.match(r"^\(([A-Z])\)", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if raw.upper() in ("A", "B", "C", "D", "E"):
        return raw.upper()
    return None


def parse_choice_from_generation(generated_text: str, choice_letters: List[str]) -> str:
    """
    생성 텍스트에서 선택지 문자 추출. (A), (B), A, B 등 첫 매칭 반환.
    """
    if not (generated_text or "").strip():
        return choice_letters[0] if choice_letters else "A"
    text = (generated_text or "").strip()
    # (A), (B), (C) 순서로 검사
    for L in choice_letters:
        if f"({L})" in text or f"({L.lower()})" in text:
            return L
    # 단일 문자 (줄/단어 경계 우선)
    for L in choice_letters:
        if re.search(rf"\b{L}\b", text, re.IGNORECASE):
            return L.upper()
    for L in choice_letters:
        if L in text.upper():
            return L.upper()
    return choice_letters[0] if choice_letters else "A"


# --- Logit 기반 예측 ---

ANSWER_SUFFIX_KO = "\n답: "  # 한국어 답 유도 구절
ANSWER_SUFFIX_EN = "\nAnswer: "  # 영어 답 유도 구절


def predict_choice_from_logits(processor, next_logits: torch.Tensor, choice_letters: List[str]) -> str:
    """
    next_logits에서 choice_letters에 해당하는 토큰 id의 logprob를 비교해 가장 높은 선택지 반환.
    "(A)", "A", " A" 등 여러 표현 시도.
    """
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    # 후보 문자열: A, B, C / (A), (B) / 공백+문자 등
    candidates = []
    for L in choice_letters:
        for s in [L, f"({L})", f" ({L})", f" {L}"]:
            tok = tokenizer.encode(s, add_special_tokens=False)
            if len(tok) == 1:
                candidates.append((L, tok[0]))
                break
        else:
            # 여러 토큰이면 첫 토큰만 사용
            tok = tokenizer.encode(L, add_special_tokens=False)
            if tok:
                candidates.append((L, tok[0]))

    if not candidates:
        return choice_letters[0]  # fallback

    logprobs = F.log_softmax(next_logits[0], dim=-1)
    best_letter = choice_letters[0]
    best_lp = -1e9
    for L, tid in candidates:
        lp = logprobs[tid].item()
        if lp > best_lp:
            best_lp = lp
            best_letter = L
    return best_letter


def predict_choice_from_logprobs_dict(
    processor,
    token_id_to_logprob: Dict[int, float],
    choice_letters: List[str],
) -> str:
    """
    vLLM 등에서 반환하는 token_id -> logprob 딕셔너리로 선택지 예측.
    choice_letters에 해당하는 토큰 id 중 logprob이 가장 높은 문자 반환.
    """
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
    best_letter = choice_letters[0]
    best_lp = -1e9
    for L, tid in candidates:
        if tid in token_id_to_logprob:
            lp = token_id_to_logprob[tid]
            if lp > best_lp:
                best_lp = lp
                best_letter = L
    return best_letter


def evaluate_sqa(
    jsonl_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    backend_name: str = "qwen",
    max_samples: Optional[int] = None,
    base_dir: Optional[str] = None,
    save_generation: bool = False,
    max_new_tokens: int = 64,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    answer_suffix: str = ANSWER_SUFFIX_KO,
    prompt_prefix: Optional[str] = None,
    model=None,
) -> Dict:
    """
    SQA 평가: logit 기반 선택지 예측 후 정답률 계산. (모델 무관, backends 사용)

    Args:
        jsonl_path: SQA final JSONL 경로 (또는 base_dir 기준 상대 경로)
        output_dir: 결과 JSONL 및 요약 저장 디렉토리
        model_path: 모델 경로
        backend_name: 추론 백엔드 이름 (backends에 등록된 이름, 기본 'qwen')
        max_samples: 최대 샘플 수 (None이면 전체)
        base_dir: 오디오 raw 경로의 기준 디렉토리 (None이면 SQA 폴더 부모)
        save_generation: True면 모델 생성 문장도 결과에 저장 (generation 필드)
        max_new_tokens: save_generation 시 생성 최대 토큰 수

    Returns:
        평가 결과 딕셔너리 (accuracy, total, correct, ...)
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_absolute():
        jsonl_path = Path.cwd() / jsonl_path
    jsonl_path = jsonl_path.resolve()
    # 오디오 raw 경로 기준: JSONL이 SQA/ 안에 있으면 프로젝트 루트 = SQA의 부모
    base_dir = Path(base_dir or jsonl_path.parent.parent)
    base_dir = base_dir.resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader_kw = {"max_samples": max_samples}
    if prompt_prefix:
        loader_kw["custom_prompt"] = prompt_prefix
    dataloader = get_dataloader("sqa", str(jsonl_path), **loader_kw)
    items = list(dataloader)

    if model is None:
        model = get_backend(backend_name, model_path, tensor_parallel_size=tensor_parallel_size)
    use_batch = batch_size > 1 and hasattr(model, "inference_batch")
    if use_batch:
        print(f"배치 추론 사용 (batch_size={batch_size})")

    result_by_index = {}
    correct = 0
    correct_generation = 0
    invalid_gt = 0
    start_time = time.time()

    if use_batch:
        valid_list = []
        for item in items:
            index = item["index"]
            audio_path = item["audio_path"]
            if not os.path.isabs(audio_path):
                audio_path = str(base_dir / audio_path)
            if not os.path.exists(audio_path):
                r = {"index": index, "prediction": "", "prediction_gen": "", "answer_ko": item["answer_ko"], "correct": False, "correct_generation": False, "note": "audio_missing"}
                if save_generation:
                    r["generation"] = ""
                result_by_index[index] = r
                continue
            valid_list.append({
                "index": index,
                "audio_path": audio_path,
                "text_input": item["text_input"],
                "choices_ko": item["choices_ko"],
                "answer_ko": item["answer_ko"],
            })
        num_batches = (len(valid_list) + batch_size - 1) // batch_size
        use_inference_batch = hasattr(model, "inference_batch")
        for batch_idx in tqdm(range(num_batches), desc=f"SQA 배치 (bs={batch_size})"):
            batch = valid_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # 한 번의 inference_batch 호출로 generation과 logits 모두 획득
            if use_inference_batch:
                gen_inputs = [
                    {"audio_path": it["audio_path"], "prompt": (it["text_input"] or "").strip() + answer_suffix}
                    for it in batch
                ]
                gen_texts, logits_list = model.inference_batch(gen_inputs, max_new_tokens=max_new_tokens, return_first_logits=True)
            else:
                # fallback: 개별 처리
                batch_inputs = [{"audio_path": it["audio_path"], "text_input": it["text_input"]} for it in batch]
                logits_list = model.get_next_token_logits_batch(batch_inputs, answer_suffix) if hasattr(model, "get_next_token_logits_batch") else [None] * len(batch)
                gen_texts = [
                    model.inference(it["audio_path"], (it["text_input"] or "").strip() + answer_suffix, max_new_tokens=max_new_tokens)
                    if hasattr(model, "inference") else ""
                    for it in batch
                ]
            
            for it, next_logits, gen_text in zip(batch, logits_list, gen_texts):
                index = it["index"]
                choice_letters = parse_choice_letters(it["choices_ko"])
                gt_letter = normalize_gt_to_letter(it["answer_ko"])
                if gt_letter is None:
                    invalid_gt += 1
                if next_logits is None:
                    pred_letter = choice_letters[0]
                    r = {"index": index, "prediction": f"({pred_letter})", "answer_ko": it["answer_ko"], "correct": False, "note": "audio_load_failed"}
                elif isinstance(next_logits, dict):
                    pred_letter = predict_choice_from_logprobs_dict(model.processor, next_logits, choice_letters)
                    is_correct = gt_letter is not None and pred_letter == gt_letter
                    if is_correct:
                        correct += 1
                    r = {"index": index, "prediction": f"({pred_letter})", "answer_ko": it["answer_ko"], "correct": is_correct}
                else:
                    pred_letter = predict_choice_from_logits(model.processor, next_logits, choice_letters)
                    is_correct = gt_letter is not None and pred_letter == gt_letter
                    if is_correct:
                        correct += 1
                    r = {"index": index, "prediction": f"({pred_letter})", "answer_ko": it["answer_ko"], "correct": is_correct}
                pred_letter_gen = parse_choice_from_generation(gen_text or "", choice_letters)
                is_correct_gen = gt_letter is not None and pred_letter_gen == gt_letter
                if is_correct_gen:
                    correct_generation += 1
                r["prediction_gen"] = f"({pred_letter_gen})"
                r["correct_generation"] = is_correct_gen
                if save_generation:
                    r["generation"] = gen_text or ""
                # logit 없이 생성만 있는 백엔드(qwen3_vllm 등): correct를 생성 기준으로 통일
                if next_logits is None and (gen_text or "").strip():
                    r["correct"] = r["correct_generation"]
                    if r["correct"]:
                        correct += 1
                    r.pop("note", None)
                result_by_index[index] = r
        results = [result_by_index[item["index"]] for item in items]
    else:
        results = []
        for item in tqdm(items, desc="SQA 평가"):
            index = item["index"]
            audio_path = item["audio_path"]
            text_input = item["text_input"]
            choices_ko = item["choices_ko"]
            answer_ko = item["answer_ko"]
            if not os.path.isabs(audio_path):
                audio_path = str(base_dir / audio_path)
            if not os.path.exists(audio_path):
                r = {"index": index, "prediction": "", "prediction_gen": "", "answer_ko": answer_ko, "correct": False, "correct_generation": False, "note": "audio_missing"}
                if save_generation:
                    r["generation"] = ""
                results.append(r)
                continue
            choice_letters = parse_choice_letters(choices_ko)
            gt_letter = normalize_gt_to_letter(answer_ko)
            if gt_letter is None:
                invalid_gt += 1
            next_logits = model.get_next_token_logits(audio_path, text_input, answer_suffix=answer_suffix)
            if next_logits is None:
                pred_letter = choice_letters[0]
                r = {"index": index, "prediction": f"({pred_letter})", "answer_ko": answer_ko, "correct": False, "note": "audio_load_failed"}
            elif isinstance(next_logits, dict):
                pred_letter = predict_choice_from_logprobs_dict(model.processor, next_logits, choice_letters)
                is_correct = gt_letter is not None and pred_letter == gt_letter
                if is_correct:
                    correct += 1
                r = {"index": index, "prediction": f"({pred_letter})", "answer_ko": answer_ko, "correct": is_correct}
            else:
                pred_letter = predict_choice_from_logits(model.processor, next_logits, choice_letters)
                is_correct = (gt_letter is not None and pred_letter == gt_letter)
                if is_correct:
                    correct += 1
                r = {"index": index, "prediction": f"({pred_letter})", "answer_ko": answer_ko, "correct": is_correct}
            prompt_with_suffix = (text_input or "").strip() + answer_suffix
            gen_text = model.inference(audio_path, prompt_with_suffix, max_new_tokens=max_new_tokens) if hasattr(model, "inference") else ""
            pred_letter_gen = parse_choice_from_generation(gen_text or "", choice_letters)
            is_correct_gen = gt_letter is not None and pred_letter_gen == gt_letter
            if is_correct_gen:
                correct_generation += 1
            r["prediction_gen"] = f"({pred_letter_gen})"
            r["correct_generation"] = is_correct_gen
            if save_generation:
                r["generation"] = gen_text or ""
            # logit 없이 생성만 있는 백엔드(qwen3_vllm 등): correct를 생성 기준으로 통일
            if next_logits is None and (gen_text or "").strip():
                r["correct"] = r["correct_generation"]
                if r["correct"]:
                    correct += 1
                r.pop("note", None)
            results.append(r)

    elapsed = time.time() - start_time
    total = len(results)
    acc_logit = correct / total if total else 0.0
    acc_generation = correct_generation / total if total else 0.0

    # 결과 JSONL 저장
    out_jsonl = output_dir / "sqa_predictions.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "accuracy_logit": acc_logit,
        "accuracy_generation": acc_generation,
        "correct_logit": correct,
        "correct_generation": correct_generation,
        "total": total,
        "invalid_gt_count": invalid_gt,
        "elapsed_seconds": elapsed,
        "predictions_file": str(out_jsonl),
    }
    with open(output_dir / "sqa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SQA 평가 결과")
    print("=" * 60)
    print(f"정확도 (로짓 기반): {acc_logit:.4f} ({correct}/{total})")
    print(f"정확도 (생성 기반): {acc_generation:.4f} ({correct_generation}/{total})")
    print(f"GT 파싱 불가: {invalid_gt}건")
    print(f"소요 시간: {elapsed:.1f}초")
    print(f"결과 파일: {out_jsonl}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="SQA 평가 (logit 기반 선택지 예측, 모델 무관)",
        epilog=f"사용 가능 백엔드: {', '.join(list_backends())}",
    )
    parser.add_argument("--jsonl", type=str, required=True, help="SQA JSONL 경로")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="결과 저장 디렉토리")
    parser.add_argument("--model_path", type=str, default=None,
                    help="비우면 백엔드별 기본 모델 사용 (HuggingFace cache 또는 다운로드)")
    parser.add_argument("--backend", type=str, default="qwen", choices=list_backends(),
                        help="추론 백엔드 (모델별로 backends.py에 등록된 이름)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--base_dir", type=str, default=None, help="오디오 raw 경로 기준 디렉토리")
    parser.add_argument("--save_generation", action="store_true", help="모델 생성 문장도 JSONL에 저장 (generation 필드)")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="--save_generation 시 생성 최대 토큰 수")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기 (백엔드가 get_next_token_logits_batch 지원 시)")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1, help="GPU 수 (vLLM 백엔드 사용 시)")
    parser.add_argument("--answer-suffix-en", type=str, default=None,
                        help="영어 답 유도 구절. 주면 한국어/영어 각각 한 번씩 돌려서 전체 정확도 비교 후 더 좋은 쪽을 최종 결과로 저장")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="프롬프트 설정 YAML 파일 경로. 지정 시 sqa 섹션의 모든 프롬프트로 각각 평가 후 최적 결과 저장")
    parser.add_argument("--prompt-name", type=str, default=None,
                        help="prompt-file 사용 시 해당 name 만 실행. 없으면 첫 항목만.")
    args = parser.parse_args()

    if args.prompt_file:
        import yaml
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_cfg = yaml.safe_load(f)
        sqa_prompts = prompt_cfg.get('sqa', [])
        if args.prompt_name:
            sqa_prompts = [p for p in sqa_prompts if p.get("name") == args.prompt_name]
            if not sqa_prompts:
                sqa_prompts = prompt_cfg.get('sqa', [])[:1]
                print(f"[prompt-file] --prompt-name={args.prompt_name} 없음 → 첫 프롬프트만 사용")
        if not sqa_prompts:
            raise ValueError("prompt file에 'sqa' 섹션이 없거나 비어 있습니다.")
        import shutil
        out_base = Path(args.output_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        model = get_backend(args.backend, args.model_path, tensor_parallel_size=args.tensor_parallel_size)
        summaries = []
        for i, p in enumerate(sqa_prompts):
            name = p['name']
            answer_suffix = p.get('answer_suffix', ANSWER_SUFFIX_KO)
            prompt_prefix = p.get('prompt_prefix')
            out_dir = str(out_base / f'prompt_{name}')
            print(f"\n[{i+1}/{len(sqa_prompts)}] 프롬프트: {name}")
            summary = evaluate_sqa(
                jsonl_path=args.jsonl,
                output_dir=out_dir,
                model_path=args.model_path,
                backend_name=args.backend,
                max_samples=args.max_samples,
                base_dir=args.base_dir,
                save_generation=args.save_generation,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                answer_suffix=answer_suffix,
                prompt_prefix=prompt_prefix,
                model=model,
            )
            summaries.append({
                'name': name,
                'prompt_prefix': prompt_prefix,
                'answer_suffix': answer_suffix,
                'accuracy_logit': summary['accuracy_logit'],
                'accuracy_generation': summary['accuracy_generation'],
            })
        best = max(summaries, key=lambda x: x['accuracy_logit'])
        best_dir = out_base / f"prompt_{best['name']}"
        for suf in ('sqa_predictions.jsonl', 'sqa_summary.json'):
            src = best_dir / suf
            dst = out_base / suf
            if src.is_file():
                shutil.copy2(src, dst)
        comparison = {'prompts': summaries, 'best': best['name']}
        cmp_path = out_base / 'prompt_comparison.json'
        with open(cmp_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print('\n' + '=' * 60)
        print('프롬프트 비교 (로짓 정확도)')
        print('=' * 60)
        for s in summaries:
            marker = '  ← best' if s['name'] == best['name'] else ''
            print(f"  {s['name']}: logit={s['accuracy_logit']:.4f}, gen={s['accuracy_generation']:.4f}{marker}")
        print(f"  채택: {best['name']}")
        print(f"  비교 결과: {cmp_path}")
        print('=' * 60)
    elif args.answer_suffix_en:
        # 한국어/영어 각각 한 번씩 실행 후 전체 정확도 비교, 더 좋은 쪽을 최종 결과로
        import shutil
        dataset_name = Path(args.jsonl).stem
        out_ko = Path(args.output_dir) / 'prompt_ko'
        out_en = Path(args.output_dir) / 'prompt_en'
        out_ko.mkdir(parents=True, exist_ok=True)
        out_en.mkdir(parents=True, exist_ok=True)
        print("\n[1/2] 한국어 답 구절로 평가...")
        summary_ko = evaluate_sqa(
            jsonl_path=args.jsonl,
            output_dir=str(out_ko),
            model_path=args.model_path,
            backend_name=args.backend,
            max_samples=args.max_samples,
            base_dir=args.base_dir,
            save_generation=args.save_generation,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            answer_suffix=ANSWER_SUFFIX_KO,
        )
        print("\n[2/2] 영어 답 구절로 평가...")
        summary_en = evaluate_sqa(
            jsonl_path=args.jsonl,
            output_dir=str(out_en),
            model_path=args.model_path,
            backend_name=args.backend,
            max_samples=args.max_samples,
            base_dir=args.base_dir,
            save_generation=args.save_generation,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            answer_suffix=args.answer_suffix_en,
        )
        acc_ko = summary_ko["accuracy_logit"]
        acc_en = summary_en["accuracy_logit"]
        better = "ko" if acc_ko >= acc_en else "en"
        better_dir = out_ko if better == "ko" else out_en
        for suf in ("sqa_predictions.jsonl", "sqa_summary.json"):
            src = better_dir / suf
            dst = Path(args.output_dir) / suf
            if src.is_file():
                shutil.copy2(src, dst)
        comparison = {
            "answer_suffix_ko": ANSWER_SUFFIX_KO,
            "answer_suffix_en": args.answer_suffix_en,
            "accuracy_ko": acc_ko,
            "accuracy_en": acc_en,
            "better": better,
        }
        cmp_path = Path(args.output_dir) / "prompt_comparison.json"
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 60)
        print("답 구절 비교 (전체 정확도)")
        print("=" * 60)
        print(f"  한국어: {acc_ko:.4f}")
        print(f"  영어:   {acc_en:.4f}")
        print(f"  채택:   {better} (위 결과를 최종 저장)")
        print(f"  비교 결과: {cmp_path}")
        print("=" * 60)
    else:
        evaluate_sqa(
            jsonl_path=args.jsonl,
            output_dir=args.output_dir,
            model_path=args.model_path,
            backend_name=args.backend,
            max_samples=args.max_samples,
            base_dir=args.base_dir,
            save_generation=args.save_generation,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            answer_suffix=ANSWER_SUFFIX_KO,
        )


if __name__ == "__main__":
    main()

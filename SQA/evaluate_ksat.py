#!/usr/bin/env python3
"""
K-SAT (수능 듣기) 평가 스크립트 — Logit + Generation 기반

SQA evaluate_sqa.py와 동일한 방식이나, K-SAT JSONL 형식에 맞춘 버전:
  - 선택지: (1)~(5) 숫자 (SQA는 (A)~(D) 알파벳)
  - 질문 필드: prompt_ko (SQA는 prompt)
  - raw 경로에 .wav 확장자가 없을 수 있음 → 자동 보정
  - answer_ko 빈 항목은 스킵

Usage:
  python evaluate_ksat.py \
    --jsonl SQA/K-SAT_2006.jsonl \
    --output_dir results_real/K-SAT/qwen/K-SAT_2006 \
    --backend qwen --model_path Qwen/Qwen2-Audio-7B-Instruct
"""

import os
import sys
import json
import re
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F

_KO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_KO_SRC) not in sys.path:
    sys.path.insert(0, str(_KO_SRC))
from backends import get_backend, list_backends


ANSWER_SUFFIX_KO = "\n답: "
CHOICE_NUMBERS = ["1", "2", "3", "4", "5"]


# ──────────────────────────────────────────────────────────────
#  유틸리티
# ──────────────────────────────────────────────────────────────

def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break
    return data


def parse_choice_numbers(choices_ko: str) -> List[str]:
    """choices_ko에서 선택지 번호 추출. '(1) ... (2) ...' -> ['1', '2', '3', '4', '5']"""
    if not (choices_ko or "").strip():
        return []
    nums = []
    seen = set()
    for m in re.finditer(r"\((\d+)\)", choices_ko):
        n = m.group(1)
        if n not in seen:
            seen.add(n)
            nums.append(n)
    return nums if nums else CHOICE_NUMBERS[:5]


def normalize_gt_to_number(answer_ko: str) -> Optional[str]:
    """answer_ko를 선택지 번호로 정규화. '(5)사막화의 해결 방안' -> '5', '(3)' -> '3'"""
    raw = (answer_ko or "").strip()
    if not raw:
        return None
    m = re.match(r"^\((\d+)\)", raw)
    if m:
        return m.group(1)
    if raw.isdigit() and 1 <= int(raw) <= 9:
        return raw
    return None


def parse_choice_from_generation(generated_text: str, choice_nums: List[str]) -> str:
    """생성 텍스트에서 선택지 번호 추출."""
    if not (generated_text or "").strip():
        return choice_nums[0] if choice_nums else "1"
    text = (generated_text or "").strip()
    for n in choice_nums:
        if f"({n})" in text:
            return n
    for n in choice_nums:
        if re.search(rf"\b{n}\b", text):
            return n
    return choice_nums[0] if choice_nums else "1"


def resolve_audio_path(raw: str) -> str:
    """raw 경로에 .wav 확장자가 없으면 추가."""
    if not raw:
        return raw
    if os.path.exists(raw):
        return raw
    if not raw.endswith(".wav"):
        wav = raw + ".wav"
        if os.path.exists(wav):
            return wav
    return raw


# ──────────────────────────────────────────────────────────────
#  Logit 기반 예측
# ──────────────────────────────────────────────────────────────

def predict_from_logits(processor, next_logits: torch.Tensor, choice_nums: List[str]) -> str:
    tokenizer = processor.tokenizer
    candidates = []
    for n in choice_nums:
        for s in [n, f"({n})", f" ({n})", f" {n}"]:
            tok = tokenizer.encode(s, add_special_tokens=False)
            if len(tok) == 1:
                candidates.append((n, tok[0]))
                break
        else:
            tok = tokenizer.encode(n, add_special_tokens=False)
            if tok:
                candidates.append((n, tok[0]))
    if not candidates:
        return choice_nums[0]
    logprobs = F.log_softmax(next_logits[0], dim=-1)
    best, best_lp = choice_nums[0], -1e9
    for n, tid in candidates:
        lp = logprobs[tid].item()
        if lp > best_lp:
            best_lp, best = lp, n
    return best


def predict_from_logprobs_dict(
    processor, token_id_to_logprob: Dict[int, float], choice_nums: List[str]
) -> str:
    tokenizer = processor.tokenizer
    candidates = []
    for n in choice_nums:
        for s in [n, f"({n})", f" ({n})", f" {n}"]:
            tok = tokenizer.encode(s, add_special_tokens=False)
            if len(tok) == 1:
                candidates.append((n, tok[0]))
                break
        else:
            tok = tokenizer.encode(n, add_special_tokens=False)
            if tok:
                candidates.append((n, tok[0]))
    if not candidates:
        return choice_nums[0]
    best, best_lp = choice_nums[0], -1e9
    for n, tid in candidates:
        if tid in token_id_to_logprob and token_id_to_logprob[tid] > best_lp:
            best_lp = token_id_to_logprob[tid]
            best = n
    return best


def run_one(model, audio_path: str, text_input: str,
            choice_nums: List[str], answer_suffix: str,
            max_new_tokens: int = 64) -> Tuple[str, str, bool]:
    """(pred_logit, pred_gen, used_generation)"""
    next_logits = (
        model.get_next_token_logits(audio_path, text_input, answer_suffix=answer_suffix)
        if hasattr(model, "get_next_token_logits") else None
    )

    if next_logits is None:
        pred_logit = None
    elif isinstance(next_logits, dict):
        pred_logit = predict_from_logprobs_dict(model.processor, next_logits, choice_nums)
    else:
        pred_logit = predict_from_logits(model.processor, next_logits, choice_nums)

    prompt_full = (text_input or "").strip() + answer_suffix
    gen_text = (
        model.inference(audio_path, prompt_full, max_new_tokens=max_new_tokens)
        if hasattr(model, "inference") else ""
    )
    pred_gen = parse_choice_from_generation(gen_text or "", choice_nums)

    return pred_logit, pred_gen, gen_text or ""


# ──────────────────────────────────────────────────────────────
#  메인 평가
# ──────────────────────────────────────────────────────────────

DEFAULT_PROMPT = "다음 음성을 듣고 질문에 맞는 답을 고르세요."


def evaluate_ksat(
    jsonl_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    backend_name: str = "qwen",
    max_samples: Optional[int] = None,
    tensor_parallel_size: int = 1,
    answer_suffix: str = ANSWER_SUFFIX_KO,
    max_new_tokens: int = 64,
    prompt_prefix: Optional[str] = None,
    model=None,
) -> Dict:
    jsonl_path = Path(jsonl_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(str(jsonl_path), max_samples=max_samples)
    if not items:
        raise SystemExit(f"샘플 없음: {jsonl_path}")

    if model is None:
        model = get_backend(backend_name, model_path or "",
                            tensor_parallel_size=tensor_parallel_size)

    prompt = prompt_prefix or DEFAULT_PROMPT
    results = []
    correct_logit = 0
    correct_gen = 0
    skipped = 0
    start = time.time()

    for i, item in enumerate(tqdm(items, desc="K-SAT 평가")):
        index = item.get("index") or str(i).zfill(6)
        prompt_ko = (item.get("prompt_ko") or "").strip()
        choices_ko = (item.get("choices_ko") or "").strip()
        answer_ko = (item.get("answer_ko") or "").strip()

        # hslim 형식: choices_ko 비었고 question_en에 선택지 있는 경우
        if not choices_ko and item.get("question_en"):
            choices_ko = (item.get("question_en") or "").strip()
        if not answer_ko and item.get("answer_en"):
            answer_ko = (item.get("answer_en") or "").strip()

        gt = normalize_gt_to_number(answer_ko)
        if gt is None:
            skipped += 1
            results.append({
                "index": index, "prediction_logit": "", "prediction_gen": "",
                "answer_ko": answer_ko, "correct_logit": False,
                "correct_gen": False, "note": "no_answer",
            })
            continue

        raw = (item.get("raw") or "").strip()
        audio_path = resolve_audio_path(raw)
        if not os.path.exists(audio_path):
            results.append({
                "index": index, "prediction_logit": "", "prediction_gen": "",
                "answer_ko": answer_ko, "correct_logit": False,
                "correct_gen": False, "note": "audio_missing",
            })
            continue

        choice_nums = parse_choice_numbers(choices_ko)

        text_input = f"{prompt}\n\n{prompt_ko}\n\n{choices_ko}" if prompt_ko else f"{prompt}\n\n{choices_ko}"

        pred_l, pred_g, gen_text = run_one(
            model, audio_path, text_input, choice_nums, answer_suffix, max_new_tokens
        )

        # logit 결과 없으면 generation으로 대체
        if pred_l is None:
            pred_l = pred_g

        is_correct_l = (pred_l == gt)
        is_correct_g = (pred_g == gt)
        if is_correct_l:
            correct_logit += 1
        if is_correct_g:
            correct_gen += 1

        results.append({
            "index": index,
            "prediction_logit": f"({pred_l})",
            "prediction_gen": f"({pred_g})",
            "answer_ko": answer_ko,
            "correct_logit": is_correct_l,
            "correct_gen": is_correct_g,
            "generation": gen_text,
        })

    elapsed = time.time() - start
    total = len(results)
    evaluated = total - skipped

    acc_logit = correct_logit / evaluated if evaluated else 0.0
    acc_gen = correct_gen / evaluated if evaluated else 0.0

    # 저장
    out_jsonl = output_dir / "ksat_predictions.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "accuracy_logit": round(acc_logit, 4),
        "accuracy_generation": round(acc_gen, 4),
        "correct_logit": correct_logit,
        "correct_generation": correct_gen,
        "total": total,
        "evaluated": evaluated,
        "skipped_no_answer": skipped,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(output_dir / "ksat_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("K-SAT 평가 결과")
    print("=" * 60)
    print(f"  정확도 (logit) : {acc_logit:.4f}  ({correct_logit}/{evaluated})")
    print(f"  정확도 (생성)  : {acc_gen:.4f}  ({correct_gen}/{evaluated})")
    print(f"  전체: {total}건, 평가: {evaluated}건, 스킵(정답없음): {skipped}건")
    print(f"  소요 시간: {elapsed:.1f}초")
    print(f"  결과: {out_jsonl}")
    print("=" * 60)

    return summary


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="K-SAT (수능 듣기) 평가 — logit + generation 기반",
        epilog=f"백엔드: {', '.join(list_backends())}",
    )
    parser.add_argument("--jsonl", type=str, required=True, help="K-SAT JSONL 경로")
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--backend", type=str, default="qwen", choices=list_backends())
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt-prefix", type=str, default=None,
                        help="커스텀 프롬프트 (기본: '다음 음성을 듣고 질문에 맞는 답을 고르세요.')")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="프롬프트 YAML 경로. 지정 시 ksat 섹션의 모든 프롬프트로 평가 후 최적 결과 저장")
    parser.add_argument("--prompt-name", type=str, default=None,
                        help="prompt-file 사용 시 해당 name만 실행 (예: v1)")
    args = parser.parse_args()

    if args.prompt_file:
        import yaml
        import shutil
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_cfg = yaml.safe_load(f)
        ksat_prompts = prompt_cfg.get("ksat", [])
        if not ksat_prompts:
            raise ValueError("prompt file에 'ksat' 섹션이 없거나 비어 있습니다.")
        if args.prompt_name:
            ksat_prompts = [p for p in ksat_prompts if p.get("name") == args.prompt_name]
            if not ksat_prompts:
                ksat_prompts = prompt_cfg.get("ksat", [])[:1]
                print(f"[prompt-file] --prompt-name={args.prompt_name} 없음 → 첫 프롬프트만 사용")
        out_base = Path(args.output_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        model = get_backend(args.backend, args.model_path or "", tensor_parallel_size=args.tensor_parallel_size)
        summaries = []
        for i, p in enumerate(ksat_prompts):
            name = p["name"]
            answer_suffix = p.get("answer_suffix", ANSWER_SUFFIX_KO)
            prompt_prefix = p.get("prompt_prefix")
            out_dir = str(out_base / f"prompt_{name}")
            print(f"\n[{i+1}/{len(ksat_prompts)}] 프롬프트: {name}")
            summary = evaluate_ksat(
                jsonl_path=args.jsonl,
                output_dir=out_dir,
                model_path=args.model_path,
                backend_name=args.backend,
                max_samples=args.max_samples,
                tensor_parallel_size=args.tensor_parallel_size,
                answer_suffix=answer_suffix,
                max_new_tokens=args.max_new_tokens,
                prompt_prefix=prompt_prefix,
                model=model,
            )
            summaries.append({
                "name": name,
                "accuracy_logit": summary["accuracy_logit"],
                "accuracy_generation": summary["accuracy_generation"],
            })
        best = max(summaries, key=lambda x: x["accuracy_logit"])
        best_dir = out_base / f"prompt_{best['name']}"
        for suf in ("ksat_predictions.jsonl", "ksat_summary.json"):
            src = best_dir / suf
            dst = out_base / suf
            if src.is_file():
                shutil.copy2(src, dst)
        comparison = {"prompts": summaries, "best": best["name"]}
        cmp_path = out_base / "prompt_comparison.json"
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 60)
        print("프롬프트 비교 (로짓 정확도)")
        print("=" * 60)
        for s in summaries:
            marker = "  ← best" if s["name"] == best["name"] else ""
            print(f"  {s['name']}: logit={s['accuracy_logit']:.4f}, gen={s['accuracy_generation']:.4f}{marker}")
        print(f"  채택: {best['name']}")
        print("=" * 60)
    else:
        evaluate_ksat(
            jsonl_path=args.jsonl,
            output_dir=args.output_dir,
            model_path=args.model_path,
            backend_name=args.backend,
            max_samples=args.max_samples,
            tensor_parallel_size=args.tensor_parallel_size,
            max_new_tokens=args.max_new_tokens,
            prompt_prefix=args.prompt_prefix,
        )


if __name__ == "__main__":
    main()

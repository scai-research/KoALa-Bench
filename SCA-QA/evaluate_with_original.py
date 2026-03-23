#!/usr/bin/env python3
"""
K-disentQA 3-조건 평가 + 4-case 분석

3가지 평가 조건:
  1) text_only          : 무음 wav + question + choices  →  정답 = answer
  2) speech_new_context : new_context 음성 + question    →  정답 = new_answer
  3) speech_orig_context: original_context 음성 + question → 정답 = answer

text_only를 맞춘 샘플에서 나머지 두 조건의 정답 여부에 따른 4-case 분석:
  Case 1 (new O, orig O) : 음성 컨텍스트를 모두 올바르게 활용 (이상적)
  Case 2 (new O, orig X) : 새 컨텍스트는 따르지만 원본 컨텍스트에 혼란
  Case 3 (new X, orig O) : 새 컨텍스트 무시, parametric knowledge 유지
  Case 4 (new X, orig X) : 음성 입력 전반에 취약

Usage:
  python evaluate_with_original.py \
    --jsonl K-disentQA/history_after_chosun_tts_filtered.jsonl \
    --output_dir results/K-disentQA/qwen/history_after_chosun_tts_filtered_3cond \
    --speech-dir audio/k-disentqa/history_after_chosun_final \
    --original-speech-dir audio/k-disentqa-original/history_after_chosun_final \
    --backend qwen --model_path Qwen/Qwen2-Audio-7B-Instruct
"""

import json
import os
import sys
import wave
import struct
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import torch
import torch.nn.functional as F

_KO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_KO_SRC) not in sys.path:
    sys.path.insert(0, str(_KO_SRC))
from backends import get_backend, list_backends


ANSWER_SUFFIX_KO = "\n답: "
CHOICE_LETTERS = ["A", "B", "C", "D"]

PROMPT_TEXT_ONLY = "다음 질문에 답변하세요."
PROMPT_SPEECH = "음성에서 들려준 내용만을 근거로 다음 질문에 답변하세요."


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


def build_text_input(
    question: str,
    choices: List[str],
    use_speech_prompt: bool = False,
    custom_text_only: Optional[str] = None,
    custom_speech: Optional[str] = None,
) -> str:
    if use_speech_prompt:
        prefix = custom_speech or PROMPT_SPEECH
    else:
        prefix = custom_text_only or PROMPT_TEXT_ONLY
    lines = [prefix, "질문: " + (question or "").strip()]
    for i, c in enumerate(choices[:4]):
        letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else chr(ord("A") + i)
        lines.append(f"({letter}) {c}")
    return "\n\n".join(lines)


def answer_to_choice_index(answer_text: str, choices: List[str]) -> int:
    if not answer_text or not choices:
        return 0
    s = answer_text.strip()
    for i, c in enumerate(choices):
        if (c or "").strip() == s:
            return i
    return 0


def correct_index_to_letter(idx: int) -> str:
    if 0 <= idx < len(CHOICE_LETTERS):
        return CHOICE_LETTERS[idx]
    return CHOICE_LETTERS[0]


def create_silent_wav(path: str, duration_sec: float = 0.1, sample_rate: int = 16000):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_sec * sample_rate)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n}h", *([0] * n)))


# ──────────────────────────────────────────────────────────────
#  추론
# ──────────────────────────────────────────────────────────────

def predict_choice_from_logits(processor, next_logits: torch.Tensor, choice_letters: List[str]) -> str:
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


def predict_choice_from_logprobs_dict(
    processor, token_id_to_logprob: Dict[int, float], choice_letters: List[str]
) -> str:
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
    best_letter, best_lp = choice_letters[0], -1e9
    for L, tid in candidates:
        if tid in token_id_to_logprob and token_id_to_logprob[tid] > best_lp:
            best_lp = token_id_to_logprob[tid]
            best_letter = L
    return best_letter


def run_one(model, audio_path: str, text_input: str,
            choice_letters: List[str], answer_suffix: str) -> Tuple[str, bool]:
    next_logits = (
        model.get_next_token_logits(audio_path, text_input, answer_suffix=answer_suffix)
        if hasattr(model, "get_next_token_logits") else None
    )
    if next_logits is None:
        prompt = (text_input or "").strip() + answer_suffix
        gen = model.inference(audio_path, prompt, max_new_tokens=32) if hasattr(model, "inference") else ""
        gen = (gen or "").strip()
        for L in choice_letters:
            if f"({L})" in gen or f"({L.lower()})" in gen:
                return L, True
            if L in gen.upper():
                return L, True
        return choice_letters[0], True

    if isinstance(next_logits, dict):
        pred = predict_choice_from_logprobs_dict(model.processor, next_logits, choice_letters)
    else:
        pred = predict_choice_from_logits(model.processor, next_logits, choice_letters)
    return pred, False


# ──────────────────────────────────────────────────────────────
#  오디오 경로 해석
# ──────────────────────────────────────────────────────────────

def resolve_audio(item: Dict, raw_key: str, speech_dir: Optional[Path],
                  index: str, suffix: str, jsonl_path: Path) -> Optional[str]:
    """speech_dir/{index}{suffix} → JSONL[raw_key] 순으로 시도."""
    if speech_dir:
        p = speech_dir / f"{index}{suffix}"
        if p.exists():
            return str(p)
    raw = (item.get(raw_key) or "").strip()
    if raw:
        rp = Path(raw)
        if not rp.is_absolute():
            ko_base = jsonl_path.resolve().parent.parent
            rp = (ko_base / raw).resolve()
        if rp.is_file():
            return str(rp)
    return None


# ──────────────────────────────────────────────────────────────
#  메인 평가
# ──────────────────────────────────────────────────────────────

def evaluate(
    jsonl_path: str,
    output_dir: str,
    speech_dir: str,
    original_speech_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    backend_name: str = "qwen",
    max_samples: Optional[int] = None,
    tensor_parallel_size: int = 1,
    prompt_text_only: Optional[str] = None,
    prompt_speech: Optional[str] = None,
    answer_suffix_override: Optional[str] = None,
    audio_suffix: str = "_tts.wav",
    original_audio_suffix: str = "_tts.wav",
    model=None,
) -> Dict:
    jsonl_path = Path(jsonl_path).resolve()
    output_dir = Path(output_dir).resolve()
    speech_dir_p = Path(speech_dir).resolve()
    orig_dir_p = Path(original_speech_dir).resolve() if original_speech_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(str(jsonl_path), max_samples=max_samples)
    if not items:
        raise SystemExit(f"샘플 없음: {jsonl_path}")

    silent_wav = output_dir / "_silent.wav"
    create_silent_wav(str(silent_wav))

    if model is None:
        model = get_backend(backend_name, model_path or "",
                            tensor_parallel_size=tensor_parallel_size)
    answer_suffix = answer_suffix_override or ANSWER_SUFFIX_KO

    res_text, res_new, res_orig = [], [], []
    start = time.time()

    for i, item in enumerate(tqdm(items, desc="K-disentQA 3-조건 평가")):
        index = item.get("index") or str(i).zfill(6)
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer_original = item.get("answer", "")
        answer_new = item.get("new_answer", "")
        choice_letters = CHOICE_LETTERS[: max(4, len(choices))]

        gt_text = correct_index_to_letter(answer_to_choice_index(answer_original, choices))
        gt_new = correct_index_to_letter(answer_to_choice_index(answer_new, choices))
        gt_orig = gt_text  # original context → 정답 = answer

        inp_text = build_text_input(question, choices, False, prompt_text_only)
        inp_speech = build_text_input(question, choices, True, custom_speech=prompt_speech)

        # ── 1) text_only ──
        pred_t, _ = run_one(model, str(silent_wav), inp_text, choice_letters, answer_suffix)
        res_text.append({"index": index, "prediction": pred_t,
                         "correct": pred_t == gt_text, "gt": gt_text})

        # ── 2) speech + new_context ──
        a_new = resolve_audio(item, "raw", speech_dir_p, index, audio_suffix, jsonl_path)
        if a_new:
            pred_n, _ = run_one(model, a_new, inp_speech, choice_letters, answer_suffix)
            res_new.append({"index": index, "prediction": pred_n,
                            "correct": pred_n == gt_new, "gt": gt_new})
        else:
            res_new.append({"index": index, "prediction": "", "correct": False,
                            "gt": gt_new, "note": "audio_missing"})

        # ── 3) speech + original_context ──
        a_orig = resolve_audio(item, "original_raw", orig_dir_p, index,
                               original_audio_suffix, jsonl_path)
        if a_orig:
            pred_o, _ = run_one(model, a_orig, inp_speech, choice_letters, answer_suffix)
            res_orig.append({"index": index, "prediction": pred_o,
                             "correct": pred_o == gt_orig, "gt": gt_orig})
        else:
            res_orig.append({"index": index, "prediction": "", "correct": False,
                             "gt": gt_orig, "note": "audio_missing"})

    elapsed = time.time() - start
    total = len(res_text)

    # ── 정확도 ──
    c_text = sum(r["correct"] for r in res_text)
    c_new = sum(r["correct"] for r in res_new)
    c_orig = sum(r["correct"] for r in res_orig)
    acc_text = c_text / total if total else 0.0
    acc_new = c_new / total if total else 0.0
    acc_orig = c_orig / total if total else 0.0

    # ── parametric proportion (기존 호환) ──
    param_same = sum(
        1 for rt, rn in zip(res_text, res_new)
        if rt["correct"] and rt["prediction"] == rn["prediction"]
    )
    param_prop = param_same / c_text if c_text else 0.0

    # ── 4-case 분석 (text_only 정답인 샘플만) ──
    cases = {1: [], 2: [], 3: [], 4: []}
    for rt, rn, ro in zip(res_text, res_new, res_orig):
        if not rt["correct"]:
            continue
        new_ok = rn["correct"]
        orig_ok = ro["correct"]
        if new_ok and orig_ok:
            cases[1].append(rt["index"])
        elif new_ok and not orig_ok:
            cases[2].append(rt["index"])
        elif not new_ok and orig_ok:
            cases[3].append(rt["index"])
        else:
            cases[4].append(rt["index"])

    cn = {k: len(v) for k, v in cases.items()}

    # ── 저장 ──
    for fname, data in [
        ("predictions_text.jsonl", res_text),
        ("predictions_speech_new.jsonl", res_new),
        ("predictions_speech_original.jsonl", res_orig),
    ]:
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total": total,
        "accuracy_text": round(acc_text, 4),
        "accuracy_speech_new": round(acc_new, 4),
        "accuracy_speech_original": round(acc_orig, 4),
        "correct_text": c_text,
        "correct_speech_new": c_new,
        "correct_speech_original": c_orig,
        "parametric_proportion": round(param_prop, 4),
        "text_correct_and_new_same": param_same,
        "case_analysis": {
            "text_only_correct": c_text,
            "case_1_new_O_orig_O": cn[1],
            "case_2_new_O_orig_X": cn[2],
            "case_3_new_X_orig_O": cn[3],
            "case_4_new_X_orig_X": cn[4],
        },
        "case_proportions": {
            f"case_{k}": round(cn[k] / c_text, 4) if c_text else 0.0
            for k in range(1, 5)
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(output_dir / "kdisentqa_3cond_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "case_details.json", "w", encoding="utf-8") as f:
        json.dump({f"case_{k}": v for k, v in cases.items()}, f, ensure_ascii=False, indent=2)

    # ── 출력 ──
    print()
    print("=" * 70)
    print("K-disentQA 3-조건 평가 결과")
    print("=" * 70)
    print(f"  text_only           : {acc_text:.4f}  ({c_text}/{total})")
    print(f"  speech+new_context  : {acc_new:.4f}  ({c_new}/{total})")
    print(f"  speech+orig_context : {acc_orig:.4f}  ({c_orig}/{total})")
    print(f"  parametric proportion: {param_prop:.4f}  ({param_same}/{c_text})")
    print()
    if c_text:
        print(f"  ── text_only 정답 ({c_text}건) 기준 4-case ──")
        print(f"  Case 1 (new O, orig O) : {cn[1]:4d}  ({cn[1]/c_text:.4f})  음성 컨텍스트 모두 활용")
        print(f"  Case 2 (new O, orig X) : {cn[2]:4d}  ({cn[2]/c_text:.4f})  원본 컨텍스트에 혼란")
        print(f"  Case 3 (new X, orig O) : {cn[3]:4d}  ({cn[3]/c_text:.4f})  parametric knowledge 유지")
        print(f"  Case 4 (new X, orig X) : {cn[4]:4d}  ({cn[4]/c_text:.4f})  음성 입력 전반에 취약")
    else:
        print("  text_only 정답 0건 → case 분석 불가")
    print(f"  소요 시간: {elapsed:.1f}초")
    print("=" * 70)

    return summary


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="K-disentQA 3-조건 평가 (text_only / speech+new / speech+original) + 4-case 분석",
        epilog=f"백엔드: {', '.join(list_backends())}",
    )
    parser.add_argument("--jsonl", type=str, required=True,
                        help="K-disentQA JSONL (index, question, choices, answer, new_answer, raw, original_raw)")
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--speech-dir", type=str, required=True,
                        help="new_context 음성 디렉토리 (기존 평가의 speech_output_dir)")
    parser.add_argument("--original-speech-dir", type=str, default=None,
                        help="original_context 음성 디렉토리 (미지정 시 JSONL의 original_raw 키로 폴백)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--backend", type=str, default="qwen", choices=list_backends())
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument("--audio-suffix", type=str, default="_tts.wav",
                        help="new_context 음성 파일 접미사")
    parser.add_argument("--original-audio-suffix", type=str, default="_tts.wav",
                        help="original_context 음성 파일 접미사")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="프롬프트 YAML (kdisentqa 섹션 사용)")
    parser.add_argument("--prompt-name", type=str, default=None,
                        help="prompt-file 내 특정 프롬프트 name 필터")
    args = parser.parse_args()

    if args.prompt_file:
        import yaml
        import shutil
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kd = cfg.get("kdisentqa", [])
        if args.prompt_name:
            kd = [p for p in kd if p.get("name") == args.prompt_name]
            if not kd:
                kd = cfg.get("kdisentqa", [])[:1]
                print(f"[prompt-file] --prompt-name={args.prompt_name} 없음 → 첫 프롬프트 사용")
        if not kd:
            raise ValueError("prompt file에 'kdisentqa' 섹션이 없거나 비어 있습니다.")

        out_base = Path(args.output_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        _model = get_backend(args.backend, args.model_path or "",
                             tensor_parallel_size=args.tensor_parallel_size)
        summaries = []
        for i, p in enumerate(kd):
            name = p["name"]
            out_dir = str(out_base / f"prompt_{name}")
            print(f"\n[{i+1}/{len(kd)}] 프롬프트: {name}")
            s = evaluate(
                jsonl_path=args.jsonl, output_dir=out_dir,
                speech_dir=args.speech_dir,
                original_speech_dir=args.original_speech_dir,
                model_path=args.model_path, backend_name=args.backend,
                max_samples=args.max_samples,
                tensor_parallel_size=args.tensor_parallel_size,
                prompt_text_only=p.get("prompt_text_only"),
                prompt_speech=p.get("prompt_speech"),
                answer_suffix_override=p.get("answer_suffix"),
                audio_suffix=args.audio_suffix,
                original_audio_suffix=args.original_audio_suffix,
                model=_model,
            )
            summaries.append({"name": name, **s})

        best = max(summaries, key=lambda x: x["accuracy_speech_new"])
        best_dir = out_base / f"prompt_{best['name']}"
        for suf in ("predictions_text.jsonl", "predictions_speech_new.jsonl",
                     "predictions_speech_original.jsonl",
                     "kdisentqa_3cond_summary.json", "case_details.json"):
            src = best_dir / suf
            dst = out_base / suf
            if src.is_file():
                shutil.copy2(src, dst)

        comparison = {"prompts": summaries, "best": best["name"]}
        cmp_path = out_base / "prompt_comparison.json"
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print("프롬프트 비교")
        print("=" * 70)
        for s in summaries:
            marker = "  <- best" if s["name"] == best["name"] else ""
            print(f"  {s['name']}: text={s['accuracy_text']:.4f}  "
                  f"new={s['accuracy_speech_new']:.4f}  "
                  f"orig={s['accuracy_speech_original']:.4f}  "
                  f"param={s['parametric_proportion']:.4f}{marker}")
        print(f"  채택: {best['name']}")
        print("=" * 70)
    else:
        evaluate(
            jsonl_path=args.jsonl, output_dir=args.output_dir,
            speech_dir=args.speech_dir,
            original_speech_dir=args.original_speech_dir,
            model_path=args.model_path, backend_name=args.backend,
            max_samples=args.max_samples,
            tensor_parallel_size=args.tensor_parallel_size,
            audio_suffix=args.audio_suffix,
            original_audio_suffix=args.original_audio_suffix,
        )


if __name__ == "__main__":
    main()

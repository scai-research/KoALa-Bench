#!/usr/bin/env python3
"""
K-disentQA 평가: Ko-Speech-Eval 백엔드로 두 가지 방식 추론 후 정확도·parametric proportion 계산.

1) 텍스트만: "다음 질문에 답변하세요." + question + choices (무음 오디오) → 예측 저장
2) 음성+텍스트: "음성을 듣고 다음 질문에 답변하세요." + question + choices (context wav) → 예측 저장

- accuracy_text, accuracy_speech 각각 계산·출력
- 텍스트만: 정답 = answer (원래 정답)
- 음성+텍스트: 정답 = new_answer (entity 변환 후 정답)
- parametric_proportion: (1번에서 맞춘 것들 중 2번에서도 1번과 같은 답을 고른 것) / (1번에서 맞춘 개수)
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


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break
    return data


PROMPT_TEXT_ONLY = "다음 질문에 답변하세요."
PROMPT_SPEECH = "음성을 듣고 다음 질문에 답변하세요."


def build_text_input(
    question: str,
    choices: List[str],
    use_speech_prompt: bool = False,
    custom_text_only: Optional[str] = None,
    custom_speech: Optional[str] = None,
) -> str:
    """K-disentQA용 프롬프트: 지시문 + 질문 + (A)/(B)/(C)/(D) 선택지."""
    if use_speech_prompt:
        prefix = custom_speech or PROMPT_SPEECH
    else:
        prefix = custom_text_only or PROMPT_TEXT_ONLY
    lines = [prefix, "질문: " + (question or "").strip()]
    for i, c in enumerate(choices[:4]):
        letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else chr(ord("A") + i)
        lines.append(f"({letter}) {c}")
    return "\n\n".join(lines)


def correct_index_to_letter(idx: int) -> str:
    if 0 <= idx < len(CHOICE_LETTERS):
        return CHOICE_LETTERS[idx]
    return CHOICE_LETTERS[0]


def answer_to_choice_index(answer_text: str, choices: List[str]) -> int:
    """choices에서 answer_text와 일치하는 인덱스 반환. 없으면 0."""
    if not answer_text or not choices:
        return 0
    s = (answer_text or "").strip()
    for i, c in enumerate(choices):
        if (c or "").strip() == s:
            return i
    return 0


def create_silent_wav(path: str, duration_sec: float = 0.1, sample_rate: int = 16000) -> None:
    """무음 wav 파일 생성 (텍스트 전용 추론 시 같은 백엔드에 넘길 오디오)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n}h", *([0] * n)))


def predict_choice_from_logits(processor, next_logits: torch.Tensor, choice_letters: List[str]) -> str:
    """next_logits(1, vocab)에서 선택지 토큰 logprob 비교."""
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
    """vLLM 등 token_id -> logprob 딕셔너리에서 선택지 예측."""
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


def run_one(
    model,
    audio_path: str,
    text_input: str,
    choice_letters: List[str],
    answer_suffix: str,
) -> Tuple[str, bool]:
    """한 샘플에 대해 로짓 또는 생성으로 예측. (pred_letter, used_generation)."""
    next_logits = model.get_next_token_logits(
        audio_path, text_input, answer_suffix=answer_suffix
    ) if hasattr(model, "get_next_token_logits") else None

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


def evaluate(
    jsonl_path: str,
    output_dir: str,
    speech_output_dir: str,
    model_path: Optional[str] = None,
    backend_name: str = "qwen",
    max_samples: Optional[int] = None,
    tensor_parallel_size: int = 1,
    prompt_text_only: Optional[str] = None,
    prompt_speech: Optional[str] = None,
    answer_suffix_override: Optional[str] = None,
    audio_suffix: str = "_tts.wav",
    model=None,
) -> Dict:
    """
    - items: K-disentQA JSONL (index, question, choices, answer, new_answer)
    - 텍스트만 정답 = answer, 음성+텍스트 정답 = new_answer (choices에서 인덱스 계산)
    - 1) 텍스트만: silent wav + question + choices → predictions_text.jsonl
    - 2) 음성: speech_output_dir/{index}.wav + question + choices → predictions_speech.jsonl
    - accuracy_text, accuracy_speech, parametric_proportion 계산
    """
    jsonl_path = Path(jsonl_path).resolve()
    output_dir = Path(output_dir).resolve()
    speech_output_dir = Path(speech_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(str(jsonl_path), max_samples=max_samples)
    if not items:
        raise SystemExit(f"샘플 없음: {jsonl_path}")

    silent_wav = output_dir / "_silent_kdisentqa.wav"
    create_silent_wav(str(silent_wav))

    if model is None:
        model = get_backend(backend_name, model_path or "", tensor_parallel_size=tensor_parallel_size)
    answer_suffix = answer_suffix_override or ANSWER_SUFFIX_KO

    results_text = []
    results_speech = []
    start = time.time()

    for i, item in enumerate(tqdm(items, desc="K-disentQA 평가")):
        index = item.get("index") or str(i).zfill(6)
        question = item.get("question", "")
        choices = item.get("choices", [])
        # 텍스트만: 정답 = answer (원래 정답)
        # 음성+텍스트: 정답 = new_answer (entity 변환 후)
        answer_orig = item.get("answer", "")
        answer_new = item.get("new_answer", "")
        correct_index_text = answer_to_choice_index(answer_orig, choices)
        correct_index_speech = answer_to_choice_index(answer_new, choices)
        gt_letter_text = correct_index_to_letter(correct_index_text)
        gt_letter_speech = correct_index_to_letter(correct_index_speech)
        text_input_text = build_text_input(
            question, choices, use_speech_prompt=False,
            custom_text_only=prompt_text_only,
        )
        text_input_speech = build_text_input(
            question, choices, use_speech_prompt=True,
            custom_speech=prompt_speech,
        )
        choice_letters = CHOICE_LETTERS[: max(4, len(choices))]

        # 1) 텍스트만: "다음 질문에 답변하세요" + question + choices (무음 오디오) → 정답 = answer
        pred_text, _ = run_one(
            model, str(silent_wav), text_input_text, choice_letters, answer_suffix
        )
        results_text.append({
            "index": index,
            "prediction": pred_text,
            "correct": pred_text == gt_letter_text,
            "correct_index": correct_index_text,
        })

        # 2) 음성 (context wav) → 정답 = new_answer
        audio_path = speech_output_dir / f"{index}{audio_suffix}"
        # speech_output_dir가 잘못 넘어온 경우(예: history_after_chosun_tts → *_tts_final 없음)
        # JSONL의 raw가 있으면 Ko-Speech-Eval 기준 상대/절대 경로로 폴백
        if not audio_path.exists():
            raw = (item.get("raw") or "").strip()
            if raw:
                raw_path = Path(raw)
                if not raw_path.is_absolute():
                    ko_base = jsonl_path.parent.parent  # .../K-disentQA -> .../Ko-Speech-Eval
                    raw_path = (ko_base / raw).resolve()
                if raw_path.is_file():
                    audio_path = raw_path
        if not audio_path.exists():
            results_speech.append({
                "index": index,
                "prediction": "",
                "correct": False,
                "correct_index": correct_index_speech,
                "note": "audio_missing",
            })
        else:
            pred_speech, _ = run_one(
                model, str(audio_path), text_input_speech, choice_letters, answer_suffix
            )
            results_speech.append({
                "index": index,
                "prediction": pred_speech,
                "correct": pred_speech == gt_letter_speech,
                "correct_index": correct_index_speech,
            })

    elapsed = time.time() - start
    total = len(results_text)

    correct_text = sum(1 for r in results_text if r["correct"])
    correct_speech = sum(1 for r in results_speech if r["correct"])
    acc_text = correct_text / total if total else 0.0
    acc_speech = correct_speech / total if total else 0.0

    # 1번에서 맞춘 것들 중, 2번에서도 1번에서 고른 정답과 같을 때
    text_correct_and_speech_same = sum(
        1 for rt, rs in zip(results_text, results_speech)
        if rt["correct"] and rt["prediction"] == rs["prediction"]
    )
    parametric_proportion = (
        text_correct_and_speech_same / correct_text if correct_text else 0.0
    )

    out_text = output_dir / "predictions_text.jsonl"
    out_speech = output_dir / "predictions_speech.jsonl"
    with open(out_text, "w", encoding="utf-8") as f:
        for r in results_text:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_speech, "w", encoding="utf-8") as f:
        for r in results_speech:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "accuracy_text": acc_text,
        "accuracy_speech": acc_speech,
        "correct_text": correct_text,
        "correct_speech": correct_speech,
        "total": total,
        "text_correct_and_speech_same": text_correct_and_speech_same,
        "parametric_proportion": parametric_proportion,
        "elapsed_seconds": elapsed,
        "predictions_text": str(out_text),
        "predictions_speech": str(out_speech),
    }
    with open(output_dir / "kdisentqa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("K-disentQA 평가 결과")
    print("=" * 60)
    print(f"정확도 (텍스트만, question+choices): {acc_text:.4f} ({correct_text}/{total})")
    print(f"정확도 (음성+텍스트, context+question+choices): {acc_speech:.4f} ({correct_speech}/{total})")
    print(f"Parametric proportion (1번 맞춤 & 2번도 1번과 같은 답) / (1번 맞춤): {parametric_proportion:.4f} ({text_correct_and_speech_same}/{correct_text})")
    print(f"소요 시간: {elapsed:.1f}초")
    print(f"결과: {out_text}, {out_speech}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="K-disentQA 평가 (텍스트만 / 음성+텍스트, Ko-Speech-Eval 백엔드)",
        epilog=f"백엔드: {', '.join(list_backends())}",
    )
    parser.add_argument("--jsonl", type=str, required=True, help="K-disentQA JSONL (index, question, choices, answer, new_answer)")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="결과 저장 디렉토리")
    parser.add_argument("--speech_output_dir", type=str, required=True, help="context 음성 wav 디렉토리 (speech_output, index가 history_before_chosun/000000 형식이면 speech_output/history_before_chosun/000000.wav)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--backend", type=str, default="qwen", choices=list_backends())
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument("--audio_suffix", type=str, default="_tts.wav",
                        help="음성 파일명 접미사 (기본: _tts.wav, noisy: _tts_noisy.wav)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="프롬프트 설정 YAML 파일 경로. 지정 시 kdisentqa 섹션의 모든 프롬프트로 각각 평가 후 최적 결과 저장")
    parser.add_argument("--prompt-name", type=str, default=None,
                        help="prompt-file 사용 시 name 이 일치하는 프롬프트만 실행 (API 비용 절약). 없으면 첫 항목만.")
    args = parser.parse_args()

    if args.prompt_file:
        import yaml
        import shutil
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_cfg = yaml.safe_load(f)
        kd_prompts = prompt_cfg.get('kdisentqa', [])
        if args.prompt_name:
            kd_prompts = [p for p in kd_prompts if p.get("name") == args.prompt_name]
            if not kd_prompts:
                kd_prompts = prompt_cfg.get('kdisentqa', [])[:1]
                print(f"[prompt-file] --prompt-name={args.prompt_name} 없음 → v1 등 첫 프롬프트만 사용")
        if not kd_prompts:
            raise ValueError("prompt file에 'kdisentqa' 섹션이 없거나 비어 있습니다.")
        out_base = Path(args.output_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        _model = get_backend(args.backend, args.model_path or "", tensor_parallel_size=args.tensor_parallel_size)
        summaries = []
        for i, p in enumerate(kd_prompts):
            name = p['name']
            out_dir = str(out_base / f'prompt_{name}')
            print(f"\n[{i+1}/{len(kd_prompts)}] 프롬프트: {name}")
            summary = evaluate(
                jsonl_path=args.jsonl,
                output_dir=out_dir,
                speech_output_dir=args.speech_output_dir,
                model_path=args.model_path,
                backend_name=args.backend,
                max_samples=args.max_samples,
                tensor_parallel_size=args.tensor_parallel_size,
                prompt_text_only=p.get('prompt_text_only'),
                prompt_speech=p.get('prompt_speech'),
                answer_suffix_override=p.get('answer_suffix'),
                audio_suffix=args.audio_suffix,
                model=_model,
            )
            summaries.append({
                'name': name,
                'accuracy_text': summary['accuracy_text'],
                'accuracy_speech': summary['accuracy_speech'],
                'parametric_proportion': summary['parametric_proportion'],
            })
        best = max(summaries, key=lambda x: x['accuracy_speech'])
        best_dir = out_base / f"prompt_{best['name']}"
        for suf in ('predictions_text.jsonl', 'predictions_speech.jsonl', 'kdisentqa_summary.json'):
            src = best_dir / suf
            dst = out_base / suf
            if src.is_file():
                shutil.copy2(src, dst)
        comparison = {'prompts': summaries, 'best': best['name']}
        cmp_path = out_base / 'prompt_comparison.json'
        with open(cmp_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print('\n' + '=' * 60)
        print('프롬프트 비교 (음성+텍스트 정확도)')
        print('=' * 60)
        for s in summaries:
            marker = '  ← best' if s['name'] == best['name'] else ''
            print(f"  {s['name']}: text={s['accuracy_text']:.4f}, speech={s['accuracy_speech']:.4f}, param={s['parametric_proportion']:.4f}{marker}")
        print(f"  채택: {best['name']}")
        print(f"  비교 결과: {cmp_path}")
        print('=' * 60)
    else:
        evaluate(
            jsonl_path=args.jsonl,
            output_dir=args.output_dir,
            speech_output_dir=args.speech_output_dir,
            model_path=args.model_path,
            backend_name=args.backend,
            max_samples=args.max_samples,
            tensor_parallel_size=args.tensor_parallel_size,
            audio_suffix=args.audio_suffix,
        )


if __name__ == "__main__":
    main()

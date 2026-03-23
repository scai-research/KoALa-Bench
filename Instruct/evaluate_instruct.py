#!/usr/bin/env python3
"""
Instruct 평가 스크립트 (모델 무관, backends 사용) — Judge Prompt 기반

- 추론: --run_inference 시 get_backend(backend, model_path) 로 생성 (선택)
- 평가: 추론 결과 JSONL과 원본 JSONL 매칭 후 Judge Prompt로 점수 계산
"""
import os
import sys
import json
import argparse
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from openai import OpenAI

# Ko-Speech-Eval/src (추론 시에만 사용)
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC = _SCRIPT_DIR.parent / "src"


DEFAULT_JUDGE_PROMPT = {
    "name": "single-v1",
    "type": "single",
    "system_prompt": "You are a helpful assistant.",
    "prompt_template": (
        "[Instruction]\n"
        "Please act as an impartial judge and evaluate the quality of the response "
        "provided by an AI assistant to the user question displayed below. Your evaluation "
        "should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
        "and level of detail of the response. Begin your evaluation by providing a short explanation. "
        "Be as objective as possible. After providing your explanation, you must rate the response on a scale "
        "of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n"
        "[Question]\n"
        "{question}\n\n"
        "[The Start of Assistant's Answer]\n"
        "{answer}\n"
        "[The End of Assistant's Answer]"
    ),
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[rating]]",
}

JUDGE_MAX_TOKENS = 512


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _get_question(item: Dict[str, Any]) -> str:
    return (item.get("question_ko") or item.get("question_en") or "").strip()


def _get_reference_answer(item: Dict[str, Any]) -> str:
    return (item.get("answer_ko") or item.get("answer_en") or "").strip()


def _get_generated_answer(item: Dict[str, Any]) -> str:
    return (item.get("prediction") or item.get("generated_answer") or "").strip()


def _parse_judge_response(response_text: str) -> Dict[str, Any]:
    """Judge 응답에서 설명과 [[rating]] 값을 파싱한다."""
    response_text = (response_text or "").strip()
    rating_match = re.search(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", response_text)

    if not rating_match:
        return {
            "judge_rating": None,
            "gpt_score": None,
            "judge_explanation": response_text,
            "judge_raw_response": response_text,
        }

    rating = float(rating_match.group(1))
    if not 1.0 <= rating <= 10.0:
        return {
            "judge_rating": None,
            "gpt_score": None,
            "judge_explanation": response_text[:rating_match.start()].strip(),
            "judge_raw_response": response_text,
        }

    explanation = response_text[:rating_match.start()].strip()
    return {
        "judge_rating": rating,
        "gpt_score": rating,
        "judge_explanation": explanation,
        "judge_raw_response": response_text,
    }


def get_gpt_score(client: OpenAI, question: str, generated_answer: str,
                  model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Judge prompt를 사용하여 생성된 답변을 평가

    Args:
        client: OpenAI 클라이언트
        question: 질문
        generated_answer: 생성된 답변
        model: 사용할 GPT 모델

    Returns:
        judge_rating(1~10), gpt_score(1~10), 설명, 원본 응답
    """
    prompt = DEFAULT_JUDGE_PROMPT["prompt_template"].format(
        question=question,
        answer=generated_answer,
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_JUDGE_PROMPT["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        
        score_text = response.choices[0].message.content.strip()
        parsed = _parse_judge_response(score_text)
        if parsed["gpt_score"] is None:
            print(f"Judge 응답 파싱 실패: {score_text}")
        return parsed
    except Exception as e:
        print(f"GPT 평가 실패: {e}")
        return {
            "judge_rating": None,
            "gpt_score": None,
            "judge_explanation": None,
            "judge_raw_response": None,
        }


def _run_instruct_inference(
    original_jsonl_path: str,
    prediction_jsonl_path: str,
    model_path: str,
    backend_name: str = "qwen",
    base_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    prompt_prefix: Optional[str] = None,
    model=None,
) -> None:
    """원본 JSONL 기준으로 추론 실행 후 prediction JSONL 저장 (backends 사용)"""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from backends import get_backend
    from dataloaders import get_dataloader

    base_dir = Path(base_dir).resolve() if base_dir else None
    loader_kw = {"max_samples": max_samples}
    if base_dir:
        loader_kw["base_dir"] = str(base_dir)
    if prompt_prefix:
        loader_kw["custom_prompt"] = prompt_prefix
    dataloader = get_dataloader("instruct", str(original_jsonl_path), **loader_kw)
    items = list(dataloader)
    if model is None:
        model = get_backend(backend_name, model_path, tensor_parallel_size=tensor_parallel_size)
    use_batch = batch_size > 1 and hasattr(model, "inference_batch")
    if use_batch:
        print(f"배치 추론 사용 (batch_size={batch_size})")

    pred_by_index = {}
    if use_batch:
        processed = []
        for item in items:
            audio_path = item["audio_path"]
            if base_dir and not os.path.isabs(audio_path):
                audio_path = str(base_dir / audio_path)
            if not os.path.exists(audio_path):
                pred_by_index[item["index"]] = ""
                continue
            processed.append({
                "index": item["index"],
                "audio_path": audio_path,
                "prompt": item.get("text_input", item.get("prompt", "")),
                "offset": item.get("offset", 0.0),
                "duration": item.get("duration"),
            })
        num_batches = (len(processed) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc=f"Instruct 배치 (bs={batch_size})"):
            batch = processed[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            inference_list = [
                {"audio_path": it["audio_path"], "prompt": it["prompt"], "offset": it["offset"], "duration": it["duration"]}
                for it in batch
            ]
            try:
                preds = model.inference_batch(inference_list, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"배치 오류: {e}")
                preds = [""] * len(batch)
            for it, pred in zip(batch, preds):
                pred_by_index[it["index"]] = pred or ""
    else:
        for item in tqdm(items, desc="Instruct 추론"):
            audio_path = item["audio_path"]
            prompt = item.get("text_input", item.get("prompt", ""))
            if base_dir and not os.path.isabs(audio_path):
                audio_path = str(base_dir / audio_path)
            if not os.path.exists(audio_path):
                pred_by_index[item["index"]] = ""
                continue
            try:
                pred = model.inference(audio_path, prompt, max_new_tokens=max_new_tokens)
            except Exception as e:
                print(f"오류 [{audio_path}]: {e}")
                pred = ""
            pred_by_index[item["index"]] = pred

    results = [{"index": item["index"], "prediction": pred_by_index.get(item["index"], "")} for item in items]
    Path(prediction_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"추론 결과 저장: {prediction_jsonl_path}")


def evaluate_instruct(
    original_jsonl_path: str,
    output_dir: str,
    prediction_jsonl_path: Optional[str] = None,
    run_inference: bool = False,
    model_path: Optional[str] = None,
    backend_name: str = "qwen",
    base_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    openai_api_key: Optional[str] = None,
    openai_api_key_file: Optional[str] = None,
    gpt_model: str = "gpt-4o-mini",
    max_new_tokens: int = 256,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    prompt_prefix: Optional[str] = None,
    model=None,
) -> Dict:
    """
    Instruct 평가: (선택) 추론 실행 후 추론 결과를 Judge Prompt로 평가
    
    run_inference=True 이면 get_backend(backend, model_path) 로 추론 후 prediction JSONL 생성.
    """
    original_jsonl_path = Path(original_jsonl_path)
    if not original_jsonl_path.is_absolute():
        original_jsonl_path = Path.cwd() / original_jsonl_path
    original_jsonl_path = original_jsonl_path.resolve()

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_inference:
        pred_path = prediction_jsonl_path or str(output_dir / "instruct_predictions.jsonl")
        pred_path = Path(pred_path).resolve()
        _run_instruct_inference(
            str(original_jsonl_path),
            str(pred_path),
            model_path=model_path or "",
            backend_name=backend_name,
            base_dir=base_dir,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            tensor_parallel_size=tensor_parallel_size,
            prompt_prefix=prompt_prefix,
            model=model,
        )
        prediction_jsonl_path = str(pred_path)
    if not prediction_jsonl_path:
        raise ValueError("추론 결과 JSONL이 필요합니다. --prediction_jsonl 경로를 주거나 --run_inference 로 추론을 먼저 실행하세요.")

    prediction_jsonl_path = Path(prediction_jsonl_path)
    if not prediction_jsonl_path.is_absolute():
        prediction_jsonl_path = Path.cwd() / prediction_jsonl_path
    prediction_jsonl_path = prediction_jsonl_path.resolve()

    # OpenAI 클라이언트 초기화 (키: 인자 → 파일 경로면 파일에서 읽기 → 환경변수)
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if api_key and os.path.isfile(api_key.strip()):
        with open(api_key.strip(), "r", encoding="utf-8") as f:
            api_key = f.read().strip()
    if openai_api_key_file:
        path = Path(openai_api_key_file).expanduser().resolve()
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
    if not api_key:
        raise ValueError("OpenAI API 키가 필요합니다. --openai_api_key, --openai_api_key_file 또는 OPENAI_API_KEY 환경변수를 설정해 주세요.")
    client = OpenAI(api_key=api_key)
    
    # 원본 JSONL 로드 (question, answer 포함)
    print(f"원본 JSONL 로드: {original_jsonl_path}")
    original_data = load_jsonl(str(original_jsonl_path))
    original_dict = {item["index"]: item for item in original_data}
    
    # 추론 결과 JSONL 로드 (prediction 포함)
    print(f"추론 결과 JSONL 로드: {prediction_jsonl_path}")
    prediction_data = load_jsonl(str(prediction_jsonl_path))
    prediction_dict = {item["index"]: item for item in prediction_data}
    
    # index로 매칭하여 평가
    all_indices = set(original_dict.keys()) & set(prediction_dict.keys())
    if not all_indices:
        raise ValueError("매칭되는 index가 없습니다. 원본 JSONL과 추론 결과 JSONL의 index를 확인해 주세요.")

    eval_count = min(len(prediction_data), max_samples) if max_samples else len(prediction_data)
    print(f"평가할 샘플 수: {eval_count}")

    results = []
    scores = []
    judge_ratings = []
    start_time = time.time()

    for idx, prediction_item in enumerate(tqdm(prediction_data, desc="Judge 평가")):
        index = prediction_item.get("index")
        updated_item = dict(prediction_item)
        original_item = original_dict.get(index)
        gpt_score = updated_item.get("gpt_score")
        computed_gpt_score = None
        computed_judge_rating = None

        if max_samples is not None and idx >= max_samples:
            results.append(updated_item)
            continue

        if original_item:
            question = _get_question(original_item)
            generated_answer = _get_generated_answer(prediction_item)

            if question and generated_answer:
                judge_result = get_gpt_score(client, question, generated_answer, model=gpt_model)
                computed_gpt_score = judge_result["gpt_score"]
                computed_judge_rating = judge_result["judge_rating"]
                gpt_score = computed_gpt_score

        if computed_gpt_score is not None:
            scores.append(computed_gpt_score)
        if computed_judge_rating is not None:
            judge_ratings.append(computed_judge_rating)

        updated_item["gpt_score"] = gpt_score
        results.append(updated_item)
    
    elapsed = time.time() - start_time
    total = len(results)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    avg_judge_rating = sum(judge_ratings) / len(judge_ratings) if judge_ratings else 0.0
    
    # prediction JSONL에 gpt_score만 덮어쓴다.
    out_jsonl = prediction_jsonl_path
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    summary = {
        "avg_gpt_score": avg_score,
        "avg_judge_rating": avg_judge_rating,
        "total": total,
        "scored": len(scores),
        "elapsed_seconds": elapsed,
        "predictions_file": str(out_jsonl),
        "gpt_model": gpt_model,
        "source_original_jsonl": str(original_jsonl_path),
        "source_prediction_jsonl": str(prediction_jsonl_path),
        "judge_prompt_name": DEFAULT_JUDGE_PROMPT["name"],
        "judge_output_format": DEFAULT_JUDGE_PROMPT["output_format"],
    }
    with open(output_dir / "instruct_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("Instruct 평가 결과 (Judge Prompt 기반)")
    print("=" * 60)
    print(f"평균 Judge Rating: {avg_judge_rating:.4f} / 10 ({len(judge_ratings)}/{total})")
    print(f"평균 GPT Score: {avg_score:.4f} / 10 ({len(scores)}/{total})")
    print(f"소요 시간: {elapsed:.1f}초")
    print(f"결과 파일(덮어씀): {out_jsonl}")
    print("=" * 60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Instruct 평가 (모델 무관: --run_inference 시 backends로 추론 후 Judge Prompt 평가)"
    )
    parser.add_argument("--original_jsonl", type=str, required=True, help="question과 gt(answer_ko/answer_en)가 들어있는 원본 Instruct JSONL")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="평가 요약 저장 디렉토리")
    parser.add_argument("--prediction_jsonl", type=str, default=None, help="model generation prediction JSONL (index, prediction 또는 generated_answer)")
    parser.add_argument("--run_inference", action="store_true", help="먼저 backends로 추론 실행 후 평가")
    parser.add_argument("--model_path", type=str, default=None, help="--run_inference 시 사용할 모델 경로")
    parser.add_argument("--backend", type=str, default="qwen", help="--run_inference 시 백엔드 (backends 등록명)")
    parser.add_argument("--base_dir", type=str, default=None, help="오디오 상대 경로 기준 디렉토리")
    parser.add_argument("--max_samples", type=int, default=None, help="최대 샘플 수")
    parser.add_argument("--batch_size", type=int, default=1, help="추론 배치 크기 (백엔드가 inference_batch 지원 시)")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1, help="GPU 수 (vLLM 백엔드 사용 시)")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API 키 (또는 키가 담긴 파일 경로)")
    parser.add_argument("--openai_api_key_file", type=str, default=None, help="OpenAI API 키가 담긴 파일 경로 (txt 등)")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini", help="Judge 평가 모델")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="프롬프트 설정 YAML 파일 경로. 지정 시 instruct 섹션의 모든 프롬프트로 각각 평가 후 최적 결과 저장")
    parser.add_argument("--prompt-name", type=str, default=None,
                        help="prompt-file 사용 시 해당 name 만 실행. 없으면 첫 항목만.")
    args = parser.parse_args()

    if args.prompt_file:
        import yaml
        import shutil
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_cfg = yaml.safe_load(f)
        inst_prompts = prompt_cfg.get('instruct', [])
        if args.prompt_name:
            inst_prompts = [p for p in inst_prompts if p.get("name") == args.prompt_name]
            if not inst_prompts:
                inst_prompts = prompt_cfg.get('instruct', [])[:1]
                print(f"[prompt-file] --prompt-name={args.prompt_name} 없음 → 첫 프롬프트만 사용")
        if not inst_prompts:
            raise ValueError("prompt file에 'instruct' 섹션이 없거나 비어 있습니다.")
        out_base = Path(args.output_dir).resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        _model = None
        if args.run_inference:
            if str(_SRC) not in sys.path:
                sys.path.insert(0, str(_SRC))
            from backends import get_backend
            _model = get_backend(args.backend, args.model_path or "", tensor_parallel_size=args.tensor_parallel_size)
        summaries = []
        for i, p in enumerate(inst_prompts):
            name = p['name']
            prompt_prefix = p.get('prompt_prefix')
            out_dir = str(out_base / f'prompt_{name}')
            print(f"\n[{i+1}/{len(inst_prompts)}] 프롬프트: {name}")
            summary = evaluate_instruct(
                original_jsonl_path=args.original_jsonl,
                output_dir=out_dir,
                prediction_jsonl_path=args.prediction_jsonl,
                run_inference=args.run_inference,
                model_path=args.model_path,
                backend_name=args.backend,
                base_dir=args.base_dir,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                openai_api_key=args.openai_api_key,
                openai_api_key_file=args.openai_api_key_file,
                gpt_model=args.gpt_model,
                prompt_prefix=prompt_prefix,
                model=_model,
            )
            summaries.append({
                'name': name,
                'prompt_prefix': prompt_prefix,
                'avg_gpt_score': summary['avg_gpt_score'],
            })
        best = max(summaries, key=lambda x: x['avg_gpt_score'])
        best_dir = out_base / f"prompt_{best['name']}"
        for suf in ('instruct_predictions.jsonl', 'instruct_summary.json'):
            src = best_dir / suf
            dst = out_base / suf
            if src.is_file():
                shutil.copy2(src, dst)
        comparison = {'prompts': summaries, 'best': best['name']}
        cmp_path = out_base / 'prompt_comparison.json'
        with open(cmp_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print('\n' + '=' * 60)
        print('프롬프트 비교 (GPT Score 1~10)')
        print('=' * 60)
        for s in summaries:
            marker = '  ← best' if s['name'] == best['name'] else ''
            print(f"  {s['name']}: {s['avg_gpt_score']:.4f}{marker}")
        print(f"  채택: {best['name']}")
        print(f"  비교 결과: {cmp_path}")
        print('=' * 60)
    else:
        evaluate_instruct(
            original_jsonl_path=args.original_jsonl,
            output_dir=args.output_dir,
            prediction_jsonl_path=args.prediction_jsonl,
            run_inference=args.run_inference,
            model_path=args.model_path,
            backend_name=args.backend,
            base_dir=args.base_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            openai_api_key=args.openai_api_key,
            openai_api_key_file=args.openai_api_key_file,
            gpt_model=args.gpt_model,
        )


if __name__ == "__main__":
    main()

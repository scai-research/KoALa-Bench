#!/usr/bin/env python3
"""
한국어 ASR 평가 스크립트 (모델 무관, backends 사용)

- JSONL에서 오디오 경로와 ground truth 로드
- backends.get_backend(backend, model_path) 로 추론 (Qwen/Llama 등)
- 정규화 후 CER 계산, 결과 저장
"""
import os
import sys
import json
import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Ko-Speech-Eval/src 로 경로 추가
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC = _SCRIPT_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from backends import get_backend, list_backends

from korean_normalizer import calculate_cer


def _unwrap_prediction(prediction):
    """백엔드가 (pred, note) 튜플을 반환하면 풀어서 (pred, note), 아니면 (pred, None)."""
    if isinstance(prediction, (list, tuple)) and len(prediction) == 2:
        return prediction[0] or "", prediction[1]
    return (prediction or "", None)


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """JSONL 파일 로드"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_asr(
    jsonl_path: str,
    output_dir: str,
    model_name: Optional[str] = None,
    backend_name: str = "qwen",
    prompt: str = "이 오디오를 한국어로 받아쓰기해 주세요.",
    max_samples: int = None,
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    model=None,
) -> Dict:
    """
    ASR 평가 수행 (모델 무관, backends 사용)
    
    정규화 규칙:
    - 구두점 제거 (한글 + 숫자만 유지)
    - 공백 제거
    
    Args:
        jsonl_path: 입력 JSONL 파일 경로
        output_dir: 결과 저장 디렉토리
        model_name: 모델 경로
        backend_name: 추론 백엔드 (backends에 등록된 이름)
        prompt: ASR 프롬프트
        max_samples: 최대 샘플 수 (None이면 전체)
    
    Returns:
        평가 결과 딕셔너리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n데이터 로딩: {jsonl_path}")
    data = load_jsonl(jsonl_path)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"총 {len(data)}개 샘플")
    
    if model is None:
        model = get_backend(backend_name, model_name)
    if hasattr(model, "model_path"):
        print(f"[모델] {model.model_path}")
    use_batch = batch_size > 1 and hasattr(model, 'inference_batch')
    if use_batch:
        print(f"배치 추론 사용 (batch_size={batch_size})")
    
    results = []
    total_cer = 0.0
    total_edit_distance = 0
    total_ref_length = 0
    
    # 추론 시작 시간
    start_time = time.time()
    
    if use_batch:
        # 배치 단위로 추론
        valid_items = []
        for i, item in enumerate(data):
            audio_path = item.get('raw', '')
            if not os.path.exists(audio_path):
                print(f"경고: 오디오 파일 없음 - {audio_path}")
                continue
            valid_items.append({
                "index": item.get('index', str(i)),
                "audio_path": audio_path,
                "ground_truth": item.get('ground_truth', item.get('question_ko', '')),
                "offset": item.get('offset', 0.0),
                "duration": item.get('duration'),
            })
        num_batches = (len(valid_items) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc=f"ASR 배치 (bs={batch_size})"):
            start_i = batch_idx * batch_size
            batch_items = valid_items[start_i:start_i + batch_size]
            inference_list = [
                {
                    "audio_path": it["audio_path"],
                    "prompt": prompt,
                    "offset": it["offset"],
                    "duration": it["duration"],
                }
                for it in batch_items
            ]
            try:
                predictions = model.inference_batch(inference_list, max_new_tokens=256)
            except Exception as e:
                print(f"배치 오류: {e}")
                err_msg = str(e).replace('"', "'")[:200]
                predictions = [("", f"batch_error: {err_msg}")] * len(batch_items)
            for it, prediction in zip(batch_items, predictions):
                pred_str, note = _unwrap_prediction(prediction)
                cer, edit_dist, ref_len, gt_normalized, pred_normalized = calculate_cer(
                    it["ground_truth"], pred_str
                )
                total_edit_distance += edit_dist
                total_ref_length += ref_len
                row = {
                    "index": it["index"],
                    "audio_path": it["audio_path"],
                    "ground_truth": it["ground_truth"],
                    "prediction": pred_str,
                    "pred_normalized": pred_normalized,
                    "gt_normalized": gt_normalized,
                    "cer": cer,
                    "edit_distance": edit_dist,
                    "ref_length": ref_len
                }
                if note:
                    row["note"] = note
                results.append(row)
                if (len(results)) % 100 == 0:
                    cur_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0
                    print(f"\n[{len(results)}/{len(valid_items)}] 현재 CER: {cur_cer:.4f}")
    else:
        for i, item in enumerate(tqdm(data, desc="ASR 평가")):
            audio_path = item.get('raw', '')
            ground_truth = item.get('ground_truth', item.get('question_ko', ''))
            index = item.get('index', str(i))
            if not os.path.exists(audio_path):
                print(f"경고: 오디오 파일 없음 - {audio_path}")
                _cer, _ed, ref_len, gt_norm, _ = calculate_cer(ground_truth, "")
                results.append({
                    "index": index,
                    "audio_path": audio_path,
                    "ground_truth": ground_truth,
                    "prediction": "",
                    "gt_normalized": gt_norm,
                    "pred_normalized": "",
                    "cer": _cer,
                    "edit_distance": _ed,
                    "ref_length": ref_len,
                    "note": "audio_missing",
                })
                total_edit_distance += _ed
                total_ref_length += ref_len
                continue
            try:
                prediction = model.inference(audio_path, prompt, max_new_tokens=256)
            except Exception as e:
                print(f"오류 [{audio_path}]: {e}")
                err_msg = str(e).replace('"', "'")[:200]
                prediction = ("", f"gpt_error: {err_msg}")
            pred_str, note = _unwrap_prediction(prediction)
            cer, edit_dist, ref_len, gt_normalized, pred_normalized = calculate_cer(
                ground_truth, pred_str
            )
            total_edit_distance += edit_dist
            total_ref_length += ref_len
            result = {
                "index": index,
                "audio_path": audio_path,
                "ground_truth": ground_truth,
                "prediction": pred_str,
                "gt_normalized": gt_normalized,
                "pred_normalized": pred_normalized,
                "cer": cer,
                "edit_distance": edit_dist,
                "ref_length": ref_len
            }
            if note:
                result["note"] = note
            results.append(result)
            if (i + 1) % 100 == 0:
                current_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0
                print(f"\n[{i+1}/{len(data)}] 현재 CER: {current_cer:.4f}")
    
    # 전체 CER 계산
    final_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0
    elapsed_time = time.time() - start_time
    
    # 결과 요약
    summary = {
        "dataset": os.path.basename(jsonl_path),
        "model": model_name,
        "total_samples": len(results),
        "total_cer": final_cer,
        "total_edit_distance": total_edit_distance,
        "total_ref_length": total_ref_length,
        "normalization": "구두점 제거 + 공백 제거",
        "prompt": prompt,
        "elapsed_time_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # 결과 저장
    dataset_name = Path(jsonl_path).stem
    os.makedirs(output_dir, exist_ok=True)

    # 상세 결과 저장
    detail_path = os.path.join(output_dir, f"{dataset_name}_results.jsonl")
    with open(detail_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 요약 저장
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 결과 출력
    print("\n" + "="*60)
    print(f"평가 완료: {dataset_name}")
    print("="*60)
    print(f"총 샘플: {len(results)}")
    print(f"CER: {final_cer:.4f} ({final_cer*100:.2f}%)")
    print(f"총 편집거리: {total_edit_distance}")
    print(f"총 참조 길이: {total_ref_length}")
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/len(results):.2f}초/샘플)")
    print(f"정규화: 구두점 제거 + 공백 제거")
    print(f"\n결과 저장:")
    print(f"  상세: {detail_path}")
    print(f"  요약: {summary_path}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Qwen2-Audio를 이용한 한국어 ASR 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 데이터셋 평가
  python run_asr_evaluation.py \\
    --input ./output/clovacall_test.jsonl \\
    --output-dir ./results \\
    --normalize --strict

  # 샘플 수 제한 테스트
  python run_asr_evaluation.py \\
    --input ./output/clovacall_test.jsonl \\
    --output-dir ./results \\
    --max-samples 10
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='입력 JSONL 파일 경로')
    parser.add_argument('--output-dir', '-o', default='./results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--model', '-m', default=None,
                        help='비우면 백엔드별 기본 모델 사용 (HuggingFace cache 또는 다운로드)')
    parser.add_argument('--backend', '-b', default='qwen', choices=list_backends(),
                        help='추론 백엔드 (모델별 backends 등록명)')
    parser.add_argument('--prompt', '-p',
                        default='이 오디오를 한국어로 받아쓰기해 주세요.',
                        help='ASR 프롬프트 (한국어)')
    parser.add_argument('--prompt-en', type=str, default=None,
                        help='영어 프롬프트. 주면 한국어/영어 각각 한 번씩 돌려서 전체 CER 비교 후 더 좋은 쪽을 최종 결과로 저장')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='최대 샘플 수 (테스트용)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='배치 크기 (백엔드가 inference_batch 지원 시 사용, 기본 1)')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1,
                        help='GPU 수 (vLLM 백엔드 사용 시)')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='프롬프트 설정 YAML 파일 경로. 지정 시 asr 섹션의 모든 프롬프트로 각각 평가 후 최적 결과 저장')
    parser.add_argument('--prompt-name', type=str, default=None,
                        help='prompt-file 사용 시 해당 name 만 실행. 없으면 첫 항목만.')
    
    args = parser.parse_args()

    if args.prompt_file:
        import yaml
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_cfg = yaml.safe_load(f)
        asr_prompts = prompt_cfg.get('asr', [])
        if args.prompt_name:
            asr_prompts = [p for p in asr_prompts if p.get('name') == args.prompt_name]
            if not asr_prompts:
                asr_prompts = prompt_cfg.get('asr', [])[:1]
                print(f"[prompt-file] --prompt-name={args.prompt_name} 없음 → 첫 프롬프트만 사용")
        if not asr_prompts:
            raise ValueError("prompt file에 'asr' 섹션이 없거나 비어 있습니다.")
        dataset_name = Path(args.input).stem
        os.makedirs(args.output_dir, exist_ok=True)
        model = get_backend(args.backend, args.model, tensor_parallel_size=args.tensor_parallel_size)
        summaries = []
        for i, p in enumerate(asr_prompts):
            name = p['name']
            prompt = p['prompt']
            out_dir = os.path.join(args.output_dir, f'prompt_{name}')
            print(f"\n[{i+1}/{len(asr_prompts)}] 프롬프트: {name}")
            summary = evaluate_asr(
                jsonl_path=args.input,
                output_dir=out_dir,
                model_name=args.model,
                backend_name=args.backend,
                prompt=prompt,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                model=model,
            )
            summaries.append({'name': name, 'prompt': prompt, 'total_cer': summary['total_cer']})
        best = min(summaries, key=lambda x: x['total_cer'])
        best_dir = os.path.join(args.output_dir, f"prompt_{best['name']}")
        for suf in ('_results.jsonl', '_summary.json'):
            src = os.path.join(best_dir, f'{dataset_name}{suf}')
            dst = os.path.join(args.output_dir, f'{dataset_name}{suf}')
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        comparison = {'prompts': summaries, 'best': best['name']}
        cmp_path = os.path.join(args.output_dir, 'prompt_comparison.json')
        with open(cmp_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print('\n' + '=' * 60)
        print('프롬프트 비교 (전체 CER)')
        print('=' * 60)
        for s in summaries:
            marker = '  ← best' if s['name'] == best['name'] else ''
            print(f"  {s['name']}: {s['total_cer']:.4f}{marker}")
        print(f"  채택: {best['name']}")
        print(f"  비교 결과: {cmp_path}")
        print('=' * 60)
    elif args.prompt_en:
        # 한국어/영어 각각 한 번씩 실행 후 전체 CER 비교, 더 좋은 쪽을 최종 결과로
        dataset_name = Path(args.input).stem
        out_ko = os.path.join(args.output_dir, 'prompt_ko')
        out_en = os.path.join(args.output_dir, 'prompt_en')
        os.makedirs(args.output_dir, exist_ok=True)
        print("\n[1/2] 한국어 프롬프트로 평가...")
        summary_ko = evaluate_asr(
            jsonl_path=args.input,
            output_dir=out_ko,
            model_name=args.model,
            backend_name=args.backend,
            prompt=args.prompt,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        print("\n[2/2] 영어 프롬프트로 평가...")
        summary_en = evaluate_asr(
            jsonl_path=args.input,
            output_dir=out_en,
            model_name=args.model,
            backend_name=args.backend,
            prompt=args.prompt_en,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        cer_ko = summary_ko["total_cer"]
        cer_en = summary_en["total_cer"]
        better = "ko" if cer_ko <= cer_en else "en"
        better_dir = out_ko if better == "ko" else out_en
        for suf in ("_results.jsonl", "_summary.json"):
            src = os.path.join(better_dir, f"{dataset_name}{suf}")
            dst = os.path.join(args.output_dir, f"{dataset_name}{suf}")
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        comparison = {
            "prompt_ko": {"prompt": args.prompt, "total_cer": cer_ko},
            "prompt_en": {"prompt": args.prompt_en, "total_cer": cer_en},
            "better": better,
        }
        cmp_path = os.path.join(args.output_dir, "prompt_comparison.json")
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 60)
        print("프롬프트 비교 (전체 CER)")
        print("=" * 60)
        print(f"  한국어: {cer_ko:.4f}")
        print(f"  영어:   {cer_en:.4f}")
        print(f"  채택:   {better} (위 결과를 최종 저장)")
        print(f"  비교 결과: {cmp_path}")
        print("=" * 60)
    else:
        evaluate_asr(
            jsonl_path=args.input,
            output_dir=args.output_dir,
            model_name=args.model,
            backend_name=args.backend,
            prompt=args.prompt,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )


if __name__ == '__main__':
    main()

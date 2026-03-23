#!/usr/bin/env python3
"""
한국어 Translation 평가 스크립트 (모델 무관, backends 사용)

- JSONL에서 오디오 경로와 ground truth 로드
- backends.get_backend(backend, model_path) 로 추론 (Qwen/Llama 등)
- BLEU, METEOR, BERTScore 계산, 결과 저장
"""

import os
import sys
import json
import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

import numpy as np

# Ko-Speech-Eval/src 로 경로 추가
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC = _SCRIPT_DIR.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from backends import get_backend, list_backends

from korean_normalizer import (
    calculate_translation_metrics,
    calculate_corpus_bleu,
    calculate_bertscore,
    SACREBLEU_AVAILABLE,
    NLTK_AVAILABLE,
    BERTSCORE_AVAILABLE
)


def load_jsonl(jsonl_path: str) -> List[Dict]:
    """JSONL 파일 로드"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_translation(
    jsonl_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    backend_name: str = "qwen",
    prompt: str = "이 오디오를 한국어로 번역해 주세요.",
    max_samples: int = None,
    tokenize_method: str = "character",
    gt_field: str = "question_ko",
    batch_size: int = 1,
    tensor_parallel_size: int = 1,
    model=None,
) -> Dict:
    """
    Translation 평가 수행 (모델 무관, backends 사용) — BLEU, METEOR, BERTScore
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n데이터 로딩: {jsonl_path}")
    data = load_jsonl(jsonl_path)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"총 {len(data)}개 샘플")
    print(f"평가 라이브러리 - sacrebleu: {SACREBLEU_AVAILABLE}, NLTK: {NLTK_AVAILABLE}, BERTScore: {BERTSCORE_AVAILABLE}")
    print(f"토크나이징 방법: {tokenize_method}")
    
    if model is None:
        model = get_backend(backend_name, model_name, tensor_parallel_size=tensor_parallel_size)
    
    # 결과 저장
    results = []
    all_references = []
    all_hypotheses = []
    total_bleu = 0.0
    total_meteor = 0.0
    
    start_time = time.time()
    
    use_batch = batch_size > 1 and hasattr(model, 'inference_batch')
    if use_batch:
        items = []
        ground_truths = []
        indices = []
        for i, item in enumerate(data):
            audio_path = item.get('raw', '')
            if not os.path.exists(audio_path):
                print(f"[SKIP] 오디오 파일 없음: {audio_path}")
                continue
            items.append({
                'audio_path': audio_path,
                'offset': item.get('offset', 0.0),
                'duration': item.get('duration', None)
            })
            ground_truths.append(item.get(gt_field, ''))
            indices.append(item.get('index', str(i)))
        print(f"\n번역 시작 배치 (batch_size={batch_size}, {len(items)}개 샘플)")
        predictions = []
        num_batches = (len(items) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="번역"):
            batch = items[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            inference_list = [
                {"audio_path": it["audio_path"], "prompt": prompt, "offset": it["offset"], "duration": it["duration"]}
                for it in batch
            ]
            try:
                preds = model.inference_batch(inference_list, max_new_tokens=256)
            except Exception as e:
                print(f"배치 오류: {e}")
                preds = [""] * len(batch)
            predictions.extend(preds)
        for i, (prediction, ground_truth, index) in enumerate(zip(predictions, ground_truths, indices)):
            metrics = calculate_translation_metrics(ground_truth, prediction, tokenize_method)
            
            total_bleu += metrics['bleu']
            total_meteor += metrics['meteor']
            
            all_references.append(ground_truth)
            all_hypotheses.append(prediction)
            
            result = {
                "index": index,
                "audio_path": items[i]['audio_path'],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "bleu": metrics['bleu'],
                "bleu1": metrics['bleu1'],
                "bleu2": metrics['bleu2'],
                "bleu3": metrics['bleu3'],
                "bleu4": metrics['bleu4'],
                "meteor": metrics['meteor']
            }
            results.append(result)
    else:
        # 순차 처리
        for i, item in enumerate(tqdm(data, desc="Translation 평가")):
            audio_path = item.get('raw', '')
            ground_truth = item.get(gt_field, '')
            index = item.get('index', str(i))
            offset = item.get('offset', 0.0)
            duration = item.get('duration', None)
            
            if not os.path.exists(audio_path):
                print(f"[SKIP] 오디오 파일 없음: {audio_path}")
                continue
            
            try:
                prediction = model.inference(
                    audio_path, prompt, max_new_tokens=256,
                    offset=offset, duration=duration
                )
            except Exception as e:
                print(f"[ERROR] 번역 오류 [{audio_path}]: {e}")
                prediction = ""
            
            metrics = calculate_translation_metrics(ground_truth, prediction, tokenize_method)
            
            total_bleu += metrics['bleu']
            total_meteor += metrics['meteor']
            
            all_references.append(ground_truth)
            all_hypotheses.append(prediction)
            
            result = {
                "index": index,
                "audio_path": audio_path,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "bleu": metrics['bleu'],
                "bleu1": metrics['bleu1'],
                "bleu2": metrics['bleu2'],
                "bleu3": metrics['bleu3'],
                "bleu4": metrics['bleu4'],
                "meteor": metrics['meteor']
            }
            results.append(result)
            
            if (i + 1) % 100 == 0:
                avg_bleu = total_bleu / len(results)
                avg_meteor = total_meteor / len(results)
                print(f"\n[{i+1}/{len(data)}] 현재 평균 BLEU: {avg_bleu:.2f}, METEOR: {avg_meteor:.2f}")
    
    # Corpus BLEU 계산
    corpus_bleu = calculate_corpus_bleu([[r] for r in all_references], all_hypotheses, tokenize_method)
    
    avg_meteor = total_meteor / len(results) if results else 0.0
    avg_bleu = total_bleu / len(results) if results else 0.0
    
    # BERTScore 계산 (배치로 한 번에 - 항상 실행)
    avg_bertscore_f1 = 0.0
    if BERTSCORE_AVAILABLE and results:
        print("\nBERTScore 계산 중...")
        bert_scores = calculate_bertscore(all_references, all_hypotheses, lang="ko")
        
        # 개별 결과에 추가
        for i, result in enumerate(results):
            result['bertscore_f1'] = bert_scores['f1'][i] * 100
            result['bertscore_precision'] = bert_scores['precision'][i] * 100
            result['bertscore_recall'] = bert_scores['recall'][i] * 100
        
        avg_bertscore_f1 = sum(bert_scores['f1']) / len(bert_scores['f1']) * 100 if bert_scores['f1'] else 0.0
        print(f"BERTScore 계산 완료 (평균 F1: {avg_bertscore_f1:.2f})")
    
    elapsed_time = time.time() - start_time
    
    # 결과 요약
    summary = {
        "dataset": os.path.basename(jsonl_path),
        "model": model_name,
        "total_samples": len(results),
        "tokenize_method": tokenize_method,
        "corpus_bleu": corpus_bleu['bleu'],
        "corpus_bleu1": corpus_bleu.get('bleu1', 0.0),
        "corpus_bleu2": corpus_bleu.get('bleu2', 0.0),
        "corpus_bleu3": corpus_bleu.get('bleu3', 0.0),
        "corpus_bleu4": corpus_bleu.get('bleu4', 0.0),
        "avg_bleu": avg_bleu,
        "avg_meteor": avg_meteor,
        "avg_bertscore_f1": avg_bertscore_f1,
        "prompt": prompt,
        "elapsed_time_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # 결과 저장
    dataset_name = Path(jsonl_path).stem
    
    detail_path = os.path.join(output_dir, f"{dataset_name}_translation_results.jsonl")
    with open(detail_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    summary_path = os.path.join(output_dir, f"{dataset_name}_translation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 결과 출력
    print("\n" + "="*60)
    print(f"Translation 평가 완료: {dataset_name}")
    print("="*60)
    print(f"총 샘플: {len(results)}")
    print(f"토크나이징: {tokenize_method}")
    print("-"*60)
    print("BLEU 점수:")
    print(f"  Corpus BLEU: {corpus_bleu['bleu']:.2f}")
    print(f"  BLEU-1: {corpus_bleu.get('bleu1', 0.0):.2f}")
    print(f"  BLEU-2: {corpus_bleu.get('bleu2', 0.0):.2f}")
    print(f"  BLEU-3: {corpus_bleu.get('bleu3', 0.0):.2f}")
    print(f"  BLEU-4: {corpus_bleu.get('bleu4', 0.0):.2f}")
    print(f"  Average BLEU: {avg_bleu:.2f}")
    print("-"*60)
    print(f"METEOR 점수: {avg_meteor:.2f}")
    print("-"*60)
    print(f"BERTScore F1: {avg_bertscore_f1:.2f}")
    print("-"*60)
    print(f"소요 시간: {elapsed_time:.1f}초 ({elapsed_time/len(results):.2f}초/샘플)")
    print(f"\n결과 저장:")
    print(f"  상세: {detail_path}")
    print(f"  요약: {summary_path}")
    print("="*60)
    
    # 점수 요약 테이블 출력
    print("\n" + "="*60)
    print("=== ETRI EnKoST 점수 요약 ===")
    print("="*60)
    print(f"{'Dataset':<35} {'BLEU':>10} {'METEOR':>10} {'BERTScore':>10}")
    print("-"*60)
    print(f"{dataset_name:<35} {corpus_bleu['bleu']:>10.2f} {avg_meteor:>10.2f} {avg_bertscore_f1:>10.2f}")
    print("-"*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Qwen2-Audio를 이용한 한국어 Translation 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 데이터셋 평가
  python run_translation_evaluation.py \\
    --input ./output/etri_tst-COMMON_processed.jsonl \\
    --output-dir ./results \\
    --gt-field answer_ko

  # 배치 처리 (더 빠름)
  python run_translation_evaluation.py \\
    --input ./output/etri_tst-COMMON_processed.jsonl \\
    --output-dir ./results \\
    --gt-field answer_ko \\
    --batch-size 4

  # 샘플 수 제한 테스트
  python run_translation_evaluation.py \\
    --input ./output/etri_tst-COMMON_processed.jsonl \\
    --output-dir ./results \\
    --gt-field answer_ko \\
    --max-samples 10
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='입력 JSONL 파일 경로')
    parser.add_argument('--output-dir', '-o', default='./results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--model', '-m', default=None,
                        help='비우면 백엔드별 기본 모델 사용 (HuggingFace cache 또는 다운로드)')
    parser.add_argument('--backend', '-B', default='qwen', choices=list_backends(),
                        help='추론 백엔드 (모델별 backends 등록명)')
    parser.add_argument('--prompt', '-p',
                        default='이 오디오를 한국어로 번역해 주세요.',
                        help='Translation 프롬프트 (한국어)')
    parser.add_argument('--prompt-en', type=str, default=None,
                        help='영어 프롬프트. 주면 한국어/영어 각각 한 번씩 돌려서 전체 BLEU 비교 후 더 좋은 쪽을 최종 결과로 저장')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='최대 샘플 수 (테스트용)')
    parser.add_argument('--tokenize', '-t', 
                        choices=['character', 'space', 'morpheme'],
                        default='character',
                        help='토크나이징 방법 (기본: character)')
    parser.add_argument('--gt-field', default='question_ko',
                        help='Ground truth 필드명 (기본: question_ko, ETRI는 answer_ko)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='배치 크기 (기본: 1)')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1,
                        help='GPU 수 (vLLM 백엔드용, 기본: 1)')
    parser.add_argument('--prompt-file', type=str, default=None,
                        help='프롬프트 설정 YAML 파일 경로. 지정 시 translation 섹션의 모든 프롬프트로 각각 평가 후 최적 결과 저장')
    parser.add_argument('--prompt-name', type=str, default=None,
                        help='prompt-file 사용 시 해당 name 만 실행. 없으면 첫 항목만.')
    
    args = parser.parse_args()

    if args.prompt_file:
        import yaml
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_cfg = yaml.safe_load(f)
        tr_prompts = prompt_cfg.get('translation', [])
        if args.prompt_name:
            tr_prompts = [p for p in tr_prompts if p.get('name') == args.prompt_name]
            if not tr_prompts:
                tr_prompts = prompt_cfg.get('translation', [])[:1]
                print(f"[prompt-file] --prompt-name={args.prompt_name} 없음 → 첫 프롬프트만 사용")
        if not tr_prompts:
            raise ValueError("prompt file에 'translation' 섹션이 없거나 비어 있습니다.")
        dataset_name = Path(args.input).stem
        os.makedirs(args.output_dir, exist_ok=True)
        _model = get_backend(args.backend, args.model, tensor_parallel_size=args.tensor_parallel_size)
        summaries = []
        for i, p in enumerate(tr_prompts):
            name = p['name']
            prompt = p['prompt']
            out_dir = os.path.join(args.output_dir, f'prompt_{name}')
            print(f"\n[{i+1}/{len(tr_prompts)}] 프롬프트: {name}")
            summary = evaluate_translation(
                jsonl_path=args.input,
                output_dir=out_dir,
                model_name=args.model,
                backend_name=args.backend,
                prompt=prompt,
                max_samples=args.max_samples,
                tokenize_method=args.tokenize,
                gt_field=args.gt_field,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                model=_model,
            )
            summaries.append({'name': name, 'prompt': prompt, 'corpus_bleu': summary['corpus_bleu']})
        best = max(summaries, key=lambda x: x['corpus_bleu'])
        best_dir = os.path.join(args.output_dir, f"prompt_{best['name']}")
        for suf in ('_translation_results.jsonl', '_translation_summary.json'):
            src = os.path.join(best_dir, f'{dataset_name}{suf}')
            dst = os.path.join(args.output_dir, f'{dataset_name}{suf}')
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        comparison = {'prompts': summaries, 'best': best['name']}
        cmp_path = os.path.join(args.output_dir, 'prompt_comparison.json')
        with open(cmp_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print('\n' + '=' * 60)
        print('프롬프트 비교 (Corpus BLEU)')
        print('=' * 60)
        for s in summaries:
            marker = '  ← best' if s['name'] == best['name'] else ''
            print(f"  {s['name']}: {s['corpus_bleu']:.2f}{marker}")
        print(f"  채택: {best['name']}")
        print(f"  비교 결과: {cmp_path}")
        print('=' * 60)
    elif args.prompt_en:
        # 한국어/영어 각각 한 번씩 실행 후 전체 BLEU 비교, 더 좋은 쪽을 최종 결과로
        dataset_name = Path(args.input).stem
        out_ko = os.path.join(args.output_dir, 'prompt_ko')
        out_en = os.path.join(args.output_dir, 'prompt_en')
        os.makedirs(args.output_dir, exist_ok=True)
        print("\n[1/2] 한국어 프롬프트로 평가...")
        summary_ko = evaluate_translation(
            jsonl_path=args.input,
            output_dir=out_ko,
            model_name=args.model,
            backend_name=args.backend,
            prompt=args.prompt,
            max_samples=args.max_samples,
            tokenize_method=args.tokenize,
            gt_field=args.gt_field,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        print("\n[2/2] 영어 프롬프트로 평가...")
        summary_en = evaluate_translation(
            jsonl_path=args.input,
            output_dir=out_en,
            model_name=args.model,
            backend_name=args.backend,
            prompt=args.prompt_en,
            max_samples=args.max_samples,
            tokenize_method=args.tokenize,
            gt_field=args.gt_field,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        bleu_ko = summary_ko["corpus_bleu"]
        bleu_en = summary_en["corpus_bleu"]
        better = "ko" if bleu_ko >= bleu_en else "en"
        better_dir = out_ko if better == "ko" else out_en
        for suf in ("_translation_results.jsonl", "_translation_summary.json"):
            src = os.path.join(better_dir, f"{dataset_name}{suf}")
            dst = os.path.join(args.output_dir, f"{dataset_name}{suf}")
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        comparison = {
            "prompt_ko": {"prompt": args.prompt, "corpus_bleu": bleu_ko},
            "prompt_en": {"prompt": args.prompt_en, "corpus_bleu": bleu_en},
            "better": better,
        }
        cmp_path = os.path.join(args.output_dir, "prompt_comparison.json")
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 60)
        print("프롬프트 비교 (전체 BLEU)")
        print("=" * 60)
        print(f"  한국어: {bleu_ko:.2f}")
        print(f"  영어:   {bleu_en:.2f}")
        print(f"  채택:   {better} (위 결과를 최종 저장)")
        print(f"  비교 결과: {cmp_path}")
        print("=" * 60)
    else:
        evaluate_translation(
            jsonl_path=args.input,
            output_dir=args.output_dir,
            model_name=args.model,
            backend_name=args.backend,
            prompt=args.prompt,
            max_samples=args.max_samples,
            tokenize_method=args.tokenize,
            gt_field=args.gt_field,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size
        )


if __name__ == '__main__':
    main()

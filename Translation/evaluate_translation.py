#!/usr/bin/env python3
"""
Translation 평가 스크립트

추론 결과 JSONL 파일을 읽어서 BLEU와 METEOR 점수를 계산합니다.

입력:
- 추론 결과 JSONL (prediction 필드 포함)
- Ground truth JSONL 또는 추론 JSONL 내 ground_truth 필드

출력:
- 상세 결과 JSONL (각 샘플별 BLEU, METEOR)
- 요약 JSON (Corpus BLEU, 평균 METEOR)
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from korean_normalizer import (
    calculate_translation_metrics,
    calculate_corpus_bleu,
    calculate_meteor_score,
    tokenize_korean,
    SACREBLEU_AVAILABLE,
    NLTK_AVAILABLE
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
    prediction_jsonl: str,
    output_dir: str,
    gt_jsonl: str = None,
    gt_field: str = "question_ko",
    pred_field: str = "prediction",
    tokenize_method: str = "character"
) -> Dict:
    """
    Translation 평가 수행
    
    Args:
        prediction_jsonl: 추론 결과 JSONL 파일
        output_dir: 결과 저장 디렉토리
        gt_jsonl: Ground truth JSONL 파일 (None이면 prediction_jsonl에서 gt_field 사용)
        gt_field: Ground truth 필드명
        pred_field: Prediction 필드명
        tokenize_method: 토크나이징 방법 ("character", "space", "morpheme")
    
    Returns:
        평가 결과 딕셔너리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 라이브러리 상태 출력
    print(f"\n평가 라이브러리 상태:")
    print(f"  - sacrebleu: {'사용 가능' if SACREBLEU_AVAILABLE else '사용 불가 (pip install sacrebleu)'}")
    print(f"  - NLTK: {'사용 가능' if NLTK_AVAILABLE else '사용 불가 (pip install nltk)'}")
    print(f"  - 토크나이징 방법: {tokenize_method}")
    
    # 추론 결과 로드
    print(f"\n추론 결과 로드: {prediction_jsonl}")
    pred_data = load_jsonl(prediction_jsonl)
    print(f"총 {len(pred_data)}개 샘플")
    
    # Ground truth 로드
    if gt_jsonl:
        print(f"Ground truth 로드: {gt_jsonl}")
        gt_data = load_jsonl(gt_jsonl)
        
        # index로 매핑
        gt_map = {item.get('index', str(i)): item for i, item in enumerate(gt_data)}
    else:
        gt_map = None
    
    # 평가
    results = []
    all_references = []
    all_hypotheses = []
    
    # 개별 점수 누적
    total_bleu = 0.0
    total_meteor = 0.0
    
    for i, item in enumerate(pred_data):
        index = item.get('index', str(i))
        prediction = item.get(pred_field, "")
        
        # Ground truth 가져오기
        if gt_map:
            gt_item = gt_map.get(index, {})
            ground_truth = gt_item.get(gt_field, "")
        else:
            # ETRI는 answer_ko, 기타는 question_ko 또는 ground_truth
            ground_truth = item.get(gt_field, item.get('answer_ko', item.get('question_ko', item.get('ground_truth', ""))))
        
        # BLEU, METEOR 계산
        metrics = calculate_translation_metrics(
            ground_truth, prediction, tokenize_method
        )
        
        total_bleu += metrics['bleu']
        total_meteor += metrics['meteor']
        
        # Corpus BLEU 계산을 위해 저장
        all_references.append([ground_truth])
        all_hypotheses.append(prediction)
        
        # 결과 저장
        result = {
            "index": index,
            "audio_path": item.get('audio_path', item.get('raw', '')),
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
    
    # Corpus 레벨 BLEU 계산
    corpus_bleu = calculate_corpus_bleu(all_references, all_hypotheses, tokenize_method)
    
    # 평균 계산
    num_samples = len(results)
    avg_bleu = total_bleu / num_samples if num_samples > 0 else 0.0
    avg_meteor = total_meteor / num_samples if num_samples > 0 else 0.0
    
    # 결과 요약
    summary = {
        "prediction_file": prediction_jsonl,
        "gt_file": gt_jsonl if gt_jsonl else "same as prediction",
        "total_samples": num_samples,
        "tokenize_method": tokenize_method,
        # Corpus-level BLEU
        "corpus_bleu": corpus_bleu['bleu'],
        "corpus_bleu1": corpus_bleu.get('bleu1', 0.0),
        "corpus_bleu2": corpus_bleu.get('bleu2', 0.0),
        "corpus_bleu3": corpus_bleu.get('bleu3', 0.0),
        "corpus_bleu4": corpus_bleu.get('bleu4', 0.0),
        # Average metrics
        "avg_bleu": avg_bleu,
        "avg_meteor": avg_meteor,
        "timestamp": datetime.now().isoformat()
    }
    
    # 결과 저장
    dataset_name = Path(prediction_jsonl).stem
    if dataset_name.endswith('_predictions'):
        dataset_name = dataset_name[:-12]
    if dataset_name.endswith('_translation_results'):
        dataset_name = dataset_name[:-20]
    
    # 상세 결과 저장
    detail_path = os.path.join(output_dir, f"{dataset_name}_translation_eval_results.jsonl")
    with open(detail_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 요약 저장
    summary_path = os.path.join(output_dir, f"{dataset_name}_translation_eval_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 결과 출력
    print("\n" + "="*60)
    print(f"Translation 평가 완료: {dataset_name}")
    print("="*60)
    print(f"총 샘플: {num_samples}")
    print(f"토크나이징: {tokenize_method}")
    print("-"*60)
    print("BLEU 점수:")
    print(f"  Corpus BLEU: {corpus_bleu['bleu']:.2f}")
    print(f"  BLEU-1: {corpus_bleu.get('bleu1', 0.0):.2f}")
    print(f"  BLEU-2: {corpus_bleu.get('bleu2', 0.0):.2f}")
    print(f"  BLEU-3: {corpus_bleu.get('bleu3', 0.0):.2f}")
    print(f"  BLEU-4: {corpus_bleu.get('bleu4', 0.0):.2f}")
    print(f"  Average BLEU (문장별): {avg_bleu:.2f}")
    print("-"*60)
    print(f"METEOR 점수: {avg_meteor:.2f}")
    print("-"*60)
    print(f"\n결과 저장:")
    print(f"  상세: {detail_path}")
    print(f"  요약: {summary_path}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Translation 평가 (BLEU, METEOR 계산)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 추론 결과 평가 (같은 파일에 ground truth 포함)
  python evaluate_translation.py \\
    --prediction ./results/translation_predictions.jsonl \\
    --output-dir ./results

  # 별도의 ground truth 파일 사용
  python evaluate_translation.py \\
    --prediction ./results/predictions.jsonl \\
    --gt ./output/translation_test.jsonl \\
    --output-dir ./results

  # 다른 필드명 및 토크나이징 방법 지정
  python evaluate_translation.py \\
    --prediction ./results/output.jsonl \\
    --gt-field "text" \\
    --pred-field "translation" \\
    --tokenize space \\
    --output-dir ./results

토크나이징 방법:
  - character: 문자 단위 (한국어에 권장, 기본값)
  - space: 공백 단위
  - morpheme: 형태소 단위 (konlpy 설치 필요)
        """
    )
    
    parser.add_argument('--prediction', '-p', required=True,
                        help='추론 결과 JSONL 파일')
    parser.add_argument('--gt', '-g', default=None,
                        help='Ground truth JSONL 파일 (미지정 시 prediction 파일에서 추출)')
    parser.add_argument('--output-dir', '-o', default='./results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--gt-field', default='question_ko',
                        help='Ground truth 필드명 (기본: question_ko)')
    parser.add_argument('--pred-field', default='prediction',
                        help='Prediction 필드명 (기본: prediction)')
    parser.add_argument('--tokenize', '-t',
                        choices=['character', 'space', 'morpheme'],
                        default='character',
                        help='토크나이징 방법 (기본: character)')
    
    args = parser.parse_args()
    
    evaluate_translation(
        prediction_jsonl=args.prediction,
        output_dir=args.output_dir,
        gt_jsonl=args.gt,
        gt_field=args.gt_field,
        pred_field=args.pred_field,
        tokenize_method=args.tokenize
    )


if __name__ == '__main__':
    main()

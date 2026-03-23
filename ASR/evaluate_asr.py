#!/usr/bin/env python3
"""
ASR 평가 스크립트

추론 결과 JSONL 파일을 읽어서 CER(Character Error Rate)을 계산합니다.

입력:
- 추론 결과 JSONL (prediction 필드 포함)
- Ground truth JSONL 또는 추론 JSONL 내 ground_truth/question_ko 필드

출력:
- 상세 결과 JSONL
- 요약 JSON
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from korean_normalizer import calculate_cer


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
    prediction_jsonl: str,
    output_dir: str,
    gt_jsonl: str = None,
    gt_field: str = "question_ko",
    pred_field: str = "prediction"
) -> Dict:
    """
    ASR 평가 수행
    
    Args:
        prediction_jsonl: 추론 결과 JSONL 파일
        output_dir: 결과 저장 디렉토리
        gt_jsonl: Ground truth JSONL 파일 (None이면 prediction_jsonl에서 gt_field 사용)
        gt_field: Ground truth 필드명
        pred_field: Prediction 필드명
    
    Returns:
        평가 결과 딕셔너리
    """
    os.makedirs(output_dir, exist_ok=True)
    
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
    total_edit_distance = 0
    total_ref_length = 0
    
    for i, item in enumerate(pred_data):
        index = item.get('index', str(i))
        prediction = item.get(pred_field, "")
        
        # Ground truth 가져오기
        if gt_map:
            gt_item = gt_map.get(index, {})
            ground_truth = gt_item.get(gt_field, "")
        else:
            ground_truth = item.get(gt_field, item.get('question_ko', ""))
        
        cer, edit_dist, ref_len, gt_normalized, pred_normalized = calculate_cer(
            ground_truth, prediction
        )
        
        total_edit_distance += edit_dist
        total_ref_length += ref_len
        
        # 결과 저장
        result = {
            "index": index,
            "audio_path": item.get('audio_path', item.get('raw', '')),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "gt_normalized": gt_normalized,
            "pred_normalized": pred_normalized,
            "cer": cer,
            "edit_distance": edit_dist,
            "ref_length": ref_len
        }
        results.append(result)
    
    # 전체 CER 계산
    final_cer = total_edit_distance / total_ref_length if total_ref_length > 0 else 0
    
    # 결과 요약
    summary = {
        "prediction_file": prediction_jsonl,
        "gt_file": gt_jsonl if gt_jsonl else "same as prediction",
        "total_samples": len(results),
        "total_cer": final_cer,
        "total_edit_distance": total_edit_distance,
        "total_ref_length": total_ref_length,
        "normalization": "구두점 제거 + 공백 제거",
        "timestamp": datetime.now().isoformat()
    }
    
    # 결과 저장
    dataset_name = Path(prediction_jsonl).stem
    if dataset_name.endswith('_predictions'):
        dataset_name = dataset_name[:-12]
    
    # 상세 결과 저장
    detail_path = os.path.join(output_dir, f"{dataset_name}_eval_results.jsonl")
    with open(detail_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 요약 저장
    summary_path = os.path.join(output_dir, f"{dataset_name}_eval_summary.json")
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
    print(f"\n결과 저장:")
    print(f"  상세: {detail_path}")
    print(f"  요약: {summary_path}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='ASR 평가 (CER 계산)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 추론 결과 평가 (같은 파일에 ground truth 포함)
  python evaluate_asr.py \\
    --prediction ./results/clovacall_predictions.jsonl \\
    --output-dir ./results

  # 별도의 ground truth 파일 사용
  python evaluate_asr.py \\
    --prediction ./results/predictions.jsonl \\
    --gt ./output/clovacall_test.jsonl \\
    --output-dir ./results

  # 다른 필드명 지정
  python evaluate_asr.py \\
    --prediction ./results/output.jsonl \\
    --gt-field "text" \\
    --pred-field "transcription" \\
    --output-dir ./results
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
    args = parser.parse_args()
    
    evaluate_asr(
        prediction_jsonl=args.prediction,
        output_dir=args.output_dir,
        gt_jsonl=args.gt,
        gt_field=args.gt_field,
        pred_field=args.pred_field
    )


if __name__ == '__main__':
    main()

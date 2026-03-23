#!/usr/bin/env python3
"""
한국어 ASR 평가를 위한 텍스트 정규화 모듈

핵심 규칙:
- 구두점 제거 (한글 + 숫자만 유지)
- 공백 제거
"""

import re
from typing import Tuple


# ============================================================
# 기본 텍스트 처리 함수
# ============================================================

def remove_punctuation(text: str) -> str:
    """구두점 및 특수문자 제거 (한글, 숫자만 유지)"""
    text = re.sub(r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F0-9\s]', '', text)
    return text


def remove_spaces(text: str) -> str:
    """모든 공백 제거"""
    return re.sub(r'\s+', '', text)


def normalize_text_basic(text: str) -> str:
    """
    기본 정규화
    - 구두점 제거 (한글 + 숫자만 유지)
    - 공백 제거
    """
    if not text:
        return ""

    text = remove_punctuation(text)
    text = remove_spaces(text)

    return text


# ============================================================
# CER 계산
# ============================================================

def normalize_for_comparison(text: str) -> str:
    """
    비교를 위한 기본 정규화
    - 구두점 제거 (한글 + 숫자만 유지)
    - 공백 제거
    """
    return normalize_text_basic(text)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein 편집 거리 계산"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(
    reference: str, hypothesis: str
) -> Tuple[float, int, int, str, str]:
    """
    CER 계산 (구두점 제거 + 공백 제거)

    Args:
        reference: 정답 텍스트
        hypothesis: 예측 텍스트

    Returns:
        (CER, 편집거리, 정답길이, 정규화된 정답, 정규화된 예측)
    """
    ref_clean = normalize_text_basic(reference)
    hyp_clean = normalize_text_basic(hypothesis)

    if len(ref_clean) == 0:
        if len(hyp_clean) == 0:
            return 0.0, 0, 0, ref_clean, hyp_clean
        return 1.0, len(hyp_clean), 0, ref_clean, hyp_clean

    edit_dist = levenshtein_distance(ref_clean, hyp_clean)
    cer = edit_dist / len(ref_clean)

    return cer, edit_dist, len(ref_clean), ref_clean, hyp_clean


# 하위 호환성
calculate_cer_simple = calculate_cer


# ============================================================
# 테스트
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("한국어 ASR 정규화 테스트 (구두점 제거 + 공백 제거)")
    print("=" * 70)

    test_cases = [
        ("안녕하세요!", "안녕하세요", 0, "구두점 제거"),
        ("정말요?", "정말요", 0, "물음표 제거"),
        ("네, 알겠습니다.", "네알겠습니다", 0, "쉼표+마침표 제거"),
        ("아... 그래요", "아그래요", 0, "말줄임표+공백 제거"),
        ("감사 합니다", "감사합니다", 0, "공백 제거"),
        ("오백 원", "오백원", 0, "공백 제거"),
        ("오백 원", "오천원", None, "불일치"),
        ("세 잔", "세잔", 0, "공백 제거"),
        ("세 잔", "삼잔", None, "불일치"),
    ]

    print(f"\n{'Ground Truth':<18} {'Prediction':<18} {'CER':<10} {'결과':<12} {'설명'}")
    print("-" * 80)

    for gt, pred, expected_cer, desc in test_cases:
        cer, edit_dist, ref_len, ref_norm, hyp_norm = calculate_cer(gt, pred)

        if expected_cer is not None:
            status = "PASS" if cer == expected_cer else f"FAIL (expected {expected_cer})"
        else:
            status = f"CER={cer:.2f}" + (" (정상)" if cer > 0 else " (예상외)")

        print(f"{gt:<18} {pred:<18} {cer:<10.4f} {status:<12} {desc}")

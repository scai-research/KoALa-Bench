#!/usr/bin/env python3
"""
한국어 ASR/Translation 평가를 위한 텍스트 정규화 모듈

핵심 규칙:
- Ground truth에 숫자(4)가 있으면 → 모델이 "사", "네", "4" 중 아무거나 예측해도 OK
- Ground truth가 이미 한글("네 명")이면 → 그대로 유지
- 공백은 모두 제거하고 CER 평가

Translation 평가:
- BLEU: 한국어 문자 단위 또는 형태소 단위 토크나이징
- METEOR: 한국어에 맞는 토크나이징 적용
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from itertools import product

# Translation 평가를 위한 라이브러리
try:
    from sacrebleu.metrics import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    # METEOR에 필요한 wordnet 다운로드 시도
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# BERTScore
try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

# ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# 숫자 → 가능한 한글 표현들 (한자어, 고유어)
# 1~4, 20은 고유어/한자어 둘 다 가능
DIGIT_VARIANTS = {
    '0': ['영', '공'],
    '1': ['일', '하나', '한'],
    '2': ['이', '둘', '두'],
    '3': ['삼', '셋', '세'],
    '4': ['사', '넷', '네'],
    '5': ['오', '다섯'],
    '6': ['육', '여섯'],
    '7': ['칠', '일곱'],
    '8': ['팔', '여덟'],
    '9': ['구', '아홉'],
}

# 기본 변환 (첫 번째 = 한자어)
DIGIT_TO_HANJA = {k: v[0] for k, v in DIGIT_VARIANTS.items()}

# 십 단위 숫자 변형 (10, 20, 30, ...)
TEN_VARIANTS = {
    '10': ['십', '열'],
    '20': ['이십', '스물', '스무'],
    '30': ['삼십', '서른'],
    '40': ['사십', '마흔'],
    '50': ['오십', '쉰'],
    '60': ['육십', '예순'],
    '70': ['칠십', '일흔'],
    '80': ['팔십', '여든'],
    '90': ['구십', '아흔'],
}

# 영어 알파벳 → 한글 발음
ENGLISH_TO_KOREAN = {
    'a': '에이', 'b': '비', 'c': '씨', 'd': '디', 'e': '이',
    'f': '에프', 'g': '지', 'h': '에이치', 'i': '아이', 'j': '제이',
    'k': '케이', 'l': '엘', 'm': '엠', 'n': '엔', 'o': '오',
    'p': '피', 'q': '큐', 'r': '알', 's': '에스', 't': '티',
    'u': '유', 'v': '브이', 'w': '더블유', 'x': '엑스', 'y': '와이',
    'z': '제트'
}


def remove_punctuation(text: str) -> str:
    """구두점 및 특수문자 제거 (한글, 숫자, 영어만 유지)"""
    text = re.sub(r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318Fa-zA-Z0-9\s]', '', text)
    return text


def convert_english_to_korean(text: str) -> str:
    """영어 알파벳을 한글 발음으로 변환"""
    result = []
    for char in text:
        lower_char = char.lower()
        if lower_char in ENGLISH_TO_KOREAN:
            result.append(ENGLISH_TO_KOREAN[lower_char])
        else:
            result.append(char)
    return ''.join(result)


def remove_spaces(text: str) -> str:
    """모든 공백 제거"""
    return re.sub(r'\s+', '', text)


def normalize_text_basic(text: str) -> str:
    """
    기본 정규화 (숫자 변환 없이)
    - 구두점 제거
    - 영어 → 한글
    - 공백 제거
    """
    if not text:
        return ""
    
    text = remove_punctuation(text)
    text = convert_english_to_korean(text)
    text = remove_spaces(text)
    
    return text


def generate_number_variants(text: str) -> List[str]:
    """
    텍스트에서 숫자를 찾아 가능한 모든 변형을 생성
    
    예: "4명" → ["사명", "네명", "넷명", "4명"]
    예: "30만원" → ["삼십만원", "서른만원", "30만원", ...]
    
    Args:
        text: 원본 텍스트
    
    Returns:
        가능한 모든 변형 리스트
    """
    # 숫자 위치 찾기
    pattern = r'\d+'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        # 숫자가 없으면 원본만 반환
        return [text]
    
    # 각 숫자에 대해 가능한 변형 생성
    all_variants = []
    
    for match in matches:
        number_str = match.group()
        variants_for_number = set()
        
        # 1. 원본 숫자 추가
        variants_for_number.add(number_str)
        
        # 2. 십 단위 특별 처리 (10, 20, 30, ...)
        if number_str in TEN_VARIANTS:
            variants_for_number.update(TEN_VARIANTS[number_str])
        
        # 3. 두 자리 숫자 특별 처리 (예: 30 → 삼십)
        if len(number_str) == 2 and number_str[1] == '0' and number_str[0] != '0':
            # X0 형태: 삼십, 사십 등
            tens_digit = number_str[0]
            if tens_digit in DIGIT_VARIANTS:
                for variant in DIGIT_VARIANTS[tens_digit]:
                    variants_for_number.add(variant + '십')
        
        # 4. 두 자리 숫자 (예: 12 → 십이, 열둘)
        if len(number_str) == 2 and number_str[1] != '0':
            tens_digit = number_str[0]
            ones_digit = number_str[1]
            
            # 십 단위 변형
            tens_variants = []
            if tens_digit == '1':
                tens_variants = ['십', '열']
            elif tens_digit + '0' in TEN_VARIANTS:
                tens_variants = TEN_VARIANTS[tens_digit + '0']
            
            # 일 단위 변형
            ones_variants = DIGIT_VARIANTS.get(ones_digit, [ones_digit])
            
            for t in tens_variants:
                for o in ones_variants:
                    variants_for_number.add(t + o)
        
        # 5. 각 자릿수 개별 변환 (fallback)
        digit_variants_list = []
        for digit in number_str:
            if digit in DIGIT_VARIANTS:
                digit_variants_list.append(DIGIT_VARIANTS[digit] + [digit])
            else:
                digit_variants_list.append([digit])
        
        # 조합 생성 (너무 많으면 제한)
        if len(number_str) <= 3:
            for combo in product(*digit_variants_list):
                variants_for_number.add(''.join(combo))
        else:
            # 긴 숫자는 한자어 변환만
            hanja_variant = ''.join([DIGIT_TO_HANJA.get(d, d) for d in number_str])
            variants_for_number.add(hanja_variant)
        
        all_variants.append((match.start(), match.end(), list(variants_for_number)))
    
    # 모든 숫자의 변형 조합 생성
    def build_variants(text, variants_info, index=0):
        if index >= len(variants_info):
            return [text]
        
        start, end, variants = variants_info[index]
        results = []
        
        for variant in variants:
            new_text = text[:start] + variant + text[end:]
            # 다음 숫자 위치 조정
            offset = len(variant) - (end - start)
            adjusted_variants = [
                (s + offset if s > start else s, 
                 e + offset if e > start else e, 
                 v) 
                for s, e, v in variants_info[index+1:]
            ]
            results.extend(build_variants(new_text, adjusted_variants, 0))
        
        return results
    
    variants = build_variants(text, all_variants)
    
    # 중복 제거
    return list(set(variants))


def normalize_for_comparison(text: str) -> str:
    """
    비교를 위한 정규화 (prediction용)
    - 구두점 제거
    - 영어 → 한글
    - 공백 제거
    - 숫자는 변환하지 않음 (이미 한글일 것이므로)
    """
    return normalize_text_basic(text)


def calculate_cer_with_variants(reference: str, hypothesis: str) -> Tuple[float, int, int, str]:
    """
    숫자 변형을 고려한 CER 계산
    
    Ground truth에 숫자가 있으면 가능한 모든 변형과 비교하여 최소 CER 선택
    
    Args:
        reference: 정답 텍스트 (원본, 숫자 포함 가능)
        hypothesis: 예측 텍스트
    
    Returns:
        (CER, 편집거리, 정답길이, 매칭된 정규화 형태)
    """
    # 1. 기본 정규화 (구두점, 영어, 공백)
    ref_basic = remove_punctuation(reference)
    ref_basic = convert_english_to_korean(ref_basic)
    
    hyp_normalized = normalize_for_comparison(hypothesis)
    
    # 2. Ground truth에서 숫자 변형 생성
    ref_variants = generate_number_variants(ref_basic)
    
    # 3. 각 변형에 대해 공백 제거 후 CER 계산
    min_cer = float('inf')
    min_edit_dist = 0
    min_ref_len = 0
    best_ref = ""
    
    for ref_variant in ref_variants:
        ref_clean = remove_spaces(ref_variant)
        
        if len(ref_clean) == 0:
            if len(hyp_normalized) == 0:
                return 0.0, 0, 0, ref_clean
            continue
        
        # Levenshtein 거리 계산
        edit_dist = levenshtein_distance(ref_clean, hyp_normalized)
        cer = edit_dist / len(ref_clean)
        
        if cer < min_cer:
            min_cer = cer
            min_edit_dist = edit_dist
            min_ref_len = len(ref_clean)
            best_ref = ref_clean
    
    if min_cer == float('inf'):
        return 1.0, len(hyp_normalized), 0, ""
    
    return min_cer, min_edit_dist, min_ref_len, best_ref


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
            # 삽입, 삭제, 대체
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    """
    CER 계산 (하위 호환성 유지)
    
    Args:
        reference: 정답 텍스트
        hypothesis: 예측 텍스트
    
    Returns:
        (CER, 편집거리, 정답 길이)
    """
    cer, edit_dist, ref_len, _ = calculate_cer_with_variants(reference, hypothesis)
    return cer, edit_dist, ref_len


# ============================================================
# Translation 평가 함수들 (BLEU, METEOR)
# ============================================================

def tokenize_korean(text: str, method: str = "character") -> List[str]:
    """
    한국어 텍스트 토크나이징
    
    Args:
        text: 입력 텍스트
        method: 토크나이징 방법
            - "character": 문자 단위 (한국어에 적합)
            - "space": 공백 단위
            - "morpheme": 형태소 단위 (konlpy 필요)
    
    Returns:
        토큰 리스트
    """
    if not text:
        return []
    
    # 기본 정규화 (구두점 정리)
    text = re.sub(r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318Fa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if method == "character":
        # 문자 단위 토크나이징 (공백 제외)
        return list(text.replace(" ", ""))
    elif method == "space":
        # 공백 단위 토크나이징
        return text.split()
    elif method == "morpheme":
        # 형태소 분석 (konlpy 사용 시도)
        try:
            from konlpy.tag import Okt
            okt = Okt()
            return okt.morphs(text)
        except ImportError:
            # konlpy 없으면 공백 단위로 fallback
            return text.split()
    else:
        return text.split()


def calculate_bleu_score(
    references: List[str], 
    hypothesis: str,
    tokenize_method: str = "character"
) -> Dict[str, float]:
    """
    BLEU 점수 계산 (한국어 지원)
    
    Args:
        references: 정답 텍스트 리스트 (여러 정답 가능)
        hypothesis: 예측 텍스트
        tokenize_method: 토크나이징 방법 ("character", "space", "morpheme")
    
    Returns:
        BLEU 점수 딕셔너리 (bleu, bleu1, bleu2, bleu3, bleu4)
    """
    if not SACREBLEU_AVAILABLE:
        # sacrebleu 없으면 직접 계산
        return _calculate_bleu_manual(references, hypothesis, tokenize_method)
    
    # sacrebleu 사용
    # 한국어는 문자 단위로 토크나이징
    if tokenize_method == "character":
        # 문자 단위로 변환 (공백으로 구분)
        refs_tokenized = [[" ".join(tokenize_korean(ref, "character"))] for ref in references]
        hyp_tokenized = " ".join(tokenize_korean(hypothesis, "character"))
        
        # sacrebleu는 이미 토크나이징된 텍스트에 대해 tokenize='none' 사용
        bleu = BLEU(tokenize='none')
        result = bleu.corpus_score([hyp_tokenized], refs_tokenized)
    else:
        # 공백 단위
        refs_tokenized = [[" ".join(tokenize_korean(ref, tokenize_method))] for ref in references]
        hyp_tokenized = " ".join(tokenize_korean(hypothesis, tokenize_method))
        
        bleu = BLEU(tokenize='none')
        result = bleu.corpus_score([hyp_tokenized], refs_tokenized)
    
    return {
        "bleu": result.score,
        "bleu1": result.precisions[0] if len(result.precisions) > 0 else 0.0,
        "bleu2": result.precisions[1] if len(result.precisions) > 1 else 0.0,
        "bleu3": result.precisions[2] if len(result.precisions) > 2 else 0.0,
        "bleu4": result.precisions[3] if len(result.precisions) > 3 else 0.0,
    }


def _calculate_bleu_manual(
    references: List[str], 
    hypothesis: str,
    tokenize_method: str = "character"
) -> Dict[str, float]:
    """
    BLEU 점수 수동 계산 (sacrebleu 없을 때)
    """
    from collections import Counter
    import math
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    hyp_tokens = tokenize_korean(hypothesis, tokenize_method)
    ref_tokens_list = [tokenize_korean(ref, tokenize_method) for ref in references]
    
    if not hyp_tokens:
        return {"bleu": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    precisions = []
    
    for n in range(1, 5):
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        
        # 각 reference에서 최대 매칭 카운트
        max_ref_counts = Counter()
        for ref_tokens in ref_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            for ngram in hyp_ngrams:
                max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
        
        # 클리핑된 카운트
        clipped_count = sum(min(count, max_ref_counts[ngram]) 
                          for ngram, count in hyp_ngrams.items())
        total_count = sum(hyp_ngrams.values())
        
        precision = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(precision * 100)  # 퍼센트로 변환
    
    # Brevity penalty
    hyp_len = len(hyp_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - hyp_len), x))
    
    if hyp_len > closest_ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - closest_ref_len / hyp_len)
    
    # BLEU score (geometric mean of precisions)
    if min(precisions) > 0:
        log_precisions = [math.log(p / 100) for p in precisions]
        bleu = bp * math.exp(sum(log_precisions) / len(log_precisions)) * 100
    else:
        bleu = 0.0
    
    return {
        "bleu": bleu,
        "bleu1": precisions[0],
        "bleu2": precisions[1],
        "bleu3": precisions[2],
        "bleu4": precisions[3],
    }


def calculate_meteor_score(
    references: List[str], 
    hypothesis: str,
    tokenize_method: str = "character"
) -> float:
    """
    METEOR 점수 계산 (한국어 지원)
    
    Args:
        references: 정답 텍스트 리스트
        hypothesis: 예측 텍스트
        tokenize_method: 토크나이징 방법
    
    Returns:
        METEOR 점수 (0-1)
    """
    if not NLTK_AVAILABLE:
        # NLTK 없으면 간단한 F1 기반 점수 계산
        return _calculate_meteor_manual(references, hypothesis, tokenize_method)
    
    # 한국어 토크나이징
    hyp_tokens = tokenize_korean(hypothesis, tokenize_method)
    ref_tokens_list = [tokenize_korean(ref, tokenize_method) for ref in references]
    
    if not hyp_tokens or not ref_tokens_list:
        return 0.0
    
    # METEOR 계산 (각 reference에 대해 계산 후 최대값)
    scores = []
    for ref_tokens in ref_tokens_list:
        try:
            # NLTK meteor_score는 토큰 리스트를 받음
            score = meteor_score([ref_tokens], hyp_tokens)
            scores.append(score)
        except Exception:
            # 오류 시 수동 계산으로 fallback
            scores.append(_calculate_meteor_manual([' '.join(ref_tokens)], 
                                                   ' '.join(hyp_tokens), 
                                                   tokenize_method))
    
    return max(scores) if scores else 0.0


def _calculate_meteor_manual(
    references: List[str], 
    hypothesis: str,
    tokenize_method: str = "character"
) -> float:
    """
    METEOR 점수 수동 계산 (NLTK 없을 때)
    간단한 F1 기반 유사도 사용
    """
    hyp_tokens = set(tokenize_korean(hypothesis, tokenize_method))
    
    if not hyp_tokens:
        return 0.0
    
    best_score = 0.0
    
    for ref in references:
        ref_tokens = set(tokenize_korean(ref, tokenize_method))
        
        if not ref_tokens:
            continue
        
        # 매칭 토큰
        matches = len(hyp_tokens & ref_tokens)
        
        if matches == 0:
            continue
        
        # Precision & Recall
        precision = matches / len(hyp_tokens)
        recall = matches / len(ref_tokens)
        
        # F1 score (METEOR의 간단한 근사)
        f1 = 2 * precision * recall / (precision + recall)
        
        best_score = max(best_score, f1)
    
    return best_score


def calculate_bertscore(
    references: List[str],
    hypotheses: List[str],
    lang: str = "ko",
    model_type: str = "xlm-roberta-large",
    rescale_with_baseline: bool = True
) -> Dict[str, List[float]]:
    """
    BERTScore 계산 (배치 단위)
    
    Args:
        references: 정답 텍스트 리스트
        hypotheses: 예측 텍스트 리스트
        lang: 언어 코드 ("ko", "en" 등)
        model_type: 사용할 BERT 모델
        rescale_with_baseline: baseline으로 rescale 여부
    
    Returns:
        {'precision': [...], 'recall': [...], 'f1': [...]}
    """
    if not BERTSCORE_AVAILABLE:
        # BERTScore 없으면 0 반환
        n = len(hypotheses)
        return {'precision': [0.0] * n, 'recall': [0.0] * n, 'f1': [0.0] * n}
    
    try:
        P, R, F1 = bert_score_func(
            hypotheses, references,
            lang=lang,
            model_type=model_type,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False
        )
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
    except Exception as e:
        print(f"[WARNING] BERTScore 계산 실패: {e}")
        n = len(hypotheses)
        return {'precision': [0.0] * n, 'recall': [0.0] * n, 'f1': [0.0] * n}


def calculate_rouge_score(
    reference: str,
    hypothesis: str,
    rouge_types: List[str] = None
) -> Dict[str, float]:
    """
    ROUGE 점수 계산
    
    Args:
        reference: 정답 텍스트
        hypothesis: 예측 텍스트
        rouge_types: 계산할 ROUGE 타입 리스트 (기본: rouge1, rouge2, rougeL)
    
    Returns:
        ROUGE 점수 딕셔너리 (rouge1, rouge2, rougeL의 f-score)
    """
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    if not ROUGE_AVAILABLE:
        # ROUGE 없으면 0 반환
        return {f'{rt}_f': 0.0 for rt in rouge_types}
    
    try:
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        
        result = {}
        for rouge_type in rouge_types:
            result[f'{rouge_type}_precision'] = scores[rouge_type].precision * 100
            result[f'{rouge_type}_recall'] = scores[rouge_type].recall * 100
            result[f'{rouge_type}_f'] = scores[rouge_type].fmeasure * 100
        
        return result
    except Exception as e:
        print(f"[WARNING] ROUGE 계산 실패: {e}")
        return {f'{rt}_f': 0.0 for rt in rouge_types}


def calculate_translation_metrics(
    reference: str, 
    hypothesis: str,
    tokenize_method: str = "character"
) -> Dict[str, float]:
    """
    번역 평가 메트릭 계산 (BLEU + METEOR)
    
    Note: BERTScore는 배치로 계산하는 것이 효율적이므로 
          evaluate_translation에서 별도로 계산합니다.
    
    Args:
        reference: 정답 텍스트
        hypothesis: 예측 텍스트
        tokenize_method: 토크나이징 방법 ("character", "space", "morpheme")
    
    Returns:
        메트릭 딕셔너리 (bleu, bleu1-4, meteor)
    """
    references = [reference]  # 단일 reference
    
    # BLEU 계산
    bleu_scores = calculate_bleu_score(references, hypothesis, tokenize_method)
    
    # METEOR 계산
    meteor = calculate_meteor_score(references, hypothesis, tokenize_method)
    
    result = {
        **bleu_scores,
        "meteor": meteor * 100,  # 퍼센트로 변환
    }
    
    return result


def calculate_corpus_bleu(
    references_list: List[List[str]], 
    hypotheses: List[str],
    tokenize_method: str = "character"
) -> Dict[str, float]:
    """
    코퍼스 레벨 BLEU 계산
    
    Args:
        references_list: 각 샘플의 정답 리스트들의 리스트
        hypotheses: 예측 텍스트 리스트
        tokenize_method: 토크나이징 방법
    
    Returns:
        코퍼스 BLEU 점수
    """
    if not SACREBLEU_AVAILABLE:
        # 개별 BLEU의 평균으로 대체
        scores = []
        for refs, hyp in zip(references_list, hypotheses):
            score = calculate_bleu_score(refs, hyp, tokenize_method)
            scores.append(score["bleu"])
        return {"bleu": sum(scores) / len(scores) if scores else 0.0}
    
    # sacrebleu 코퍼스 BLEU
    if tokenize_method == "character":
        refs_tokenized = []
        for refs in references_list:
            refs_tokenized.append([" ".join(tokenize_korean(ref, "character")) for ref in refs])
        
        # Transpose for sacrebleu format
        refs_transposed = list(zip(*refs_tokenized)) if refs_tokenized else [[]]
        refs_transposed = [list(r) for r in refs_transposed]
        
        hyps_tokenized = [" ".join(tokenize_korean(hyp, "character")) for hyp in hypotheses]
        
        bleu = BLEU(tokenize='none')
        result = bleu.corpus_score(hyps_tokenized, refs_transposed)
    else:
        refs_tokenized = []
        for refs in references_list:
            refs_tokenized.append([" ".join(tokenize_korean(ref, tokenize_method)) for ref in refs])
        
        refs_transposed = list(zip(*refs_tokenized)) if refs_tokenized else [[]]
        refs_transposed = [list(r) for r in refs_transposed]
        
        hyps_tokenized = [" ".join(tokenize_korean(hyp, tokenize_method)) for hyp in hypotheses]
        
        bleu = BLEU(tokenize='none')
        result = bleu.corpus_score(hyps_tokenized, refs_transposed)
    
    return {
        "bleu": result.score,
        "bleu1": result.precisions[0] if len(result.precisions) > 0 else 0.0,
        "bleu2": result.precisions[1] if len(result.precisions) > 1 else 0.0,
        "bleu3": result.precisions[2] if len(result.precisions) > 2 else 0.0,
        "bleu4": result.precisions[3] if len(result.precisions) > 3 else 0.0,
    }


# 테스트
if __name__ == '__main__':
    print("=" * 60)
    print("한국어 ASR/Translation 정규화 테스트")
    print("=" * 60)
    
    print("\n=== 1. 숫자 변형 생성 테스트 ===\n")
    
    test_texts = ["4명", "12시", "A세트"]
    for text in test_texts:
        variants = generate_number_variants(text)
        print(f"'{text}' → {variants[:5]}{'...' if len(variants) > 5 else ''}")
    
    print("\n=== 2. CER 계산 테스트 (핵심!) ===\n")
    
    test_cases = [
        # (Ground Truth, Prediction, 기대결과)
        ("4명이요.", "사명이요", "GT가 숫자 → 한자어 OK"),
        ("4명이요.", "네명이요", "GT가 숫자 → 고유어 OK"),
        ("4명이요.", "4명이요", "GT가 숫자 → 숫자 OK"),
        ("네 명이요", "네명이요", "GT가 한글 → 공백만 제거"),
        ("네 명이요", "사명이요", "GT가 한글 → 다른 숫자는 오류"),
        ("세트 A요", "세트에이요", "영어 변환"),
    ]
    
    print(f"{'Ground Truth':<15} {'Prediction':<15} {'CER':<10} {'설명'}")
    print("-" * 60)
    
    for gt, pred, desc in test_cases:
        cer, edit_dist, ref_len, matched = calculate_cer_with_variants(gt, pred)
        print(f"{gt:<15} {pred:<15} {cer:.4f}     {desc}")
        if cer > 0:
            print(f"  → 매칭 시도: '{matched}' vs '{normalize_for_comparison(pred)}'")
    
    print("\n=== 3. 핵심 검증 ===\n")
    
    # 핵심 케이스: GT="4명" 일 때 "사명", "네명" 모두 CER=0
    gt = "4명이요."
    for pred in ["사명이요", "네명이요", "4명이요"]:
        cer, _, _, _ = calculate_cer_with_variants(gt, pred)
        status = "✅ PASS" if cer == 0 else "❌ FAIL"
        print(f"GT='{gt}' vs Pred='{pred}' → CER={cer:.4f} {status}")
    
    # GT가 한글일 때는 그대로
    gt = "네 명이요"
    for pred in ["네명이요", "사명이요"]:
        cer, _, _, _ = calculate_cer_with_variants(gt, pred)
        expected = 0 if pred == "네명이요" else None
        if expected is not None:
            status = "✅ PASS" if cer == expected else "❌ FAIL"
        else:
            status = f"(오류 예상: {cer:.4f})"
        print(f"GT='{gt}' vs Pred='{pred}' → CER={cer:.4f} {status}")
    
    print("\n" + "=" * 60)
    print("Translation 평가 테스트 (BLEU, METEOR)")
    print("=" * 60)
    
    print(f"\nsacrebleu 사용 가능: {SACREBLEU_AVAILABLE}")
    print(f"NLTK 사용 가능: {NLTK_AVAILABLE}")
    
    print("\n=== 4. BLEU/METEOR 계산 테스트 ===\n")
    
    translation_cases = [
        ("오늘 날씨가 좋습니다", "오늘 날씨가 좋습니다", "완전 일치"),
        ("오늘 날씨가 좋습니다", "오늘 날씨가 좋아요", "유사"),
        ("오늘 날씨가 좋습니다", "내일 비가 옵니다", "완전 다름"),
        ("I love you", "나는 너를 사랑해", "영어→한국어"),
    ]
    
    print(f"{'Reference':<20} {'Hypothesis':<20} {'BLEU':<10} {'METEOR':<10} {'설명'}")
    print("-" * 80)
    
    for ref, hyp, desc in translation_cases:
        metrics = calculate_translation_metrics(ref, hyp, tokenize_method="character")
        print(f"{ref:<20} {hyp:<20} {metrics['bleu']:.2f}     {metrics['meteor']:.2f}      {desc}")
    
    print("\n=== 5. 토크나이징 방법 비교 ===\n")
    
    ref = "오늘 날씨가 정말 좋습니다"
    hyp = "오늘 날씨가 매우 좋아요"
    
    for method in ["character", "space"]:
        metrics = calculate_translation_metrics(ref, hyp, tokenize_method=method)
        print(f"Method: {method:<12} BLEU: {metrics['bleu']:.2f}, METEOR: {metrics['meteor']:.2f}")

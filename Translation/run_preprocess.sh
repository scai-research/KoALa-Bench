#!/bin/bash
#
# ETRI EnKoST-C Translation 데이터셋 전처리 스크립트
# 
# 사용법:
#   ./run_preprocess.sh etri
#
# 참고:
#   - etri: translate_etri.py로 전처리 후 output 디렉토리로 복사
#
# Translation 평가 시 필요한 필드:
#   - raw: 오디오 파일 경로 (영어 음성 wav)
#   - answer_ko: Ground truth 한국어 번역
#   - answer_en: 영어 전사 (참고용)
#   - index: 샘플 인덱스
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

# ETRI 소스 경로
ETRI_SOURCE_DIR="/home/jiwonyoon/data1/projects/SpeechLM_benchmark/hslim/process_datasets/etri_enkost"

usage() {
    echo "사용법:"
    echo "  $0 etri"
    echo ""
    echo "설명:"
    echo "  ETRI EnKoST-C 데이터셋을 output 디렉토리로 복사합니다."
    echo "  (translate_etri.py로 미리 전처리되어 있어야 합니다)"
    echo ""
    echo "출력 JSONL 형식:"
    echo "  - raw: 오디오 파일 절대 경로 (영어 음성 wav)"
    echo "  - answer_ko: Ground truth 한국어 번역"
    echo "  - answer_en: 영어 전사 (참고용)"
    echo "  - index: 샘플 인덱스"
    echo ""
    echo "예시:"
    echo "  $0 etri"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

DATASET=$1

if [ "$DATASET" != "etri" ]; then
    echo "오류: 이 스크립트는 ETRI 데이터셋만 지원합니다."
    echo "사용법: $0 etri"
    exit 1
fi

# ETRI 데이터셋 처리
echo "======================================"
echo "ETRI EnKoST-C 데이터셋 준비"
echo "======================================"
echo ""

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 파일 존재 확인 및 복사
ETRI_COMMON="${ETRI_SOURCE_DIR}/etri_tst-COMMON_processed.jsonl"
ETRI_HE="${ETRI_SOURCE_DIR}/etri_tst-HE_processed.jsonl"

if [ ! -f "$ETRI_COMMON" ]; then
    echo "오류: tst-COMMON 파일을 찾을 수 없습니다: $ETRI_COMMON"
    echo ""
    echo "먼저 translate_etri.py를 실행하세요:"
    echo "  cd /home/jiwonyoon/data1/projects/SpeechLM_benchmark/hslim"
    echo "  python translate_etri.py"
    exit 1
fi

if [ ! -f "$ETRI_HE" ]; then
    echo "오류: tst-HE 파일을 찾을 수 없습니다: $ETRI_HE"
    echo ""
    echo "먼저 translate_etri.py를 실행하세요:"
    echo "  cd /home/jiwonyoon/data1/projects/SpeechLM_benchmark/hslim"
    echo "  python translate_etri.py"
    exit 1
fi

# 파일 복사
echo "파일 복사 중..."
cp "$ETRI_COMMON" "$OUTPUT_DIR/"
cp "$ETRI_HE" "$OUTPUT_DIR/"

echo ""
echo "✓ tst-COMMON 복사 완료: $(wc -l < ${OUTPUT_DIR}/etri_tst-COMMON_processed.jsonl) 샘플"
echo "✓ tst-HE 복사 완료: $(wc -l < ${OUTPUT_DIR}/etri_tst-HE_processed.jsonl) 샘플"

echo ""
echo "======================================"
echo "ETRI 데이터셋 준비 완료!"
echo "======================================"
echo ""
echo "다음 단계:"
echo "  1. ETRI tst-COMMON 평가:"
echo "     ./run_translation_eval.sh etri_common <model_path> ./results"
echo ""
echo "  2. ETRI tst-HE 평가:"
echo "     ./run_translation_eval.sh etri_he <model_path> ./results"
echo ""
echo "  3. ETRI 전체 평가:"
echo "     ./run_translation_eval.sh etri <model_path> ./results"
echo ""

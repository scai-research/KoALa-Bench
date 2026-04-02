#!/bin/bash
#
# 한국어 ASR 데이터셋 전처리 스크립트
# 
# 사용법:
#   ./run_preprocess.sh ksponspeech <trn_path> <audio_root> <wav_output_dir> <output_jsonl> [clean|other]
#   ./run_preprocess.sh commonvoice <tsv_path> <audio_root> <wav_output_dir> <output_jsonl>
#   ./run_preprocess.sh zeroth <data_dir> <audio_root> <wav_output_dir> <output_jsonl>
#
# 참고:
#   - ksponspeech: pcm → wav 변환 (clean/other 지정 시 영어→한국어 자동 매핑)
#   - commonvoice: mp3 → wav 변환
#   - zeroth: flac → wav 변환
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/preprocess_korean_asr.py"

usage() {
    echo "사용법:"
    echo "  $0 ksponspeech <trn_path> <audio_root> <wav_output_dir> <output.jsonl> [clean|other]"
    echo "  $0 commonvoice <tsv_path> <audio_root> <wav_output_dir> <output.jsonl>"
    echo "  $0 zeroth <data_dir> <audio_root> <wav_output_dir> <output.jsonl>"
    echo ""
    echo "데이터셋별 오디오 처리:"
    echo "  ksponspeech  - pcm → wav 변환 (인자 5~6개, clean/other 지정 시 영어→한국어 자동 매핑)"
    echo "  commonvoice  - mp3 → wav 변환 (인자 5개)"
    echo "  zeroth       - flac → wav 변환 (인자 5개)"
    echo ""
    echo "예시:"
    echo "  $0 ksponspeech /path/to/eval_clean.trn /path/to/eval_clean /path/to/wav_out ./output/ksponspeech_eval_clean.jsonl clean"
    echo "  $0 ksponspeech /path/to/eval_other.trn /path/to/eval_other /path/to/wav_out ./output/ksponspeech_eval_other.jsonl other"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

DATASET=$1

case "$DATASET" in
    ksponspeech)
        if [ $# -lt 5 ] || [ $# -gt 6 ]; then
            echo "오류: ksponspeech은 5~6개의 인자가 필요합니다."
            echo "사용법: $0 ksponspeech <trn_path> <audio_root> <wav_output_dir> <output.jsonl> [clean|other]"
            exit 1
        fi
        INPUT=$2
        AUDIO_ROOT=$3
        WAV_OUTPUT=$4
        OUTPUT=$5
        SPLIT=${6:-""}
        
        echo "=== KsponSpeech 처리 (PCM → WAV) ==="
        cmd="python3 \"$PREPROCESS_SCRIPT\" ksponspeech \
            --input \"$INPUT\" \
            --audio-root \"$AUDIO_ROOT\" \
            --wav-output \"$WAV_OUTPUT\" \
            --output \"$OUTPUT\""
        
        if [ -n "$SPLIT" ]; then
            echo "스플릿: eval_${SPLIT} (영어→한국어 자동 매핑 적용)"
            cmd="$cmd --split \"$SPLIT\""
        fi
        
        eval $cmd
        ;;
        
    commonvoice)
        if [ $# -ne 5 ]; then
            echo "오류: commonvoice는 5개의 인자가 필요합니다."
            echo "사용법: $0 commonvoice <tsv_path> <audio_root> <wav_output_dir> <output.jsonl>"
            exit 1
        fi
        INPUT=$2
        AUDIO_ROOT=$3
        WAV_OUTPUT=$4
        OUTPUT=$5
        
        echo "=== Common Voice 처리 (MP3 → WAV) ==="
        python3 "$PREPROCESS_SCRIPT" commonvoice \
            --input "$INPUT" \
            --audio-root "$AUDIO_ROOT" \
            --wav-output "$WAV_OUTPUT" \
            --output "$OUTPUT"
        ;;
        
    zeroth)
        if [ $# -ne 5 ]; then
            echo "오류: zeroth는 5개의 인자가 필요합니다."
            echo "사용법: $0 zeroth <data_dir> <audio_root> <wav_output_dir> <output.jsonl>"
            exit 1
        fi
        INPUT=$2
        AUDIO_ROOT=$3
        WAV_OUTPUT=$4
        OUTPUT=$5
        
        echo "=== Zeroth Korean 처리 (FLAC → WAV) ==="
        python3 "$PREPROCESS_SCRIPT" zeroth \
            --input "$INPUT" \
            --audio-root "$AUDIO_ROOT" \
            --wav-output "$WAV_OUTPUT" \
            --output "$OUTPUT"
        ;;
        
    *)
        echo "오류: 알 수 없는 데이터셋 '$DATASET'"
        usage
        ;;
esac

echo ""
echo "완료!"

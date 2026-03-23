# 한국어 ASR 평가

한국어 음성인식(ASR) 평가 도구입니다. 다양한 모델 백엔드를 지원합니다.

## 📁 폴더 구조

```
ASR/
├── README.md                    # 이 문서
├── preprocess_korean_asr.py     # 데이터 전처리 스크립트
├── run_preprocess.sh            # 전처리 실행 스크립트
├── evaluate_asr.py              # CER 평가 스크립트
├── run_evaluate.sh              # 평가 실행 스크립트
├── korean_normalizer.py         # 한국어 텍스트 정규화 모듈
├── output/                      # 전처리된 JSONL 파일들
└── results/                     # 평가 결과
```

---

## 📊 파이프라인

```
[원본 데이터] → [전처리] → [추론] → [평가]
                  │          │        │
               JSONL      JSONL    Summary
           (question_ko) (prediction) (CER)
```

1. **전처리**: KsponSpeech TRN 파일을 통합 JSONL로 변환 (Common Voice / Zeroth Korean은 HuggingFace에서 전처리 완료된 파일 제공)
2. **추론**: `../src/run_inference.sh`를 사용하여 음성인식 수행
3. **평가**: 추론 결과와 Ground truth를 비교하여 CER 계산

---

## 📂 지원 데이터셋

> ⚠️ **라이선스 안내**: 아래 데이터셋들의 모든 권한은 각 데이터셋 제작자에게 있습니다. 본 프로젝트는 **전처리 및 평가 코드만 제공**하며, 데이터셋 사용 시 원본 라이선스를 반드시 확인하시기 바랍니다.
>
> 📋 **참고**: KsponSpeech는 전처리가 필요하며, Common Voice / Zeroth Korean은 HuggingFace에서 전처리된 파일을 다운로드하여 사용합니다.

### KsponSpeech

AI Hub에서 공개한 대규모 한국어 자발적 발화 데이터셋입니다.
방송, 강연, 대화 등 다양한 주제의 자연스러운 발화로 구성되어 있으며, 한국어 ASR 연구에 널리 사용됩니다.
**eval_clean**과 **eval_other** 두 평가 세트를 지원합니다.
전사 텍스트에 포함된 영어 약어(예: `TV`, `SKT`, `OT`)는 `--split clean|other` 지정 시 한국어 발음으로 자동 치환됩니다.
입력 형식: TRN 파일 (`eval_clean.trn`, `eval_other.trn`) + PCM 오디오 파일 (→ WAV 변환 필요)

### Common Voice / Zeroth Korean

Common Voice(Mozilla)와 Zeroth Korean 데이터셋은 전처리된 JSONL 및 오디오 파일을 HuggingFace에서 제공합니다.
별도의 전처리 없이 다운로드 후 바로 추론 및 평가에 사용할 수 있습니다.

📥 **다운로드**: [https://huggingface.co/datasets/YOUR_ORG/ko-speech-eval-benchmark](https://huggingface.co/datasets/YOUR_ORG/ko-speech-eval-benchmark)

---

## 🚀 Step 1: 데이터 전처리

KsponSpeech는 원본 TRN + PCM 파일로부터 전처리가 필요합니다.
Common Voice와 Zeroth Korean은 위 HuggingFace 링크에서 전처리 완료된 파일을 다운로드하면 됩니다.

### 사용법

```bash
./run_preprocess.sh ksponspeech <trn_path> <audio_root> <wav_output_dir> <output.jsonl> [clean|other]
```

### 예시

```bash
# KsponSpeech eval_clean (pcm → wav 변환 + 영어→한국어 자동 매핑)
./run_preprocess.sh ksponspeech \
  /path/to/eval_clean.trn \
  /path/to/eval_clean \
  ./output/ksponspeech_wav \
  ./output/ksponspeech_eval_clean.jsonl \
  clean

# KsponSpeech eval_other (pcm → wav 변환 + 영어→한국어 자동 매핑)
./run_preprocess.sh ksponspeech \
  /path/to/eval_other.trn \
  /path/to/eval_other \
  ./output/ksponspeech_wav \
  ./output/ksponspeech_eval_other.jsonl \
  other
```

### 영어→한국어 발음 자동 치환

KsponSpeech 전사 텍스트에는 영어 약어/단어가 포함된 경우가 있습니다 (예: `TV`→티비, `SKT`→에스케이티, `OT`→오티).
`--split clean` 또는 `--split other`를 지정하면, 스크립트 내부에 사전 정의된 매핑 테이블에 따라
해당 스플릿의 WAV 파일명을 기준으로 영어를 한국어 발음으로 자동 치환합니다.
별도의 매핑 파일을 준비할 필요가 없습니다.

### 출력 형식

```json
{
  "index": "000000",
  "raw": "/full/path/to/audio.wav",
  "prompt": "Transcribe the following audio to Korean. Output only the transcribed Korean text without any explanations or additional content:",
  "question_ko": "안녕하세요",
  "speaker_id": "speaker_001"
}
```

---

## 🎯 Step 2: 모델 추론

`../src/` 폴더의 공통 추론 코드를 사용합니다. **task별 DataLoader를 사용합니다.**

### 사용법

```bash
./run_inference.sh <task> <input.jsonl> <output.jsonl> <model_path> [max_samples]
```

### 예시

```bash
cd ../src

# ASR 추론 (JSONL의 prompt 필드 자동 사용)
./run_inference.sh asr \
  ../ASR/output/ksponspeech_eval_clean.jsonl \
  ../ASR/results/ksponspeech_eval_clean_predictions.jsonl \
  /path/to/Qwen2-Audio-7B-Instruct

# 테스트 (10개만)
./run_inference.sh asr \
  ../ASR/output/ksponspeech_eval_clean.jsonl \
  ../ASR/results/test_predictions.jsonl \
  /path/to/model \
  10
```

### 출력 형식 (간결)

```json
{"index": "000000", "prediction": "안녕하세요"}
{"index": "000001", "prediction": "감사합니다"}
```

---

## 📐 Step 3: CER 평가

### 사용법

```bash
./run_evaluate.sh <prediction.jsonl> <gt.jsonl> [output_dir]
```

### 예시

```bash
# 평가 실행 (prediction + ground truth 비교)
./run_evaluate.sh \
  ./results/ksponspeech_eval_clean_predictions.jsonl \
  ./output/ksponspeech_eval_clean.jsonl \
  ./results
```

### 출력 파일

```
results/
├── ksponspeech_eval_clean_predictions.jsonl      # 추론 결과
├── ksponspeech_eval_clean_eval_results.jsonl     # 평가 상세 결과
└── ksponspeech_eval_clean_eval_summary.json      # 평가 요약
```

### 평가 요약 예시

```json
{
  "prediction_file": "./results/ksponspeech_eval_clean_predictions.jsonl",
  "gt_file": "./output/ksponspeech_eval_clean.jsonl",
  "total_samples": 3000,
  "total_cer": 0.1234,
  "total_edit_distance": 5678,
  "total_ref_length": 45678,
  "normalization": "구두점 제거 + 공백 제거"
}
```

---

## 📝 정규화 규칙

CER 계산 시 다음 정규화가 GT와 hypothesis **양쪽 모두**에 적용됩니다:

### 1. 구두점 제거

모든 구두점 및 특수문자를 제거하여 **한글과 아라비아 숫자만** 유지합니다.
ASR 모델이 구두점을 생략하거나 삽입하는 차이가 CER에 반영되지 않도록 합니다.

| Ground Truth | 모델 예측 | CER | 설명 |
|-------------|----------|-----|-----|
| `안녕하세요!` | `안녕하세요` | 0% ✅ | 느낌표 제거 |
| `네, 알겠습니다.` | `네알겠습니다` | 0% ✅ | 쉼표+마침표 제거 |

### 2. 공백 제거

모든 공백을 제거합니다.
한국어 띄어쓰기는 원어민도 일관되지 않으므로 오류로 간주하지 않습니다.

| Ground Truth | 모델 예측 | CER | 설명 |
|-------------|----------|-----|-----|
| `감사 합니다` | `감사합니다` | 0% ✅ | 공백 제거 |
| `세 잔` | `세잔` | 0% ✅ | 공백 제거 |
| `오백 원` | `오천원` | >0% ❌ | 공백 제거 후 불일치 |

---

## 📋 전체 실행 예시

```bash
# 현재 위치: Ko-Speech-Eval/ASR

# 1. 전처리 (KsponSpeech eval_clean, 영어→한국어 자동 매핑)
./run_preprocess.sh ksponspeech \
  /data/KsponSpeech_eval/eval_clean.trn \
  /data/KsponSpeech_eval/eval_clean \
  ./output/ksponspeech_wav \
  ./output/ksponspeech_eval_clean.jsonl \
  clean

# 1-2. 전처리 (KsponSpeech eval_other, 영어→한국어 자동 매핑)
./run_preprocess.sh ksponspeech \
  /data/KsponSpeech_eval/eval_other.trn \
  /data/KsponSpeech_eval/eval_other \
  ./output/ksponspeech_wav \
  ./output/ksponspeech_eval_other.jsonl \
  other

# 2. 추론
cd ../src
./run_inference.sh asr \
  ../ASR/output/ksponspeech_eval_clean.jsonl \
  ../ASR/results/ksponspeech_eval_clean_predictions.jsonl \
  /models/Qwen2-Audio-7B-Instruct

# 3. 평가
cd ../ASR
./run_evaluate.sh \
  ./results/ksponspeech_eval_clean_predictions.jsonl \
  ./output/ksponspeech_eval_clean.jsonl \
  ./results

# 4. 결과 확인
cat ./results/ksponspeech_eval_clean_eval_summary.json | jq '.total_cer'
```

---

## 🐛 문제 해결

### CUDA 메모리 부족

추론 시 `max_samples` 옵션으로 샘플 수를 제한하여 테스트:

```bash
cd ../src
./run_inference.sh asr input.jsonl output.jsonl /path/to/model 100
```

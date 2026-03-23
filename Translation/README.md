# 한국어 Speech Translation 평가

Qwen2-Audio 모델을 이용한 영어→한국어 음성 번역(Speech Translation) 평가 도구입니다.

## 📁 폴더 구조

```
Translation/
├── README.md                        # 이 문서
├── preprocess_translation.py        # 데이터 전처리 스크립트
├── run_preprocess.sh                # 전처리 실행 스크립트
├── run_translation_evaluation.py    # 번역 평가 스크립트 (추론 + 평가)
├── run_translation_eval.sh          # 평가 실행 스크립트
├── korean_normalizer.py             # 한국어 텍스트 정규화 및 메트릭 계산
├── output/                          # 전처리된 JSONL 파일들
└── results/                         # 평가 결과
```

---

## 📊 데이터셋

### ETRI EnKoST-C (English-Korean Speech Translation)

ETRI에서 제공하는 TED 기반 영어→한국어 음성 번역 데이터셋입니다.

| 데이터셋 | 설명 | TED Talks | 샘플 수 | 오디오 길이 |
|---------|------|-----------|---------|-------------|
| `etri_common` | tst-COMMON (표준 테스트셋) | 27개 | 2,532 | ~4시간 2분 |
| `etri_he` | tst-HE (추가 테스트셋) | 11개 | 544 | ~1시간 4분 |
| `etri` | 전체 (COMMON + HE) | 38개 | 3,076 | ~5시간 6분 |

#### 데이터셋 차이점

- **`etri_common` (tst-COMMON)**
  - 베이스라인 시스템 평가에 사용되는 **표준 테스트셋**
  - 더 많은 TED talks (27개)와 샘플 수 (2,532개)로 구성
  - 일반적인 성능 측정 및 벤치마킹에 사용
  
- **`etri_he` (tst-HE)**  
  - 베이스라인 훈련 및 평가에 사용되지 않는 **추가 테스트셋**
  - 상대적으로 적은 TED talks (11개)와 샘플 수 (544개)

---

## 🚀 빠른 시작

### 1. 전처리

```bash
# ETRI tst-COMMON 전처리
./run_preprocess.sh etri_common \
  /path/to/ETRI_translate_data1/en-ko/data/tst-COMMON \
  ./output/etri_tst-COMMON_processed.jsonl

# ETRI tst-HE 전처리
./run_preprocess.sh etri_he \
  /path/to/ETRI_translate_data1/en-ko/data/tst-HE \
  ./output/etri_tst-HE_processed.jsonl
```

### 2. 평가 (추론 + 메트릭 계산)

```bash
# ETRI tst-COMMON 평가 (배치 처리)
./run_translation_eval.sh etri_common \
  /path/to/Qwen2-Audio-7B-Instruct \
  ./results \
  "" \
  character \
  4

# 또는 Python 직접 실행
python run_translation_evaluation.py \
  --input ./output/etri_tst-COMMON_processed.jsonl \
  --output-dir ./results \
  --model /path/to/Qwen2-Audio-7B-Instruct \
  --gt-field answer_ko \
  --batch-size 4
```

### 3. 결과 확인

```bash
cat ./results/etri_tst-COMMON_processed_translation_summary.json
```

---

## 📐 평가 메트릭

모든 평가에서 다음 3가지 메트릭이 자동으로 계산됩니다:

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| **BLEU** | N-gram 기반 번역 품질 (Corpus BLEU, BLEU-1~4) | 0-100 |
| **METEOR** | 동의어/어간 고려한 번역 품질 | 0-100 |
| **BERTScore** | BERT 기반 의미적 유사도 (xlm-roberta-large) | 0-100 |

---

## 🔧 상세 사용법

세부 옵션은 스크립트 `run_translation_eval.sh` 및 `run_translation_evaluation.py --help`를 참고하세요.

---

## 📋 출력 형식

### 상세 결과 (JSONL)

```json
{
  "index": "000000",
  "audio_path": "/path/to/audio.wav",
  "ground_truth": "뉴욕에선, 저는 로빈 후드라는 비영리단체의 개발 책임자입니다.",
  "prediction": "뉴욕에 돌아와서, 저는 로비드 헤드 개발자입니다.",
  "bleu": 18.46,
  "bleu1": 70.00,
  "bleu2": 42.11,
  "bleu3": 22.22,
  "bleu4": 5.88,
  "meteor": 49.98,
  "bertscore_f1": 90.84
}
```

### 요약 결과 (JSON)

```json
{
  "dataset": "etri_tst-COMMON_processed.jsonl",
  "model": "/path/to/Qwen2-Audio-7B-Instruct",
  "total_samples": 2532,
  "corpus_bleu": 20.42,
  "avg_meteor": 43.53,
  "avg_bertscore_f1": 90.84,
  "elapsed_time_seconds": 6330.5
}
```

### 결과 출력 예시

```
============================================================
Translation 평가 완료: etri_tst-COMMON_processed
============================================================
총 샘플: 2532
토크나이징: character
------------------------------------------------------------
BLEU 점수:
  Corpus BLEU: 20.42
  BLEU-1: 43.47
  BLEU-2: 25.16
  BLEU-3: 16.55
  BLEU-4: 10.76
  Average BLEU: 19.42
------------------------------------------------------------
METEOR 점수: 43.53
------------------------------------------------------------
BERTScore F1: 90.84
------------------------------------------------------------

============================================================
=== ETRI EnKoST 점수 요약 ===
============================================================
Dataset                              BLEU     METEOR  BERTScore
------------------------------------------------------------
etri_tst-COMMON_processed           20.42      43.53      90.84
------------------------------------------------------------
```

---

## 📝 JSONL 형식

### 입력 형식 (전처리 후)

```json
{
  "index": "000000",
  "raw": "/path/to/audio.wav",
  "offset": 12.6,
  "duration": 4.08,
  "prompt": "이 오디오를 한국어로 번역해 주세요.",
  "question_en": "In New York, I'm the head of development for a non-profit called Robin Hood.",
  "answer_ko": "뉴욕에선, 저는 로빈 후드라는 비영리단체의 개발 책임자입니다."
}
```

**필드 설명:**
- `raw`: 오디오 파일 경로
- `offset`: 오디오 시작 위치 (초)
- `duration`: 오디오 길이 (초)
- `question_en`: 영어 원문 (참고용)
- `answer_ko`: 한국어 번역 (Ground Truth)

---

## 🐛 문제 해결

### BERTScore 경고

```
Warning: Baseline not Found for xlm-roberta-large on ko
```

이 경고는 무시해도 됩니다. 한국어 baseline이 없어서 나타나는 경고이며, 점수 계산에는 문제가 없습니다.

### CUDA 메모리 부족

배치 크기를 줄이거나 샘플 수를 제한하여 테스트:

```bash
python run_translation_evaluation.py \
  --input ./output/etri_tst-COMMON_processed.jsonl \
  --output-dir ./results \
  --model /path/to/model \
  --gt-field answer_ko \
  --batch-size 1 \
  --max-samples 100
```

### 필요한 패키지

```bash
pip install transformers torch librosa sacrebleu nltk bert-score
```

---

## 📚 참고

- [ETRI EnKoST-C Dataset](https://aihub.or.kr)
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [BERTScore](https://github.com/Tiiiger/bert_score)

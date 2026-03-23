# Evaluation Scripts

각 스크립트 상단의 변수를 수정해서 사용

## 사용법

```bash
# 개별 실행
bash eval_asr.sh
bash eval_sqa.sh
bash eval_translation.sh
bash eval_instruct.sh

# 전체 실행
bash run_all.sh
```

## 벤치마크

### ASR
| 벤치마크 | 설명 |
|----------|------|
| clovacall_test | ClovaCall 테스트셋 |
| ksponspeech_eval_clean | KsponSpeech eval_clean |
| ksponspeech_eval_other | KsponSpeech eval_other |
| common_voice_ko_test | Common Voice 한국어 |
| zeroth_test | Zeroth Korean |

### SQA
| 벤치마크 | 설명 |
|----------|------|
| click_final | CLICK 한국어 청해 |
| kobest_boolq_test_final | KoBEST BoolQ |
| cn_college_listen_final | CN College Listening |

### Translation
| 벤치마크 | 설명 |
|----------|------|
| etri_tst-COMMON_processed | ETRI EnKoST tst-COMMON |
| etri_tst-HE_processed | ETRI EnKoST tst-HE |

### Instruct
| 벤치마크 | 설명 |
|----------|------|
| vicuna_test_translated | Vicuna Test (번역) |
| alpaca_audio | Alpaca Audio |
| openhermes_instruction | OpenHermes Instruction |
| kudge_pairwise | KUdge Pairwise |

## 주요 변수

| 변수 | 설명 |
|------|------|
| `BASE_PATH` | 프로젝트 루트 경로 |
| `MODEL_PATH` | 모델 경로 (HuggingFace 또는 로컬) |
| `OUTPUT_BASE` | 결과 저장 베이스 디렉토리 |
| `BACKEND` | 모델 백엔드 (`qwen`, `llama` 등) |
| `BENCHMARKS` | 실행할 벤치마크 배열 |
| `MAX_SAMPLES` | 테스트용 샘플 수 제한 (빈 값이면 전체) |

## 결과 저장 구조

```
results/
├── ASR/
│   ├── clovacall_test/
│   ├── ksponspeech_eval_clean/
│   └── ...
├── SQA/
│   ├── click_final/
│   ├── kobest_boolq_test_final/
│   └── ...
├── Translation/
│   ├── etri_tst-COMMON_processed/
│   └── ...
└── Instruct/
    ├── vicuna_test_translated/
    └── ...
```

# Ko-Speech-Eval

한국어 음성 기반 모델 평가 저장소입니다.  

- 지원 태스크: `ASR`, `SQA`, `K-SAT`, `LSQA`, `K-disentQA`, `Translation`, `Instruct`
- 공통 구조: 각 태스크의 `*.py` 평가 스크립트 + 평가용 `*.jsonl`
- 백엔드 구조: `src/backends.py`에서 모델 백엔드를 이름으로 등록 후 공통 평가 스크립트에서 호출

---

## 0) 데이터 획득 정책 (가장 중요)

평가를 시작하기 전에, 데이터 획득 경로가 태스크별로 다릅니다.

- `ASR`, `Translation`:
  - 이 두 태스크는 전처리 코드가 포함되어 있으며, **반드시 원본 데이터 제작자/배포처에서 직접 다운로드**해야 합니다.
  - 관련 전처리 스크립트는 `ASR/preprocess_korean_asr.py`, `Translation/preprocess_korean_asr.py` 등을 참고하세요.
  - 세부 전처리/실행 가이드는 `ASR/README.md`, `Translation/README.md`를 우선 참고하세요.
  - 특히 `ASR/ksponspeech_eval_clean.jsonl`, `ASR/ksponspeech_eval_other.jsonl`은 **직접 전처리로 생성해야 하는 로컬 산출물**이며 `.gitignore`에 포함되어 있습니다.
- 그 외 태스크(`SQA`, `K-SAT`, `PA-QA`, `SCA-QA`, `Instruct`):
  - **Hugging Face에 업로드된 데이터에서 다운로드**해 사용해야 합니다.

> 요약: `ASR/Translation = 원본 소스에서 직접 다운로드`, `나머지 = Hugging Face 다운로드`

### Data Access Policy (Most Important)

Before running evaluations, note that data sources are split by task:

- `ASR`, `Translation`:
  - Preprocessing code is included, and datasets must be downloaded directly from the original data creators/providers.
  - For detailed preprocessing/evaluation usage, check `ASR/README.md` and `Translation/README.md`.
  - In particular, `ASR/ksponspeech_eval_clean.jsonl` and `ASR/ksponspeech_eval_other.jsonl` are **locally generated preprocessing outputs** and are intentionally git-ignored.
- Other tasks (`SQA`, `K-SAT`, `PA-QA`, `SCA-QA`, `Instruct`):
  - Datasets should be downloaded from Hugging Face.

> Summary: `ASR/Translation = original source`, `others = Hugging Face`

---

## 1) 빠른 실행

### Gemini Flash 전체 실행

```bash
bash scripts/evaluation/gemini-flash/run_all.sh
```

### 태스크별 실행 (Gemini Flash)

```bash
bash scripts/evaluation/gemini-flash/eval_asr.sh
bash scripts/evaluation/gemini-flash/eval_sqa.sh
bash scripts/evaluation/gemini-flash/eval_lsqa.sh
bash scripts/evaluation/gemini-flash/eval_kdisentqa.sh
bash scripts/evaluation/gemini-flash/eval_kdisentqa_3cond.sh
bash scripts/evaluation/gemini-flash/eval_translation.sh
bash scripts/evaluation/gemini-flash/eval_instruct.sh
```

### K-SAT 별도 실행

```bash
bash scripts/evaluation/eval_ksat.sh
```

---

## 2) 실행 전 반드시 확인할 것

각 쉘 스크립트 상단 변수 값을 환경에 맞게 바꿔야 합니다.

- `BASE_PATH`: 현재 로컬 프로젝트 절대 경로
- `BACKEND`: 사용할 백엔드 이름 (`gemini_flash`, `qwen`, `qwen3_vllm` 등)
- `MODEL_PATH`: 모델 식별자 또는 경로
- `PROMPT_FILE`, `PROMPT_NAME`: 프롬프트 설정
- `MAX_SAMPLES`, `BATCH_SIZE`, `TENSOR_PARALLEL_SIZE`: 실행 규모/자원 설정
- API 키 파일 사용 시: `gemini_key.txt`, `openai_key.txt` 경로/환경변수 설정

> 참고: 현재 정리 과정에서 루트의 키 파일은 삭제했으므로, 필요하면 다시 생성하거나 환경변수로 직접 주입하세요.

---

## 3) 프로젝트 구조 (정리본)

```text
Ko-Speech_eval/
├── ASR/
│   ├── run_asr_evaluation.py
│   ├── evaluate_asr.py
│   ├── korean_normalizer.py
│   └── *.jsonl
├── Instruct/
│   ├── evaluate_instruct.py
│   └── *.jsonl
├── SCA-QA/
│   ├── evaluate.py
│   ├── evaluate_with_original.py
│   └── *_clean.jsonl
├── PA-QA/
│   ├── evaluate_lsqa.py
│   └── mctest_*_filtered.jsonl
├── SQA/
│   ├── evaluate_sqa.py
│   ├── evaluate_ksat.py
│   └── *.jsonl
├── Translation/
│   ├── run_translation_evaluation.py
│   ├── evaluate_translation.py
│   ├── korean_normalizer.py
│   └── output/*.jsonl
├── src/
│   ├── backends.py
│   ├── *_inference.py
│   └── dataloaders/
├── scripts/evaluation/
│   ├── eval_ksat.sh
│   ├── eval_kdisentqa_3cond.sh
│   └── <model-family>/*.sh
└── prompts.yaml
```

---

## 4) 새 모델(백엔드) 추가 방법

모델 추가는 항상 아래 3단계로 진행합니다.

### Step A. 백엔드 구현 파일 추가

`src/` 아래에 `<ModelName>_inference.py` 파일을 만들고, 기존 파일 형식에 맞춰 클래스/함수를 구현합니다.

기존 구현 예시:
- `src/Gemini_flash_inference.py`
- `src/Qwen_2_Audio_inference.py`
- `src/Qwen3_Omni_inference.py`
- `src/voxtral_3B_inference.py`

최소한 공통 평가 코드가 기대하는 인터페이스를 제공해야 합니다.
- 생성형 추론 메서드 (음성 + 프롬프트 입력)
- 필요 시 next-token logit 메서드 (`SQA`, `K-SAT`, `LSQA`에서 사용)

### Step B. `src/backends.py` 등록

`src/backends.py`에서:
- import 추가
- 백엔드 이름 문자열(예: `"my_model"`)과 구현 클래스를 매핑에 등록

이 등록이 되어야 `--backend my_model` 또는 쉘의 `BACKEND="my_model"`이 동작합니다.

### Step C. 실행 스크립트 추가/복제

`scripts/evaluation/` 아래에 모델 폴더를 만들고(예: `my-model/`), 기존 폴더(예: `gemini-flash/`)를 복제해 변수만 수정하는 방식이 가장 안전합니다.

필수 스크립트:
- `eval_asr.sh`
- `eval_sqa.sh`
- `eval_lsqa.sh`
- `eval_kdisentqa.sh`
- `eval_kdisentqa_3cond.sh`
- `eval_translation.sh`
- `eval_instruct.sh`
- `run_all.sh` (선택이지만 권장)

---

## 5) 평가 JSONL 기준

현재 저장소에서 평가에 사용되는 JSONL은 아래 스크립트 기준으로 고정되어 있습니다.

- `scripts/evaluation/gemini-flash/*.sh`에서 참조하는 JSONL 전부
- `scripts/evaluation/eval_ksat.sh`에서 참조하는 JSONL 전부

즉, 위 스크립트에 등장하지 않는 JSONL은 정리 대상이며, 실행 중 필요한 파일은 이미 남겨져 있습니다.

---

## 6) 태스크별 실행 엔트리 (Python)

직접 실행이 필요하면 아래 스크립트를 호출하면 됩니다.

- ASR: `ASR/run_asr_evaluation.py`
- SQA: `SQA/evaluate_sqa.py`
- K-SAT: `SQA/evaluate_ksat.py`
- LSQA: `PA-QA/evaluate_lsqa.py`
- SCA-QA: `SCA-QA/evaluate.py`
- SCA-QA 3조건: `SCA-QA/evaluate_with_original.py`
- Translation: `Translation/run_translation_evaluation.py`
- Instruct: `Instruct/evaluate_instruct.py`

공통적으로 `PYTHONPATH`에 `src`가 포함되어야 합니다.

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

---

## 7) 트러블슈팅

- `JSONL not found`:
  - 스크립트의 `BASE_PATH`가 현재 경로와 다를 가능성이 큽니다.
  - `INPUT_JSONL` 경로 구성 로직을 확인하세요.
- 백엔드 인식 실패:
  - `src/backends.py`에 백엔드 이름 등록 여부 확인
  - 백엔드 구현 파일 import 에러 확인
- Instruct 평가 실패:
  - OpenAI API 키 설정 확인 (`OPENAI_API_KEY` 또는 키 파일)
- OOM:
  - `BATCH_SIZE`를 1로 낮추고 `MAX_SAMPLES`로 샘플 제한 후 점진적으로 확대

## 데이터 고지

- 본 저장소는 평가 코드/스크립트 중심이며, 데이터 배포 권한은 원본 제작자 정책을 따릅니다.
- 데이터가 필요한 경우 원본 제작자에게 직접 요청해야 합니다.

---

## English

This repository is a Korean speech / multimodal evaluation framework.

It provides task-specific JSONL inputs and shared inference backends registered in `src/backends.py`. Each evaluation task has its own Python entrypoint and is typically launched via shell scripts under `scripts/evaluation/`.

### Quick Start (Gemini Flash)

Run all tasks:
```bash
bash scripts/evaluation/gemini-flash/run_all.sh
```

Run 1 sample per task (recommended for debugging missing paths / `raw` audio issues):
```bash
bash scripts/evaluation/gemini-flash/run_all_1sample.sh
```

Task-by-task:
```bash
bash scripts/evaluation/gemini-flash/eval_asr.sh
bash scripts/evaluation/gemini-flash/eval_sqa.sh
bash scripts/evaluation/gemini-flash/eval_lsqa.sh
bash scripts/evaluation/gemini-flash/eval_kdisentqa.sh
bash scripts/evaluation/gemini-flash/eval_kdisentqa_3cond.sh
bash scripts/evaluation/gemini-flash/eval_translation.sh
bash scripts/evaluation/gemini-flash/eval_instruct.sh
```

### Smoke Test (Path Check Only)

```bash
bash scripts/evaluation/gemini-flash/smoke_test_paths.sh
```

### Before Running

1. Update variables at the top of each `scripts/evaluation/**/eval_*.sh`:
   - `BASE_PATH`: absolute path to this project on your machine
   - `BACKEND`: backend name (e.g. `gemini_flash`)
   - `MODEL_PATH`: model identifier/path (backend-specific)
   - `PROMPT_FILE`, `PROMPT_NAME`
   - `MAX_SAMPLES`, `BATCH_SIZE`, `TENSOR_PARALLEL_SIZE`
2. Set Gemini API key:
   - either `export GEMINI_API_KEY=...`
   - or create `gemini_key.txt` at the project root (excluded by `.gitignore`)

### Adding a New Backend

1. Implement `<ModelName>_inference.py` under `src/` with the required inference interface.
2. Register the backend name in `src/backends.py`.
3. Add/duplicate shell scripts under `scripts/evaluation/<your-family>/` to wire `BACKEND`, `MODEL_PATH`, and JSONL benchmarks.

### Important Note about Audio

Even if JSONL files and scripts are correct, inference requires the audio files referenced by the `raw` fields inside the JSONL. If those audio files are missing, evaluations will fail or skip samples.

### Data Availability Notice

- This repository focuses on evaluation code/scripts and does not imply redistribution rights for the data.
- If you need the data, please contact the original creators directly.

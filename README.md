# KoALa-Bench: Evaluating Large Audio Language Models on Korean Speech Understanding and Faithfulness 

<p align="center">
  <img src="main_figure.png" width="80%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>
  <a href="https://huggingface.co/datasets/scailaboratory/KoALA"><img src="https://img.shields.io/badge/🤗-HuggingFace-yellow"></a>
  <a href="https://리더보드URL"><img src="https://img.shields.io/badge/🏆-Leaderboard-blue"></a>
</p>

한국어 음성 기반 모델 평가 저장소입니다.  

- 지원 태스크: `ASR`, `SQA`, `K-SAT`, `LSQA`, `K-disentQA`, `Translation`, `Instruct`
- 공통 구조: 각 태스크의 `*.py` 평가 스크립트 + 평가용 `*.jsonl`
- 백엔드 구조: `src/backends.py`에서 모델 백엔드를 이름으로 등록 후 공통 평가 스크립트에서 호출

---

## 0) 데이터 다운로드 정책 (가장 중요)

평가를 시작하기 전에, 데이터 다운로드 경로가 태스크별로 다릅니다.

- `ASR`, `Translation`:
  - 이 두 태스크는 전처리 코드가 포함되어 있으며, **반드시 원본 데이터 제작자/배포처에서 직접 다운로드**해야 합니다.
  - 관련 전처리 스크립트는 `ASR/preprocess_korean_asr.py`, `Translation/preprocess_korean_asr.py` 등을 참고하세요.
  - 세부 전처리/실행 가이드는 `ASR/README.md`, `Translation/README.md`를 우선 참고하세요.
  - 특히, `ASR/ksponspeech_eval_clean.jsonl`, `ASR/ksponspeech_eval_other.jsonl`은 **직접 전처리** 해야합니다.
- 그 외 태스크(`SQA`, `K-SAT`, `PA-QA`, `SCA-QA`, `Instruct`):
  - **Hugging Face에 업로드된 데이터에서 다운로드**해 사용해야 합니다.

> 정리: `ASR/Translation = 원본 소스에서 직접 다운로드`, `나머지 = Hugging Face 다운로드`
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
---

## 2) 실행 전 반드시 확인할 것

각 쉘 스크립트 상단 변수 값을 환경에 맞게 바꿔야 합니다.

- `BASE_PATH`: 현재 로컬 프로젝트 절대 경로
- `BACKEND`: 사용할 백엔드 이름 (`gemini_flash`, `qwen`, `qwen3_vllm` 등)
- `MODEL_PATH`: 모델 식별자 또는 경로
- `PROMPT_FILE`, `PROMPT_NAME`: 프롬프트 설정
- `MAX_SAMPLES`, `BATCH_SIZE`, `TENSOR_PARALLEL_SIZE`: 실행 규모/자원 설정
- API 키 파일 사용 시: `gemini_key.txt`, `openai_key.txt` 경로/환경변수 설정

> 참고: 평가하고자 하는 모델의 추론 코드와 Backend에 추가하여야 합니다.

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
## 5) 태스크별 실행 엔트리 (Python)

직접 실행이 필요하면 아래 스크립트를 호출하면 됩니다.

- ASR: `ASR/run_asr_evaluation.py`
- SQA: `SQA/evaluate_sqa.py`
- K-SAT: `SQA/evaluate_ksat.py`
- LSQA: `PA-QA/evaluate_lsqa.py`
- SCA-QA: `SCA-QA/evaluate.py`
- SCA-QA 3조건: `SCA-QA/evaluate_with_original.py`
- Translation: `Translation/run_translation_evaluation.py`
- Instruct: `Instruct/evaluate_instruct.py`
---

## 7) 트러블슈팅

- Audio Missing
  - 오디오가 다운받아졌는지, jsonl파일의 경로와 일치하는지 확인해야 합니다
- `JSONL not found`:
  - 스크립트의 `BASE_PATH`가 현재 경로와 다를 가능성이 큽니다.
  - `INPUT_JSONL` 경로 구성 로직을 확인하세요.
- 백엔드 인식 실패:
  - `src/backends.py`에 백엔드 이름 등록 여부 확인
  - 백엔드 구현 파일 import 에러 확인
- Instruct 평가 실패:
  - OpenAI API 키 설정 확인 (`OPENAI_API_KEY` 또는 키 파일)


## 데이터 고지

- 본 저장소는 평가 코드/스크립트 중심이며, 데이터 배포 권한은 원본 제작자 정책을 따릅니다.
- 데이터가 필요한 경우 원본 제작자에게 직접 요청해야 합니다.

---

## Licensing

> **Important:** KoALa-Bench is constructed from multiple source datasets, each with its own license. Users must comply with the license terms of each original source.

| Task | Original Datasets | Original License |
|------|-------------|--------|
| ASR | commonVoice | CC0 1.0|
| ASR | zeroth_korean | CC BY 4.0|
| SQA | CLIcK | Other|
| SQA |Kobest-BoolQ | CC-BY-SA-4.0 |
| SIF|alpaca|CC BY-NC 4.0|
| SIF|kudge|Academic-only|
| SIF|openhermes|Academic-only|
| SIF|vicuna|Apache License 2.0|
| ST |ETRI|CC BY-NC-ND 4.0|
|SCA-QA|-|-|
|PA-QA|MCTest|MSR-LA|

### Source Datasets
Please also cite the original datasets used in KoALa-Bench:
```bibtex

----------------ASR----------------
@article{DBLP:journals/corr/abs-1912-06670,
  author       = {Rosana Ardila and
                  Megan Branson and
                  Kelly Davis and
                  Michael Henretty and
                  Michael Kohler and
                  Josh Meyer and
                  Reuben Morais and
                  Lindsay Saunders and
                  Francis M. Tyers and
                  Gregor Weber},
  title        = {Common Voice: {A} Massively-Multilingual Speech Corpus},
  journal      = {CoRR},
  volume       = {abs/1912.06670},
  year         = {2019},
  url          = {http://arxiv.org/abs/1912.06670},
  eprinttype   = {arXiv},
  eprint       = {1912.06670},
  timestamp    = {Thu, 02 Jan 2020 18:08:18 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1912-06670.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
@misc{zeroth_korean,
  title        = {Zeroth-Korean: Korean Open-source Speech Corpus for Speech Recognition},
  author       = {{Zeroth Project}},
  howpublished = {\url{https://www.openslr.org/40/}},
  note         = {OpenSLR SLR40},
  year         = {2018}
}
----------------SQA----------------
@misc{kim2024click,
      title={CLIcK: A Benchmark Dataset of Cultural and Linguistic Intelligence in Korean}, 
      author={Eunsu Kim and Juyoung Suk and Philhoon Oh and Haneul Yoo and James Thorne and Alice Oh},
      year={2024},
      eprint={2403.06412},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@misc{https://doi.org/10.48550/arxiv.2204.04541,
  doi = {10.48550/ARXIV.2204.04541},
  url = {https://arxiv.org/abs/2204.04541},
  author = {Kim, Dohyeong and Jang, Myeongjun and Kwon, Deuk Sin and Davis, Eric},
  title = {KOBEST: Korean Balanced Evaluation of Significant Tasks},
  publisher = {arXiv},
  year = {2022},
}
----------------SIF----------------
@article{son2024llm,
  title={LLM-as-a-Judge \& Reward Model: What They Can and Cannot Do},
  author={Son, Guijin and Ko, Hyunwoo and Lee, Hoyoung and Kim, Yewon and Hong, Seunghyeok},
  journal={arXiv preprint arXiv:2409.11239},
  year={2024}
}
@article{wang2024audiobench,
  title={AudioBench: A Universal Benchmark for Audio Large Language Models},
  author={Wang, Bin and Zou, Xunlong and Lin, Geyu and Sun, Shuo and Liu, Zhuohan and Zhang, Wenyu and Liu, Zhengyuan and Aw, AiTi and Chen, Nancy F},
  journal={NAACL},
  year={2025}
}
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://lmsys.org/blog/2023-03-30-vicuna/},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}
----------------PA-QA----------------
@inproceedings{richardson-etal-2013-mctest,
    title = "{MCT}est: A Challenge Dataset for the Open-Domain Machine Comprehension of Text",
    author = "Richardson, Matthew  and
      Burges, Christopher J.C.  and
      Renshaw, Erin",
    editor = "Yarowsky, David  and
      Baldwin, Timothy  and
      Korhonen, Anna  and
      Livescu, Karen  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D13-1020/",
    pages = "193--203"
}

```


## Citation

If you use KoALa-Bench in your research, please cite:

```bibtex
----
```

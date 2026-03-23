#!/usr/bin/env python3
"""
Instruct (Instruction Following) DataLoader

JSONL 형식:
{
    "index": "000000",
    "raw": "",
    "prompt": "연설에서 제시된 질문을 참고해 주시기 바랍니다.",
    "question_ko": "19세기 말에 사설 서당의 확산으로 문해율이 상승하기 시작했습니다...",
    "answer_ko": "19세기 초부터 말까지 문해율이 얼마나 증가했는지 정확한 수치로 나타내기는 어렵습니다..."
}

추론에 사용하는 필드:
- raw: 오디오 파일 경로
- prompt: 텍스트 프롬프트
- question_ko: 질문 (모델이 답변해야 할 질문)
- answer_ko: 정답 (평가용)
"""

from typing import Dict
from pathlib import Path
from .base import BaseDataLoader


class InstructDataLoader(BaseDataLoader):
    """Instruct용 DataLoader"""
    
    DEFAULT_INSTRUCT_PROMPT = "다음 질문에 답변해 주세요."
    
    def __init__(self, jsonl_path: str, max_samples: int = None, custom_prompt: str = None, base_dir: str = None):
        """
        Args:
            jsonl_path: 입력 JSONL 파일 경로
            max_samples: 최대 샘플 수
            custom_prompt: 커스텀 프롬프트 (None이면 JSONL의 prompt 필드 또는 기본값 사용)
            base_dir: 오디오 파일 기준 디렉토리 (None이면 JSONL 파일 디렉토리/[데이터셋명] 사용)
        """
        super().__init__(jsonl_path, max_samples)
        self.custom_prompt = custom_prompt
        # 오디오 파일 기준 디렉토리: base_dir 우선, 없으면 JSONL 파일 디렉토리/[데이터셋명]
        if base_dir:
            self.audio_base_dir = Path(base_dir).resolve()
        else:
            jsonl_path_obj = Path(jsonl_path)
            dataset_name = jsonl_path_obj.stem  # 확장자 제거 (예: kudge_pairwise.jsonl -> kudge_pairwise)
            self.audio_base_dir = (jsonl_path_obj.parent / dataset_name).resolve()
    
    @property
    def task_name(self) -> str:
        return "instruct"
    
    @property
    def default_prompt(self) -> str:
        return self.DEFAULT_INSTRUCT_PROMPT
    
    def process_item(self, item: Dict) -> Dict:
        """
        Instruct 아이템 처리
        
        Returns:
            - index: 샘플 인덱스
            - audio_path: 오디오 파일 경로
            - text_input: 텍스트 인풋 (prompt + question_ko)
            - prompt: 텍스트 프롬프트
            - question_ko: 질문
            - answer_ko: 정답 (평가용)
        """
        index = item.get("index", "")
        raw_path = item.get("raw", item.get("audio_path", ""))
        
        # raw 키가 없거나 비어있을 때: base_dir + index.wav
        if not raw_path or raw_path.strip() == "":
            if index:
                audio_path = str(self.audio_base_dir / f"{index}.wav")
            else:
                audio_path = ""
        else:
            audio_path = raw_path
        
        question_ko = (item.get("question_ko") or "").strip()
        answer_ko = (item.get("answer_ko") or "").strip()
        
        # 프롬프트 결정 우선순위:
        # 1. 커스텀 프롬프트 (명령줄에서 지정)
        # 2. JSONL의 prompt 필드
        # 3. 기본 Instruct 프롬프트
        if self.custom_prompt:
            prompt = self.custom_prompt
        else:
            prompt = (item.get("prompt") or "").strip()
        if not prompt:
            prompt = self.DEFAULT_INSTRUCT_PROMPT
        
        # 텍스트 인풋 = prompt + question_ko (모델에 줄 텍스트)
        text_input = prompt
        if question_ko:
            text_input = f"{prompt}\n\n{question_ko}" if prompt else question_ko
        
        return {
            "index": index,
            "audio_path": audio_path,
            "text_input": text_input,
            "prompt": prompt,
            "question_ko": question_ko,
            "answer_ko": answer_ko,
        }


#!/usr/bin/env python3
"""
SQA (Spoken Question Answering) DataLoader

인풋 구성:
- 음성 인풋: raw 오디오 (question_ko 내용이 TTS로 된 음성 = 질문/대화)
- 텍스트 인풋: prompt + choices_ko (지시문 + 선택지)

JSONL 형식:
{
    "index": "000001",
    "raw": "audio/dream_tts_tts_merged/000001.wav",
    "prompt": "그 사람에 따르면, ...",
    "question_ko": "대체 에너지원이라는 용어는 ...",
    "choices_ko": "(A) ...\n(B) ...\n(C) ...",
    "answer_ko": "(C)"
}
"""

from typing import Dict
from .base import BaseDataLoader


class SQADataLoader(BaseDataLoader):
    """SQA(Spoken QA)용 DataLoader"""

    DEFAULT_SQA_PROMPT = "다음 음성을 듣고 질문에 맞는 답을 고르세요."

    def __init__(self, jsonl_path: str, max_samples: int = None, custom_prompt: str = None):
        """
        Args:
            jsonl_path: 입력 JSONL 파일 경로
            max_samples: 최대 샘플 수
            custom_prompt: 커스텀 프롬프트 (None이면 JSONL의 prompt 필드 또는 기본값 사용)
        """
        super().__init__(jsonl_path, max_samples)
        self.custom_prompt = custom_prompt

    @property
    def task_name(self) -> str:
        return "sqa"

    @property
    def default_prompt(self) -> str:
        return self.DEFAULT_SQA_PROMPT

    def process_item(self, item: Dict) -> Dict:
        """
        SQA 아이템 처리

        Returns:
            - index: 샘플 인덱스
            - audio_path: 음성 인풋 경로 (질문/대화 TTS)
            - text_input: 텍스트 인풋 (prompt + choices_ko)
            - prompt, question_ko, choices_ko, answer_ko: 참고/평가용
        """
        index = item.get("index", "")
        audio_path = item.get("raw", item.get("audio_path", ""))
        question_ko = (item.get("question_ko") or "").strip()
        choices_ko = (item.get("choices_ko") or "").strip()
        answer_ko = (item.get("answer_ko") or "").strip()

        if self.custom_prompt:
            prompt = self.custom_prompt
        else:
            prompt = (item.get("prompt") or "").strip()
        if not prompt:
            prompt = self.DEFAULT_SQA_PROMPT

        # 텍스트 인풋 = prompt + choices_ko (모델에 줄 텍스트)
        text_input = prompt
        if choices_ko:
            text_input = f"{prompt}\n\n{choices_ko}" if prompt else choices_ko

        return {
            "index": index,
            "audio_path": audio_path,  # 음성 인풋 (question_ko의 TTS)
            "text_input": text_input,  # 텍스트 인풋 (prompt + choices_ko)
            "prompt": prompt,
            "question_ko": question_ko,
            "choices_ko": choices_ko,
            "answer_ko": answer_ko,
        }

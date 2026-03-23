#!/usr/bin/env python3
"""
ASR (Automatic Speech Recognition) DataLoader

JSONL 형식:
{
    "index": "000000",
    "raw": "/path/to/audio.wav",
    "prompt": "Transcribe the following audio to Korean. Output only the transcribed Korean text without any explanations or additional content:",
    "question_ko": "4명이요.",
    "speaker_id": "03380"
}

추론에 사용하는 필드:
- raw: 오디오 파일 경로
- prompt: 텍스트 프롬프트
- index: 샘플 인덱스 (출력용)
"""

from typing import Dict
from .base import BaseDataLoader


class ASRDataLoader(BaseDataLoader):
    """ASR용 DataLoader"""
    
    DEFAULT_ASR_PROMPT = "Transcribe the following audio to Korean. Output only the transcribed Korean text without any explanations or additional content:"
    
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
        return "asr"
    
    @property
    def default_prompt(self) -> str:
        return self.DEFAULT_ASR_PROMPT
    
    def process_item(self, item: Dict) -> Dict:
        """
        ASR 아이템 처리
        
        Returns:
            - index: 샘플 인덱스
            - audio_path: 오디오 파일 경로
            - prompt: 텍스트 프롬프트
        """
        # 인덱스
        index = item.get('index', '')
        
        # 오디오 경로 (raw 또는 audio_path)
        audio_path = item.get('raw', item.get('audio_path', ''))
        
        # 프롬프트 결정 우선순위:
        # 1. 커스텀 프롬프트 (명령줄에서 지정)
        # 2. JSONL의 prompt 필드
        # 3. 기본 ASR 프롬프트
        if self.custom_prompt:
            prompt = self.custom_prompt
        else:
            prompt = item.get('prompt', self.DEFAULT_ASR_PROMPT)
        
        return {
            'index': index,
            'audio_path': audio_path,
            'prompt': prompt,
        }

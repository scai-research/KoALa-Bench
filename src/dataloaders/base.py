#!/usr/bin/env python3
"""
DataLoader 기본 인터페이스

모든 task별 dataloader는 이 인터페이스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator
import json


class BaseDataLoader(ABC):
    """DataLoader 기본 클래스"""
    
    def __init__(self, jsonl_path: str, max_samples: int = None):
        """
        Args:
            jsonl_path: 입력 JSONL 파일 경로
            max_samples: 최대 샘플 수 (None이면 전체)
        """
        self.jsonl_path = jsonl_path
        self.max_samples = max_samples
        self.data = self._load_jsonl()
    
    def _load_jsonl(self) -> List[Dict]:
        """JSONL 파일 로드"""
        data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        if self.max_samples:
            data = data[:self.max_samples]
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> Iterator[Dict]:
        for item in self.data:
            yield self.process_item(item)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.process_item(self.data[idx])
    
    @abstractmethod
    def process_item(self, item: Dict) -> Dict:
        """
        단일 아이템 처리 (각 task별로 구현)
        
        Returns:
            Dict with at least:
            - index: 샘플 인덱스
            - audio_path: 오디오 파일 경로
            - prompt: 텍스트 프롬프트
        """
        pass
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task 이름 반환"""
        pass
    
    @property
    def default_prompt(self) -> str:
        """기본 프롬프트 (task별로 오버라이드 가능)"""
        return "이 오디오의 내용을 설명해 주세요."

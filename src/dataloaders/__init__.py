#!/usr/bin/env python3
"""
Task별 DataLoader 모음

사용법:
    from dataloaders import get_dataloader
    
    loader = get_dataloader('asr', 'input.jsonl', max_samples=10)
    for item in loader:
        print(item['audio_path'], item['prompt'])
"""

from .base import BaseDataLoader
from .asr import ASRDataLoader
from .sqa import SQADataLoader
from .instruct import InstructDataLoader

# Task 이름 → DataLoader 클래스 매핑
DATALOADER_REGISTRY = {
    'asr': ASRDataLoader,
    'sqa': SQADataLoader,
    'instruct': InstructDataLoader,
}


def get_dataloader(task: str, jsonl_path: str, max_samples: int = None, **kwargs) -> BaseDataLoader:
    """
    Task에 맞는 DataLoader 반환
    
    Args:
        task: task 이름 ('asr', 'sqa', 'emotion', 'speaker' 등)
        jsonl_path: JSONL 파일 경로
        max_samples: 최대 샘플 수
        **kwargs: DataLoader별 추가 인자
    
    Returns:
        BaseDataLoader 인스턴스
    
    Raises:
        ValueError: 지원하지 않는 task인 경우
    """
    task = task.lower()
    
    if task not in DATALOADER_REGISTRY:
        available = ', '.join(DATALOADER_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 task: '{task}'. 사용 가능: {available}")
    
    loader_class = DATALOADER_REGISTRY[task]
    return loader_class(jsonl_path, max_samples=max_samples, **kwargs)


def list_available_tasks() -> list:
    """사용 가능한 task 목록 반환"""
    return list(DATALOADER_REGISTRY.keys())


__all__ = [
    'BaseDataLoader',
    'ASRDataLoader',
    'SQADataLoader',
    'InstructDataLoader',
    'get_dataloader',
    'list_available_tasks',
    'DATALOADER_REGISTRY',
]

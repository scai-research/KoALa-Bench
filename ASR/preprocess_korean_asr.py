#!/usr/bin/env python3
"""
한국어 ASR 데이터셋 전처리 스크립트

지원 데이터셋:
- KsponSpeech: eval_clean.trn, eval_other.trn (pcm → wav 변환, --split 지정 시 영어→한국어 자동 매핑)
- Common Voice: test.tsv (mp3 → wav 변환)
- Zeroth Korean: 폴더 구조 (flac → wav 변환)

출력: JSONL 파일
"""

import os
import json
import re
import argparse
import struct
import wave
from pathlib import Path
from typing import Dict, List, Optional

# ASR 기본 프롬프트
DEFAULT_ASR_PROMPT = "Transcribe the following audio to Korean. Output only the transcribed Korean text without any explanations or additional content:"

# 오디오 변환을 위한 라이브러리
try:
    import soundfile as sf
    import numpy as np
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# pydub는 mp3용 백업 (ffmpeg 필요)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# miniaudio는 ffmpeg 없이 mp3/flac 처리 가능
try:
    import miniaudio
    MINIAUDIO_AVAILABLE = True
except ImportError:
    MINIAUDIO_AVAILABLE = False


def convert_pcm_to_wav(pcm_path: str, wav_path: str, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bool:
    """
    PCM 파일을 WAV로 변환 (순수 Python)
    
    Args:
        pcm_path: 입력 PCM 파일 경로
        wav_path: 출력 WAV 파일 경로
        sample_rate: 샘플레이트 (기본 16kHz)
        channels: 채널 수 (기본 1=mono)
        sample_width: 샘플 너비 바이트 (기본 2=16bit)
    """
    try:
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        
        with open(pcm_path, 'rb') as pcm_file:
            pcm_data = pcm_file.read()
        
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        return True
    except Exception as e:
        print(f"PCM→WAV 변환 오류 [{pcm_path}]: {e}")
        return False


def convert_flac_to_wav(flac_path: str, wav_path: str) -> bool:
    """
    FLAC 파일을 WAV로 변환
    
    우선순위:
    1. soundfile
    2. miniaudio (ffmpeg 불필요)
    """
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    if SOUNDFILE_AVAILABLE:
        try:
            data, samplerate = sf.read(flac_path)

            if len(data.shape) > 1:
                data = data.mean(axis=1)

            sf.write(wav_path, data, samplerate, subtype='PCM_16')
            return True

        except Exception as e:
            print(f"FLAC→WAV 변환 오류 (soundfile) [{flac_path}]: {e}")

    if MINIAUDIO_AVAILABLE:
        try:
            decoded = miniaudio.decode_file(flac_path, output_format=miniaudio.SampleFormat.SIGNED16)
            samples = np.frombuffer(decoded.samples, dtype=np.int16)

            if decoded.nchannels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(decoded.sample_rate)
                wav_file.writeframes(samples.tobytes())

            return True

        except Exception as e:
            print(f"FLAC→WAV 변환 오류 (miniaudio) [{flac_path}]: {e}")
            return False

    print(f"오류: FLAC 변환 라이브러리가 없습니다. 'pip install soundfile' 또는 'pip install miniaudio' 실행 필요")
    return False


def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """
    MP3 파일을 WAV로 변환
    
    우선순위:
    1. miniaudio (ffmpeg 불필요)
    2. pydub (ffmpeg 필요)
    """
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    
    # 방법 1: miniaudio 사용 (ffmpeg 불필요)
    if MINIAUDIO_AVAILABLE:
        try:
            # MP3 디코딩
            decoded = miniaudio.decode_file(mp3_path, output_format=miniaudio.SampleFormat.SIGNED16)
            samples = np.frombuffer(decoded.samples, dtype=np.int16)
            
            # Stereo → Mono 변환
            if decoded.nchannels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # 16kHz 리샘플링 (필요시)
            if decoded.sample_rate != 16000:
                # 간단한 리샘플링 (선형 보간)
                original_length = len(samples)
                target_length = int(original_length * 16000 / decoded.sample_rate)
                indices = np.linspace(0, original_length - 1, target_length)
                samples = np.interp(indices, np.arange(original_length), samples).astype(np.int16)
            
            # WAV로 저장
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(samples.tobytes())
            
            return True
            
        except Exception as e:
            print(f"MP3→WAV 변환 오류 (miniaudio) [{mp3_path}]: {e}")
            # miniaudio 실패 시 pydub 시도
    
    # 방법 2: pydub 사용 (ffmpeg 필요)
    if PYDUB_AVAILABLE:
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(wav_path, format='wav')
            return True

        except Exception as e:
            print(f"MP3→WAV 변환 오류 (pydub) [{mp3_path}]: {e}")
            return False

    print(f"오류: MP3 변환 라이브러리가 없습니다. 'pip install miniaudio' 또는 'pip install pydub' + ffmpeg 설치 필요")
    return False


def clean_ksponspeech_text(text: str) -> str:
    """
    KsponSpeech 텍스트에서 특수 표기 정리
    """
    # (원문)/(발음) 형식에서 발음만 추출
    text = re.sub(r'\([^)]+\)/\(([^)]+)\)', r'\1', text)
    text = re.sub(r'\(([^)]+)\)/\(([^)]+)\)', r'\2', text)
    
    # o/, b/, l/, n/, u/ 등 노이즈 마커 제거
    text = re.sub(r'[obnlu]/\s*', '', text)
    text = re.sub(r'(\S)/\s', r'\1 ', text)
    text = re.sub(r'\s*l/$', '', text)
    
    # 다중 공백을 단일 공백으로
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# KsponSpeech 텍스트 치환 매핑 (WAV 파일명 기반)
# 전사 텍스트에 포함된 영어 약어/숫자를 한국어로 자동 치환
KSPONSPEECH_ENGLISH_MAP = {
    "eval_clean": {
        "KsponSpeech_E00100": {"A": "에이"},
        "KsponSpeech_E00174": {"OT": "오티"},
        "KsponSpeech_E00227": {"S": "에쓰"},
        "KsponSpeech_E00371": {"B": "삐"},
        "KsponSpeech_E00640": {"VIP": "브이아이피"},
        "KsponSpeech_E00692": {"SKT": "에스케이티"},
        "KsponSpeech_E00741": {"SBS": "에쓰비에쓰"},
        "KsponSpeech_E01149": {"TV": "티비"},
        "KsponSpeech_E01290": {"SUV": "에쓰유브이"},
        "KsponSpeech_E01412": {"2": "이"},
        "KsponSpeech_E01594": {"GAP": "갭"},
        "KsponSpeech_E01785": {"OT": "오티"},
        "KsponSpeech_E02079": {"SKT": "에쓰케이티"},
        "KsponSpeech_E02102": {"TV": "티비"},
        "KsponSpeech_E02435": {"PPT": "피피티"},
        "KsponSpeech_E02441": {"TV": "티비"},
    },
    "eval_other": {
        "KsponSpeech_E03294": {"TV": "티비"},
        "KsponSpeech_E03502": {"PC": "피씨"},
        "KsponSpeech_E04674": {"S": "에쓰"},
        "KsponSpeech_E04825": {"TV": "티비"},
        "KsponSpeech_E04953": {"SUV": "에쓰유브이"},
        "KsponSpeech_E05093": {"KT": "케이티"},
        "KsponSpeech_E05104": {"TV": "티비"},
        "KsponSpeech_E05179": {"CU": "씨유"},
        "KsponSpeech_E05395": {"TV": "티비"},
        "KsponSpeech_E05456": {"CG": "씨지"},
        "KsponSpeech_E05743": {"u": "어"},
        "KsponSpeech_E05824": {"u": "그게"},
        "KsponSpeech_E05963": {"A": "에이", "B": "비"},
        "KsponSpeech_E05979": {"TV": "티비"},
    },
}


def apply_english_mapping(text: str, word_map: Dict[str, str]) -> str:
    """텍스트에서 영어 토큰을 한국어 발음으로 치환 (긴 토큰 우선)"""
    for eng in sorted(word_map.keys(), key=len, reverse=True):
        text = text.replace(eng, word_map[eng])
    return text


def process_ksponspeech(
    trn_path: str,
    audio_root: str,
    output_path: str,
    wav_output_dir: str = None,
    split: str = None
) -> int:
    """KsponSpeech 데이터셋 처리 (PCM → WAV 변환, 영어→한국어 자동 매핑)"""
    print(f"\n[KsponSpeech] 처리 시작...")
    print(f"  입력: {trn_path}")
    print(f"  음원: {audio_root}")
    print(f"  출력: {output_path}")
    
    eng_map = {}
    if split:
        split_key = f"eval_{split}"
        eng_map = KSPONSPEECH_ENGLISH_MAP.get(split_key, {})
        print(f"  스플릿: {split_key} (영어→한국어 매핑 {len(eng_map)}건 적용)")
    
    if wav_output_dir:
        print(f"  WAV 출력: {wav_output_dir}")
        os.makedirs(wav_output_dir, exist_ok=True)
    
    count = 0
    converted = 0
    skipped = 0
    eng_replaced = 0
    
    with open(trn_path, 'r', encoding='utf-8') as f, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(f):
            line = line.strip()
            if not line or '::' not in line:
                continue
            
            parts = line.split(' :: ')
            if len(parts) != 2:
                continue
            
            pcm_path_raw, text = parts
            filename = os.path.basename(pcm_path_raw).replace('.pcm', '')
            
            pcm_path = os.path.join(audio_root, f"{filename}.pcm")
            
            if wav_output_dir:
                wav_path = os.path.join(wav_output_dir, f"{filename}.wav")
            else:
                wav_path = os.path.join(audio_root, f"{filename}.wav")
            
            if not os.path.exists(wav_path):
                if os.path.exists(pcm_path):
                    if convert_pcm_to_wav(pcm_path, wav_path):
                        converted += 1
                    else:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue
            
            cleaned_text = clean_ksponspeech_text(text)
            
            if filename in eng_map:
                cleaned_text = apply_english_mapping(cleaned_text, eng_map[filename])
                eng_replaced += 1
            
            new_entry = {
                "index": f"{i:06d}",
                "raw": wav_path,
                "prompt": DEFAULT_ASR_PROMPT,
                "question_ko": cleaned_text,
                "speaker_id": ""
            }
            
            outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"  완료: {count}개 샘플 처리됨")
    if converted > 0:
        print(f"  변환: {converted}개 PCM → WAV")
    if eng_replaced > 0:
        print(f"  영어→한국어 치환: {eng_replaced}개 샘플")
    if skipped > 0:
        print(f"  스킵: {skipped}개 (파일 없음 또는 변환 실패)")
    return count


def process_commonvoice(
    tsv_path: str,
    audio_root: str,
    output_path: str,
    wav_output_dir: str = None
) -> int:
    """Common Voice 데이터셋 처리 (MP3 → WAV 변환)"""
    print(f"\n[Common Voice] 처리 시작...")
    print(f"  입력: {tsv_path}")
    print(f"  음원: {audio_root}")
    print(f"  출력: {output_path}")
    
    if wav_output_dir:
        print(f"  WAV 출력: {wav_output_dir}")
        os.makedirs(wav_output_dir, exist_ok=True)
    
    count = 0
    converted = 0
    skipped = 0
    
    with open(tsv_path, 'r', encoding='utf-8') as f, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        header = f.readline().strip().split('\t')
        
        try:
            path_idx = header.index('path')
            sentence_idx = header.index('sentence')
            client_id_idx = header.index('client_id')
        except ValueError as e:
            print(f"  오류: 필수 컬럼을 찾을 수 없습니다 - {e}")
            return 0
        
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) <= max(path_idx, sentence_idx, client_id_idx):
                continue
            
            mp3_filename = parts[path_idx]
            text = parts[sentence_idx]
            client_id = parts[client_id_idx]
            
            mp3_path = os.path.join(audio_root, mp3_filename)
            
            wav_filename = mp3_filename.replace('.mp3', '.wav')
            if wav_output_dir:
                wav_path = os.path.join(wav_output_dir, wav_filename)
            else:
                wav_path = os.path.join(audio_root, wav_filename)
            
            # WAV가 이미 있으면 변환 스킵
            if not os.path.exists(wav_path):
                if os.path.exists(mp3_path):
                    if convert_mp3_to_wav(mp3_path, wav_path):
                        converted += 1
                    else:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue
            
            speaker_id = client_id[:8] if client_id else ""
            
            new_entry = {
                "index": f"{i:06d}",
                "raw": wav_path,
                "prompt": DEFAULT_ASR_PROMPT,
                "question_ko": text,
                "speaker_id": speaker_id
            }
            
            outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"  완료: {count}개 샘플 처리됨")
    if converted > 0:
        print(f"  변환: {converted}개 MP3 → WAV")
    if skipped > 0:
        print(f"  스킵: {skipped}개 (파일 없음 또는 변환 실패)")
    return count


def process_zeroth_korean(
    data_root: str,
    audio_root: str,
    output_path: str,
    wav_output_dir: str = None
) -> int:
    """Zeroth Korean 데이터셋 처리 (FLAC → WAV 변환)"""
    print(f"\n[Zeroth Korean] 처리 시작...")
    print(f"  입력: {data_root}")
    print(f"  음원: {audio_root}")
    print(f"  출력: {output_path}")
    
    if wav_output_dir:
        print(f"  WAV 출력: {wav_output_dir}")
        os.makedirs(wav_output_dir, exist_ok=True)
    
    # 모든 trans.txt 파일 찾기
    trans_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.trans.txt'):
                trans_files.append(os.path.join(root, file))
    
    trans_files.sort()
    print(f"  발견된 trans.txt 파일: {len(trans_files)}개")
    
    count = 0
    converted = 0
    skipped = 0
    entries = []
    
    for trans_path in trans_files:
        trans_dir = os.path.dirname(trans_path)
        
        with open(trans_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                
                file_id, text = parts
                
                rel_dir = os.path.relpath(trans_dir, data_root)
                
                flac_path = os.path.join(audio_root, rel_dir, f"{file_id}.flac")
                
                if wav_output_dir:
                    wav_subdir = os.path.join(wav_output_dir, rel_dir)
                    os.makedirs(wav_subdir, exist_ok=True)
                    wav_path = os.path.join(wav_subdir, f"{file_id}.wav")
                else:
                    wav_path = os.path.join(audio_root, rel_dir, f"{file_id}.wav")
                
                # WAV가 이미 있으면 변환 스킵
                if not os.path.exists(wav_path):
                    if os.path.exists(flac_path):
                        if convert_flac_to_wav(flac_path, wav_path):
                            converted += 1
                        else:
                            skipped += 1
                            continue
                    else:
                        skipped += 1
                        continue
                
                id_parts = file_id.split('_')
                speaker_id = '_'.join(id_parts[:2]) if len(id_parts) >= 2 else file_id
                
                entries.append({
                    "file_id": file_id,
                    "audio_path": wav_path,
                    "text": text,
                    "speaker_id": speaker_id
                })
    
    entries.sort(key=lambda x: x['file_id'])
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i, entry in enumerate(entries):
            new_entry = {
                "index": f"{i:06d}",
                "raw": entry['audio_path'],
                "prompt": DEFAULT_ASR_PROMPT,
                "question_ko": entry['text'],
                "speaker_id": entry['speaker_id']
            }
            
            outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"  완료: {count}개 샘플 처리됨")
    if converted > 0:
        print(f"  변환: {converted}개 FLAC → WAV")
    if skipped > 0:
        print(f"  스킵: {skipped}개 (파일 없음 또는 변환 실패)")
    return count


def main():
    parser = argparse.ArgumentParser(
        description='한국어 ASR 데이터셋 전처리 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # KsponSpeech eval_clean (pcm → wav 변환 + 영어→한국어 자동 매핑)
  python preprocess_korean_asr.py ksponspeech \\
    --input /path/to/eval_clean.trn \\
    --audio-root /path/to/eval_clean \\
    --wav-output /path/to/wav_output \\
    --split clean \\
    --output ./ksponspeech_eval_clean.jsonl

  # KsponSpeech eval_other (pcm → wav 변환 + 영어→한국어 자동 매핑)
  python preprocess_korean_asr.py ksponspeech \\
    --input /path/to/eval_other.trn \\
    --audio-root /path/to/eval_other \\
    --wav-output /path/to/wav_output \\
    --split other \\
    --output ./ksponspeech_eval_other.jsonl

  # Common Voice (mp3 → wav 변환)
  python preprocess_korean_asr.py commonvoice \\
    --input /path/to/test.tsv \\
    --audio-root /path/to/clips \\
    --wav-output /path/to/wav_output \\
    --output ./common_voice_ko_test.jsonl

  # Zeroth Korean (flac → wav 변환)
  python preprocess_korean_asr.py zeroth \\
    --input /path/to/test_data_01/003 \\
    --audio-root /path/to/test_data_01/003 \\
    --wav-output /path/to/wav_output \\
    --output ./zeroth_korean_test.jsonl
        """
    )
    
    parser.add_argument('dataset', choices=['ksponspeech', 'commonvoice', 'zeroth'],
                        help='처리할 데이터셋 종류')
    parser.add_argument('--input', '-i', required=True,
                        help='입력 파일 또는 디렉토리 경로')
    parser.add_argument('--audio-root', '-a', required=True,
                        help='원본 음원 파일이 있는 루트 디렉토리')
    parser.add_argument('--output', '-o', required=True,
                        help='출력 JSONL 파일 경로')
    parser.add_argument('--wav-output', '-w', default=None,
                        help='변환된 WAV 파일 저장 디렉토리 (미지정 시 audio-root에 저장)')
    parser.add_argument('--split', '-s', choices=['clean', 'other'], default=None,
                        help='KsponSpeech 스플릿 (clean 또는 other 지정 시 영어→한국어 자동 매핑 적용)')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset == 'ksponspeech':
        count = process_ksponspeech(args.input, args.audio_root, args.output, args.wav_output, args.split)
    elif args.dataset == 'commonvoice':
        count = process_commonvoice(args.input, args.audio_root, args.output, args.wav_output)
    elif args.dataset == 'zeroth':
        count = process_zeroth_korean(args.input, args.audio_root, args.output, args.wav_output)
    else:
        print(f"지원하지 않는 데이터셋: {args.dataset}")
        return 1
    
    print(f"\n전체 완료: {count}개 샘플")
    return 0


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
한국어 ASR 데이터셋 전처리 스크립트

지원 데이터셋:
- ClovaCall: test_ClovaCall.json (wav 그대로 사용)
- KsponSpeech: eval_clean.trn, eval_other.trn (pcm → wav 변환)
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
DEFAULT_ASR_PROMPT = "이 오디오를 한국어로 받아쓰기해 주세요."

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
    FLAC 파일을 WAV로 변환 (soundfile 사용)
    """
    if not SOUNDFILE_AVAILABLE:
        print(f"오류: soundfile이 설치되어 있지 않습니다. 'pip install soundfile' 실행 필요")
        return False
    
    try:
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        
        # FLAC 읽기
        data, samplerate = sf.read(flac_path)
        
        # 16kHz, mono로 변환 (필요시)
        if len(data.shape) > 1:  # stereo → mono
            data = data.mean(axis=1)
        
        # 16kHz 리샘플링 (간단한 방식, 필요시)
        # 대부분의 음성 데이터는 이미 16kHz이므로 그대로 저장
        
        # WAV로 저장 (16bit PCM)
        sf.write(wav_path, data, samplerate, subtype='PCM_16')
        return True
        
    except Exception as e:
        print(f"FLAC→WAV 변환 오류 [{flac_path}]: {e}")
        return False


def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """
    MP3 파일을 WAV로 변환 (pydub 사용, ffmpeg 필요)
    """
    if not PYDUB_AVAILABLE:
        print(f"오류: pydub가 설치되어 있지 않습니다. 'pip install pydub' 실행 필요")
        return False
    
    try:
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        
        audio = AudioSegment.from_mp3(mp3_path)
        # 16kHz, 16bit, mono로 변환
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format='wav')
        return True
        
    except Exception as e:
        print(f"MP3→WAV 변환 오류 [{mp3_path}]: {e}")
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


def process_clovacall(
    json_path: str,
    audio_root: str,
    output_path: str,
    wav_output_dir: str = None
) -> int:
    """ClovaCall 데이터셋 처리 (wav 그대로 사용)"""
    print(f"\n[ClovaCall] 처리 시작...")
    print(f"  입력: {json_path}")
    print(f"  음원: {audio_root}")
    print(f"  출력: {output_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i, entry in enumerate(data):
            wav_filename = entry.get('wav', '')
            text = entry.get('text', '')
            speaker_id = entry.get('speaker_id', '')
            
            audio_path = os.path.join(audio_root, wav_filename)
            
            new_entry = {
                "index": f"{i:06d}",
                "raw": audio_path,
                "prompt": DEFAULT_ASR_PROMPT,
                "question_ko": text,
                "speaker_id": speaker_id
            }
            
            outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"  완료: {count}개 샘플 처리됨")
    return count


def process_ksponspeech(
    trn_path: str,
    audio_root: str,
    output_path: str,
    wav_output_dir: str = None
) -> int:
    """KsponSpeech 데이터셋 처리 (PCM → WAV 변환)"""
    print(f"\n[KsponSpeech] 처리 시작...")
    print(f"  입력: {trn_path}")
    print(f"  음원: {audio_root}")
    print(f"  출력: {output_path}")
    
    if wav_output_dir:
        print(f"  WAV 출력: {wav_output_dir}")
        os.makedirs(wav_output_dir, exist_ok=True)
    
    count = 0
    converted = 0
    skipped = 0
    
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
            
            # WAV가 이미 있으면 변환 스킵
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
  # ClovaCall (wav 그대로 사용)
  python preprocess_korean_asr.py clovacall \\
    --input /path/to/test_ClovaCall.json \\
    --audio-root /path/to/wavs_test \\
    --output ./clovacall_test.jsonl

  # KsponSpeech (pcm → wav 변환)
  python preprocess_korean_asr.py ksponspeech \\
    --input /path/to/eval_clean.trn \\
    --audio-root /path/to/eval_clean \\
    --wav-output /path/to/wav_output \\
    --output ./ksponspeech_eval_clean.jsonl

  # Common Voice (mp3 → wav 변환, ffmpeg 필요)
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
    
    parser.add_argument('dataset', choices=['clovacall', 'ksponspeech', 'commonvoice', 'zeroth'],
                        help='처리할 데이터셋 종류')
    parser.add_argument('--input', '-i', required=True,
                        help='입력 파일 또는 디렉토리 경로')
    parser.add_argument('--audio-root', '-a', required=True,
                        help='원본 음원 파일이 있는 루트 디렉토리')
    parser.add_argument('--output', '-o', required=True,
                        help='출력 JSONL 파일 경로')
    parser.add_argument('--wav-output', '-w', default=None,
                        help='변환된 WAV 파일 저장 디렉토리 (미지정 시 audio-root에 저장)')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if args.dataset == 'clovacall':
        count = process_clovacall(args.input, args.audio_root, args.output, args.wav_output)
    elif args.dataset == 'ksponspeech':
        count = process_ksponspeech(args.input, args.audio_root, args.output, args.wav_output)
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

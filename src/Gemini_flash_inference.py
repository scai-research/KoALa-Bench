# Gemini API 오디오 입력 추론
#
# 모델: gemini-2.0-flash-lite (gemini-3.1-flash-lite-preview 등)
# pip install google-generativeai
#
# 환경변수 GEMINI_API_KEY 또는 파일 경로 GEMINI_API_KEY_FILE 사용.

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _load_api_key(path: Optional[str] = None) -> Optional[str]:
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = path or os.environ.get("GEMINI_API_KEY_FILE", "")
    if key_file and Path(key_file).is_file():
        try:
            return Path(key_file).read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except Exception:
            pass
    return None


def _get_audio_segment(
    audio_path: str,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    """offset/duration 있으면 librosa로 구간 잘라 임시 wav 반환. (경로, 정리할_임시경로)"""
    use_seg = (offset is not None and offset != 0.0) or (duration is not None and duration > 0)
    if not use_seg:
        return (audio_path, None)
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        print(f"[Gemini] librosa/soundfile 필요: {e}")
        return (audio_path, None)
    try:
        kwargs: Dict[str, Any] = {"sr": 16000, "offset": float(offset)}
        if duration is not None and duration > 0:
            kwargs["duration"] = float(duration)
        audio, sr = librosa.load(audio_path, **kwargs)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp, audio, sr)
        return (tmp, tmp)
    except Exception as e:
        print(f"[Gemini] 오디오 구간 로드 실패 [{audio_path}]: {e}")
        return (audio_path, None)


def _mime_from_path(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {"wav": "audio/wav", "mp3": "audio/mp3", "flac": "audio/flac",
            "ogg": "audio/ogg", "m4a": "audio/mp4"}.get(ext, "audio/wav")


class GeminiFlashModel:
    """
    Gemini Flash / Flash-Lite 오디오 입력 추론 래퍼.

    인터페이스: GPTRealtimeMiniModel과 동일
      - inference(audio_path, prompt, ...) -> str
      - inference_batch(items, ...) -> List[str]
      - get_next_token_logits / get_next_token_logits_batch -> None (미지원)
    """

    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, model_path: str = "", **kwargs):
        model = (model_path or "").strip() or self.DEFAULT_MODEL
        self.model_path = model

        # API 키: 환경변수 > GEMINI_API_KEY_FILE 파일 > Ko-Speech-Eval/gemini_key.txt
        _script_dir = Path(__file__).resolve().parent.parent
        _default_key_file = str(_script_dir / "gemini_key.txt")
        api_key = _load_api_key(_default_key_file)
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY 환경변수 또는 gemini_key.txt 가 없습니다."
            )
        self._api_key = api_key

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._client = genai.GenerativeModel(model_name=model)
        except ImportError:
            raise ImportError("pip install google-generativeai")

        print(f"[Gemini] 모델 초기화 완료: {model}")

    def _call(self, audio_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        mime = _mime_from_path(audio_path)
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            print(f"[Gemini] 오디오 읽기 실패 [{audio_path}]: {e}")
            return ""
        if not audio_bytes:
            print(f"[Gemini] 오디오 비어 있음: {audio_path}")
            return ""

        audio_part = {"mime_type": mime, "data": audio_bytes}
        try:
            response = self._client.generate_content(
                [audio_part, prompt],
                generation_config=self._genai.GenerationConfig(
                    max_output_tokens=max_new_tokens,
                    temperature=0.0,
                ),
            )
            text = response.text if hasattr(response, "text") else ""
            return (text or "").strip()
        except Exception as e:
            print(f"[Gemini] generate_content 실패 [{audio_path}]: {e}")
            return ""

    def inference(
        self,
        audio_path: str,
        prompt: str,
        max_new_tokens: int = 256,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> str:
        if not os.path.exists(audio_path):
            print(f"[Gemini] 오디오 없음: {audio_path}")
            return ""
        use_path, tmp_path = _get_audio_segment(audio_path, offset=offset, duration=duration)
        try:
            return self._call(use_path, prompt, max_new_tokens)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def inference_batch(
        self,
        items: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        return_first_logits: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[None]]]:
        results: List[str] = []
        for item in items:
            audio_path = item.get("audio_path", "")
            prompt = item.get("prompt", item.get("text_input", ""))
            offset = item.get("offset", 0.0)
            duration = item.get("duration")
            result = self.inference(
                audio_path, prompt,
                max_new_tokens=max_new_tokens,
                offset=offset, duration=duration,
            )
            results.append(result)
        if return_first_logits:
            return (results, [None] * len(results))
        return results

    def get_next_token_logits(
        self, audio_path: str, text_input: str, answer_suffix: str = "\n답: "
    ) -> Optional[Dict[int, float]]:
        return None

    def get_next_token_logits_batch(
        self, items: List[Dict[str, Any]], answer_suffix: str = "\n답: "
    ) -> List[Optional[Dict[int, float]]]:
        return [None] * len(items)

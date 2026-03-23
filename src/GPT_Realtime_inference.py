# OpenAI — 오디오 입력 추론 (Chat Completions / Responses API)
#
# gpt-audio-mini: Chat·Responses 에서 audio 입력 가능, batch inference 가능, 저렴
# realtime-preview 는 v1/chat/completions 비지원(404) → 사용 안 함
#
# 1) Chat Completions
#    https://platform.openai.com/docs/guides/audio?api-mode=chat
#    messages + input_audio { data, format }
#
# 2) Responses API (멀티모달) — 사용자 스펙 예시:
#    model: gpt-4o-mini
#    input: [ { role, content: [ { type: input_audio, audio_url } ] } ]
#    로컬 파일은 audio_url 대신 Files 업로드 후 file_id 를 input_file 로 넘기거나,
#    input_file + file_data (data URL) 로 시도 (File inputs 가이드).
#
# 환경변수 OPENAI_USE_RESPONSES_API=1 이면 Responses API 먼저 사용.
# kwargs: use_responses_api=True, api 모델명은 model_path 그대로 (gpt-4o-mini 유지 가능).

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Chat/Responses audio 입력 지원 · batch 가능 · 저렴 버전 우선
DEFAULT_CHAT_AUDIO_MODEL = "gpt-audio-mini"
FALLBACK_CHAT_AUDIO_MODEL = "gpt-audio"
LEGACY_AUDIO_MODEL = "gpt-4o-mini-audio-preview"

# Chat Completions: 무조건 gpt-audio-mini 사용
_CHAT_MODEL = "gpt-audio-mini"


def _load_audio_base64(
    audio_path: str,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Optional[Tuple[str, str]]:
    """
    Chat Completions input_audio: format은 wav 또는 mp3만 지원.
    offset/duration 이 있으면 해당 구간만 잘라서 wav로 인코딩해 전달 (긴 음성 자르기).
    """
    path = Path(audio_path)
    if not path.is_file():
        return None

    use_segment = (offset is not None and offset != 0.0) or (duration is not None and duration > 0)
    if use_segment:
        try:
            import librosa
            import soundfile as sf
        except ImportError as e:
            print(f"[GPT Realtime] offset/duration 사용을 위해 librosa, soundfile 필요: {e}")
            use_segment = False
        if use_segment:
            try:
                kwargs = {"sr": 16000, "offset": float(offset)}
                if duration is not None and duration > 0:
                    kwargs["duration"] = float(duration)
                audio, sr = librosa.load(str(path), **kwargs)
                import tempfile
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    sf.write(temp_path, audio, sr)
                    with open(temp_path, "rb") as f:
                        data = f.read()
                    return base64.b64encode(data).decode("utf-8"), "wav"
                finally:
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"[GPT Realtime] 오디오 구간 로드 실패 [{audio_path}]: {e}")
                return None

    ext = path.suffix.lower().lstrip(".")
    format_map = {"wav": "wav", "mp3": "mp3"}
    fmt = format_map.get(ext)
    if fmt is None:
        print(f"[unsupported audio format] {audio_path}")
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            print(f"[empty audio file] {audio_path}")
            return None
        return base64.b64encode(data).decode("utf-8"), fmt
    except Exception as e:
        print(f"[audio read error] {audio_path}: {e}")
        return None


def _get_client(api_key: Optional[str] = None):
    from openai import OpenAI
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set or pass api_key to backend")
    return OpenAI(api_key=key)


def _load_api_key_from_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        return (lines[0] if lines else "").strip() or None
    except Exception:
        return None


def _resolve_chat_model(model_path: str) -> str:
    return _CHAT_MODEL


def _message_text_chat(resp) -> str:
    if not resp or not getattr(resp, "choices", None):
        return ""
    msg = resp.choices[0].message
    if msg is None:
        return ""
    c = getattr(msg, "content", None)
    return c.strip() if isinstance(c, str) and c.strip() else ""


def _output_text_responses(resp) -> str:
    """Responses API 결과에서 텍스트만."""
    if resp is None:
        return ""
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    # 일부 SDK 는 output 리스트
    out = getattr(resp, "output", None) or []
    for item in out:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in ("output_text", "text"):
                    txt = getattr(c, "text", None) or getattr(c, "content", None)
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
    return ""


class GPTRealtimeMiniModel:
    """
    기본 모델: gpt-audio-mini (Chat/Responses audio 입력, batch, 저렴).
    realtime-preview 는 chat 엔드포인트 미지원이라 치환하지 않으면 404.
    """

    def __init__(self, model_path: str = "", **kwargs):
        self._raw_model = (model_path or "").strip() or DEFAULT_CHAT_AUDIO_MODEL
        _env = os.environ.get("OPENAI_USE_RESPONSES_API", "").lower()
        _raw_lower = self._raw_model.lower()
        if "realtime" in _raw_lower:
            self.use_responses_api = _env in ("1", "true", "yes")
        elif _env in ("0", "false", "no"):
            self.use_responses_api = False
        elif kwargs.get("use_responses_api") is False:
            self.use_responses_api = False
        elif kwargs.get("use_responses_api") is True or _env in ("1", "true", "yes"):
            self.use_responses_api = True
        else:
            # gpt-audio-mini: Responses API 권장 (client.responses.create)
            self.use_responses_api = _raw_lower in (
                "gpt-4o-mini",
                "gpt-4o-mini-2024-07-18",
                "gpt-audio-mini",
            )
        # Chat Completions 쓸 때만 별칭 적용
        self.model_path = (
            self._raw_model if self.use_responses_api else _resolve_chat_model(self._raw_model)
        )
        self._processor = None
        self._api_key = (kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY") or "").strip() or None
        if not self._api_key:
            key_file = (
                kwargs.get("api_key_file")
                or os.environ.get("OPENAI_API_KEY_FILE")
                or str(Path(__file__).resolve().parents[1] / "openai_key.txt")
            )
            self._api_key = _load_api_key_from_file(key_file)
        self._client = None
        # speech→text 전용; audio output 옵션(modalities, audio) 불필요
        self._batch_delay = float(kwargs.get("batch_delay", 0))

    @property
    def processor(self):
        return self._processor

    def _client_or_new(self):
        if self._client is None:
            self._client = _get_client(self._api_key)
        return self._client

    def _inference_responses_api(
        self, client, audio_path: str, prompt: str, b64: str, fmt: str, max_tokens: int
    ) -> str:
        """
        Responses API (client.responses.create).
        공식 형식: input_audio with audio (base64) + format, text with type "text".
        """
        model = self.model_path  # gpt-audio-mini 등 그대로 사용
        text = (prompt or "").strip()
        fmt_val = fmt or "wav"

        def _call(content):
            return client.responses.create(
                model=model,
                input=[{"role": "user", "content": content}],
                max_output_tokens=max_tokens,
                temperature=0,
            )

        # 1) input_audio { data, format } + input_text (안정적 형식)
        try:
            content = [
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": fmt_val},
                },
                {"type": "input_text", "text": text},
            ]
            resp = _call(content)
            out = _output_text_responses(resp)
            if out:
                return out
        except Exception as e:
            print(f"[GPT Realtime] Responses API (input_audio+input_text): {e}")

        # 2) fallback: input_audio + audio_url (data URL)
        mime = f"audio/{fmt_val}" if fmt_val else "audio/wav"
        data_url = f"data:{mime};base64,{b64}"
        try:
            content = [
                {"type": "input_text", "text": text},
                {"type": "input_audio", "audio_url": data_url},
            ]
            resp = _call(content)
            out = _output_text_responses(resp)
            if out:
                return out
        except Exception as e:
            print(f"[GPT Realtime] Responses API (audio_url): {e}")

        # 3) input_file + file_data (wav)
        try:
            filename = Path(audio_path).name or f"audio.{fmt_val or 'wav'}"
            content = [
                {"type": "input_text", "text": text},
                {
                    "type": "input_file",
                    "filename": filename,
                    "file_data": data_url,
                },
            ]
            resp = _call(content)
            out = _output_text_responses(resp)
            if out:
                return out
        except Exception as e:
            print(f"[GPT Realtime] Responses API (file_data): {e}")

        # 4) 업로드 후 file_id
        try:
            with open(audio_path, "rb") as f:
                fobj = client.files.create(file=f, purpose="user_data")
            fid = getattr(fobj, "id", None)
            if fid:
                content = [
                    {"type": "input_text", "text": text},
                    {"type": "input_file", "file_id": fid},
                ]
                resp = _call(content)
                out = _output_text_responses(resp)
                if out:
                    return out
        except Exception as e:
            print(f"[GPT Realtime] Responses API (file_id): {e}")

        return ""

    def _chat_create(self, client, model: str, content: list, max_tokens: int):
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        return client.chat.completions.create(**kwargs)

    def _inference_chat_completions(
        self, client, prompt: str, b64: str, fmt: str, max_new_tokens: int
    ) -> str:
        """
        input_audio 는 gpt-audio-mini / gpt-audio / audio-preview 만 지원.
        realtime-preview 는 chat completions 404 → 절대 넣지 않음.
        """
        content = [
            {"type": "text", "text": (prompt or "").strip()},
            {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
        ]
        resolved = _resolve_chat_model(self._raw_model)
        models_to_try = [resolved]
        if resolved.lower() != DEFAULT_CHAT_AUDIO_MODEL:
            models_to_try.append(DEFAULT_CHAT_AUDIO_MODEL)
        if FALLBACK_CHAT_AUDIO_MODEL not in models_to_try:
            models_to_try.append(FALLBACK_CHAT_AUDIO_MODEL)
        if LEGACY_AUDIO_MODEL not in models_to_try:
            models_to_try.append(LEGACY_AUDIO_MODEL)
        last_err = None
        for m in models_to_try:
            if m.lower() == "gpt-4o-mini" or "realtime" in m.lower():
                continue
            try:
                resp = self._chat_create(client, m, content, max_new_tokens)
                text = _message_text_chat(resp)
                if text:
                    return text
            except Exception as e:
                last_err = e
        if last_err:
            print(f"[GPT Realtime] Chat Completions error: {last_err}")
        return ""

    def _inference_transcription(self, client, audio_path: str, prompt: str) -> str:
        """
        ASR 등: Chat/Responses 가 안 되면 Transcription API 로 텍스트만 받기.
        """
        try:
            with open(audio_path, "rb") as f:
                tr = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                    prompt=(prompt or "")[:1024] or None,
                )
            text = getattr(tr, "text", None) or ""
            return text.strip() if isinstance(text, str) else ""
        except Exception:
            try:
                with open(audio_path, "rb") as f:
                    tr = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                    )
                text = getattr(tr, "text", None) or ""
                return text.strip() if isinstance(text, str) else ""
            except Exception as e:
                print(f"[GPT Realtime] Transcription error: {e}")
        return ""

    def inference(
        self,
        audio_path: str,
        prompt: str,
        max_new_tokens: int = 256,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Union[str, Tuple[str, str]]:
        """반환: str 또는 (prediction, note). note는 오디오 없음/로드 실패 시 JSONL에 기록용."""
        if not os.path.exists(audio_path):
            print(f"[audio missing] {audio_path}")
            return ("", "audio_missing")
        packed = _load_audio_base64(audio_path, offset=offset, duration=duration)
        if not packed:
            print(f"[audio load failed] {audio_path}")
            return ("", "audio_load_failed")
        b64, fmt = packed
        if not b64:
            print(f"[audio empty] {audio_path}")
            return ("", "audio_empty")
        client = self._client_or_new()
        # print(f"[DEBUG] model={self.model_path}, audio_format={fmt}, b64_len={len(b64)}, prompt={prompt[:100]}")
        out = self._inference_chat_completions(
            client, prompt, b64, fmt, max_new_tokens
        )
        return out or ""

    def inference_batch(
        self,
        items: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        return_first_logits: bool = False,
    ) -> Union[List[Union[str, Tuple[str, str]]], Tuple[List[str], List[Optional[object]]]]:
        results: List[Union[str, Tuple[str, str]]] = []
        for i, item in enumerate(items):
            audio_path = item.get("audio_path", "")
            prompt = item.get("prompt", item.get("text_input", ""))
            offset = item.get("offset", 0.0)
            duration = item.get("duration")
            if not os.path.exists(audio_path):
                print(f"[audio missing] {audio_path}")
                results.append(("", "audio_missing"))
                continue
            out = self.inference(
                audio_path, prompt, max_new_tokens=max_new_tokens,
                offset=offset, duration=duration
            )
            if isinstance(out, tuple):
                results.append(out)
            else:
                results.append(out or "")
            if self._batch_delay > 0 and i < len(items) - 1:
                time.sleep(self._batch_delay)
        if return_first_logits:
            preds = [p[0] if isinstance(p, tuple) else p for p in results]
            return (preds, [None] * len(preds))
        return results

    def get_next_token_logits(
        self, audio_path: str, text_input: str, answer_suffix: str = "\n답: "
    ) -> Optional[Dict[int, float]]:
        return None

    def get_next_token_logits_batch(
        self, items: List[Dict[str, Any]], answer_suffix: str = "\n답: "
    ) -> List[Optional[Dict[int, float]]]:
        return [None] * len(items)

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

from .models import BridgeError


@dataclass(frozen=True)
class Settings:
    asr_provider: str = "openai"
    asr_base_url: str = "https://api.openai.com/v1"
    asr_model: str = "whisper-1"
    asr_key_env: str = "OPENAI_API_KEY"
    translation_base_url: str = "https://api.openai.com/v1"
    translation_model: str = "gpt-4o-mini"
    translation_key_env: str = "OPENAI_API_KEY"
    local_model: str = "small"
    local_device: str = "cpu"
    local_compute_type: str = "int8"
    whisper_cpp_cli: str = "whisper-cli"
    whisper_cpp_model: str = (
        "~/Library/Application Support/Vimpo/models/ggml-large-v3-turbo-q5_0.bin"
    )
    moss_cli: str = "transcribe-cli"
    moss_model: str = (
        "~/Library/Application Support/Vimpo/models/MOSS-Transcribe-Diarize-Q5_K_M.gguf"
    )
    native_backend: str = "auto"
    native_threads: int = 4
    native_timeout_seconds: int = 3600
    moss_diarize: bool = True
    chunk_seconds: int = 600
    batch_size: int = 30
    context_segments: int = 3
    timeout_seconds: int = 180
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    yt_dlp: str = "yt-dlp"
    font: str = "PingFang SC"
    font_size: int = 20
    segmenter: str = "natural"
    segment_max_chars: int = 84
    segment_max_duration: float = 6.0
    segment_min_duration: float = 0.8
    segment_pause_seconds: float = 0.6

    def __post_init__(self) -> None:
        for field in fields(self):
            if isinstance(field.default, str):
                value = getattr(self, field.name)
                if not isinstance(value, str) or not value.strip():
                    raise BridgeError(f"{field.name} must be a nonempty string.")
        if self.segmenter not in {"natural", "none"}:
            raise BridgeError("segmenter must be natural or none.")
        for name, low, high in (
            ("segment_max_duration", 0.5, 30),
            ("segment_min_duration", 0, 10),
            ("segment_pause_seconds", 0.05, 10),
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not low <= value <= high:
                raise BridgeError(f"{name} must be a number between {low} and {high}.")
        if self.segment_min_duration > self.segment_max_duration:
            raise BridgeError("segment_min_duration cannot exceed segment_max_duration.")
        if self.asr_provider not in {"openai", "local", "whisper-cpp", "moss"}:
            raise BridgeError("asr_provider must be openai, local, whisper-cpp or moss.")
        if self.native_backend not in {"auto", "metal", "cpu"}:
            raise BridgeError("native_backend must be auto, metal or cpu.")
        if type(self.moss_diarize) is not bool:
            raise BridgeError("moss_diarize must be a boolean.")
        for name in ("asr_base_url", "translation_base_url"):
            parsed = urlsplit(getattr(self, name))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise BridgeError(f"{name} must be an HTTP(S) URL.")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise BridgeError(f"{name} must not contain credentials, query, or fragment.")
        for name, low, high in (
            ("chunk_seconds", 10, 600),
            ("batch_size", 1, 100),
            ("context_segments", 0, 20),
            ("timeout_seconds", 1, 3600),
            ("font_size", 8, 100),
            ("segment_max_chars", 10, 300),
            ("native_threads", 1, 256),
            ("native_timeout_seconds", 1, 86400),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise BridgeError(f"{name} must be an integer between {low} and {high}.")

    @classmethod
    def load(cls, path: Path | None) -> Settings:
        if path is None:
            return cls()
        try:
            with path.open("rb") as file:
                return cls(**tomllib.load(file))
        except (OSError, TypeError, tomllib.TOMLDecodeError) as exc:
            raise BridgeError(f"Cannot load config {path}: {exc}") from exc

    def api_key(self, kind: str) -> str:
        name = self.asr_key_env if kind == "asr" else self.translation_key_env
        value = os.environ.get(name, "").strip()
        if not value:
            raise BridgeError(f"Set environment variable {name} for {kind}.")
        return value

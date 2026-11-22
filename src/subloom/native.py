from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import unicodedata
import wave
from collections.abc import Callable
from pathlib import Path

from .models import BridgeError, Segment, Word, atomic_text
from .settings import Settings

NATIVE_PROVIDERS = {"whisper-cpp", "moss"}


def executable(settings: Settings) -> Path | None:
    whisper = settings.asr_provider == "whisper-cpp"
    value = settings.whisper_cpp_cli if whisper else settings.moss_cli
    expanded = str(Path(value).expanduser())
    found = shutil.which(expanded)
    if found:
        return Path(found).resolve()
    if Path(value).name == value and value in {"whisper-cli", "transcribe-cli"}:
        engine = "whisper.cpp" if whisper else "transcribe.cpp"
        root = Path(__file__).resolve().parents[2]
        candidate = root / ".tools" / "build" / engine / "bin" / value
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def model_path(settings: Settings) -> Path:
    value = (
        settings.whisper_cpp_model
        if settings.asr_provider == "whisper-cpp"
        else settings.moss_model
    )
    return Path(value).expanduser().resolve()


def status(settings: Settings) -> dict:
    binary = executable(settings)
    model = model_path(settings)
    return {
        "provider": settings.asr_provider,
        "model": str(model),
        "executable": str(binary) if binary else None,
        "timestamp_kind": "word" if settings.asr_provider == "whisper-cpp" else "segment",
        "speaker_attribution": settings.asr_provider == "moss" and settings.moss_diarize,
        "backend": settings.native_backend,
        "ready": binary is not None and model.is_file(),
    }


def fingerprint(settings: Settings) -> dict:
    binary, model = executable(settings), model_path(settings)
    if binary is None:
        raise BridgeError(
            f"Missing {settings.asr_provider} executable. Run "
            "uv run --extra native-build python scripts/setup_native.py, "
            "or configure its CLI path."
        )
    if not model.is_file():
        raise BridgeError(f"Local model does not exist: {model}")
    expected = b"lmgg" if settings.asr_provider == "whisper-cpp" else b"GGUF"
    with model.open("rb") as file:
        if file.read(4) != expected:
            raise BridgeError(f"Unexpected model format for {settings.asr_provider}: {model}")

    def stamp(path: Path) -> list:
        info = path.stat()
        return [str(path), info.st_size, info.st_mtime_ns]

    return {
        "adapter": 1,
        "binary": stamp(binary),
        "model": stamp(model),
        "backend": settings.native_backend,
        "threads": settings.native_threads,
        "diarize": settings.moss_diarize if settings.asr_provider == "moss" else False,
    }


def interval(start, end) -> tuple[float, float]:
    if (
        type(start) not in (int, float)
        or type(end) not in (int, float)
        or not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or end <= start
    ):
        raise BridgeError("Native engine returned invalid millisecond timestamps.")
    return start / 1000, end / 1000


def token_words(tokens: list, start: float, end: float) -> tuple[Word, ...]:
    if not isinstance(tokens, list):
        raise BridgeError("whisper.cpp returned invalid tokens.")
    groups: list[list] = []
    for token in tokens:
        if not isinstance(token, dict) or not isinstance(token.get("text"), str):
            raise BridgeError("whisper.cpp returned an invalid token.")
        text = token["text"]
        if not text.strip() or (text.startswith("[_") and text.endswith("]")):
            continue
        offsets = token.get("offsets")
        if not isinstance(offsets, dict):
            return ()
        a, b = offsets.get("from"), offsets.get("to")
        if a == b and type(a) in (int, float) and math.isfinite(a) and a >= 0:
            if groups and not any(c.isalnum() for c in text):
                groups[-1][2] += text
                continue
            return ()
        a, b = interval(a, b)
        a, b = max(start, a), min(end, b)
        if b <= a:
            return ()
        cjk = unicodedata.east_asian_width(text.lstrip()[0]) in {"W", "F"}
        lexical = any(c.isalnum() for c in text)
        boundary = text[0].isspace() or (cjk and lexical)
        if groups and (not boundary or not lexical):
            groups[-1][1] = max(groups[-1][1], b)
            groups[-1][2] += text
        else:
            groups.append([a, b, text])
    return tuple(Word(a, b, text) for a, b, text in groups)


def parse_whisper(payload: dict) -> list[Segment]:
    try:
        rows = payload["transcription"]
        if not isinstance(rows, list):
            raise TypeError("transcription must be a list")
        result = []
        for row in rows:
            text = row["text"]
            if not isinstance(text, str):
                raise TypeError("text must be a string")
            if not text.strip():
                continue
            start, end = interval(row["offsets"]["from"], row["offsets"]["to"])
            words = token_words(row.get("tokens", []), start, end)
            result.append(Segment(len(result), start, end, text.strip(), words=words))
    except (KeyError, TypeError, AttributeError) as exc:
        raise BridgeError("Invalid whisper.cpp JSON result; expected --output-json-full.") from exc
    return validate_result(result)


def parse_moss(text: str, audio: Path) -> list[Segment]:
    try:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        rows = [row for row in records if isinstance(row, dict) and "file" in row]
        if len(rows) != 1 or Path(rows[0]["file"]).resolve() != audio.resolve():
            raise ValueError("expected one result matching the input WAV")
        payload = rows[0]
        if payload.get("error"):
            raise BridgeError(f"MOSS did not finish transcription: {payload['error']}")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise BridgeError(
                "MOSS returned no timed segments. Refusing to invent timing "
                "for plain transcript text."
            )
        result = []
        for row in segments:
            value = row["text"]
            if not isinstance(value, str):
                raise ValueError("invalid segment text")
            if not value.strip():
                continue
            start, end = interval(row["t0_ms"], row["t1_ms"])
            result.append(
                Segment(len(result), start, end, value.strip(), speaker=row.get("speaker_id"))
            )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise BridgeError(f"Invalid transcribe.cpp batch JSONL: {exc}") from exc
    return validate_result(result)


def validate_result(rows: list[Segment]) -> list[Segment]:
    if not rows:
        raise BridgeError("No speech detected by the local model.")
    if any(a.start > b.start for a, b in zip(rows, rows[1:])):
        raise BridgeError("Native engine returned unordered segments.")
    return rows


def run_engine(args: list[str], stdout: Path, stderr: Path, timeout: int) -> None:
    try:
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=os.name == "posix",
            )
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    try:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        elif process.poll() is None:
                            process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
    except FileNotFoundError as exc:
        raise BridgeError(f"Missing native executable: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BridgeError(f"Native transcription timed out; see {stderr}") from exc
    if code:
        raise BridgeError(f"Native transcription exited {code}; see {stderr}")


def transcribe_native(
    audio: Path, settings: Settings, language: str | None, progress: Callable[..., None]
) -> list[Segment]:
    identity = fingerprint(settings)
    audio = audio.resolve()
    folder = audio.parent
    backend = settings.asr_provider
    log = folder / f"{backend}.stderr.log"
    stdout = folder / f"{backend}.stdout.log"
    progress(
        "progress",
        stage="transcribe",
        provider=backend,
        phase="loading_and_inference",
        timestamp_kind="word" if backend == "whisper-cpp" else "segment",
    )
    if backend == "whisper-cpp":
        prefix = folder / "whisper-cpp.raw"
        output = prefix.with_suffix(".raw.json")
        output.unlink(missing_ok=True)
        args = [
            identity["binary"][0],
            "-m",
            identity["model"][0],
            "-f",
            str(audio),
            "-l",
            language or "auto",
            "-ojf",
            "-of",
            str(prefix),
            "-t",
            str(settings.native_threads),
        ]
        if settings.native_backend == "cpu":
            args.append("--no-gpu")
        run_engine(args, stdout, log, settings.native_timeout_seconds)
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BridgeError(f"whisper.cpp did not produce valid JSON; see {log}") from exc
        result = parse_whisper(payload)
    else:
        if language and language not in {"en", "zh", "auto"}:
            raise BridgeError("MOSS supports en, zh or auto source language.")
        if "\n" in str(audio) or "\r" in str(audio):
            raise BridgeError("MOSS batch input paths cannot contain line breaks.")
        batch = folder / "moss-inputs.txt"
        atomic_text(batch, str(audio) + "\n")
        args = [
            identity["binary"][0],
            "-m",
            identity["model"][0],
            "--backend",
            settings.native_backend,
            "--timestamps",
            "segment",
            "--batch",
            str(batch),
            "--batch-size",
            "1",
            "--batch-jsonl",
            "--quiet",
            "--threads",
            str(settings.native_threads),
            "-l",
            language or "auto",
            "--diarize" if settings.moss_diarize else "--no-diarize",
        ]
        run_engine(args, stdout, log, settings.native_timeout_seconds)
        result = parse_moss(stdout.read_text(encoding="utf-8"), audio)
    with wave.open(str(audio), "rb") as source:
        length = source.getnframes() / source.getframerate()
    if any(row.end > length + 0.25 for row in result):
        raise BridgeError("Native engine returned timestamps beyond the input audio.")
    progress(
        "progress",
        stage="transcribe",
        provider=backend,
        phase="completed",
        segments=len(result),
        words=sum(len(s.words) for s in result),
    )
    return result

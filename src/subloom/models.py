from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class BridgeError(Exception):
    pass


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (self.start, self.end)):
            raise BridgeError("Word timestamps must be finite numbers.")
        if self.start < 0 or self.end <= self.start:
            raise BridgeError("Word timestamps must have a positive duration.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise BridgeError("A timestamped word must have text.")


@dataclass(frozen=True)
class Segment:
    id: int
    start: float
    end: float
    text: str
    translation: str = ""
    words: tuple[Word, ...] = ()
    speaker: int | None = None

    def __post_init__(self) -> None:
        if type(self.id) is not int or self.id < 0:
            raise BridgeError("Segment IDs must be nonnegative integers.")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (self.start, self.end)):
            raise BridgeError("Segment timestamps must be finite numbers.")
        if self.start < 0 or self.end <= self.start:
            raise BridgeError(f"Invalid timestamps for segment {self.id}.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise BridgeError(f"Segment {self.id} has no source text.")
        if self.speaker is not None and (type(self.speaker) is not int or self.speaker < 0):
            raise BridgeError("Speaker ID must be a nonnegative integer or null.")
        if not isinstance(self.translation, str):
            raise BridgeError(f"Invalid translation for segment {self.id}.")
        try:
            words = tuple(Word(**w) if isinstance(w, dict) else w for w in self.words)
        except (TypeError, ValueError) as exc:
            raise BridgeError(f"Invalid words for segment {self.id}.") from exc
        if any(not isinstance(w, Word) for w in words):
            raise BridgeError(f"Invalid words for segment {self.id}.")
        if any(w.start < self.start - 0.001 or w.end > self.end + 0.001 for w in words):
            raise BridgeError(f"Word timestamps outside segment {self.id}.")
        if any(a.start > b.start for a, b in zip(words, words[1:])):
            raise BridgeError("Words must be ordered by start time.")
        object.__setattr__(self, "words", words)


def load_segments(path: Path) -> list[Segment]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["segments"] if isinstance(data, dict) else data
        segments = [Segment(**row) for row in rows]
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise BridgeError(f"Cannot read segments from {path}: {exc}") from exc
    if not segments:
        raise BridgeError(f"No speech segments in {path}.")
    if len({s.id for s in segments}) != len(segments):
        raise BridgeError("Segment IDs must be unique.")
    if any(a.start > b.start for a, b in zip(segments, segments[1:])):
        raise BridgeError("Segments must be ordered by start time.")
    return segments


def prepare_output(
    path: Path, suffix: str, protected: tuple[Path, ...] = (), *, overwrite: bool = False
) -> Path:
    output = path.expanduser().resolve()
    if output in {p.expanduser().resolve() for p in protected}:
        raise BridgeError("Output must be different from all input files.")
    if output.suffix.lower() != suffix:
        raise BridgeError(f"Output must use the {suffix} extension.")
    if output.is_dir():
        raise BridgeError(f"Output is a directory: {output}")
    if output.exists() and not overwrite:
        raise BridgeError(f"Output already exists: {output}. Use --overwrite to replace it.")
    return output


def publish_output(temporary: Path, output: Path, *, overwrite: bool = True) -> None:
    if overwrite:
        os.replace(temporary, output)
    else:
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise BridgeError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            ) from exc


def atomic_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n", overwrite=overwrite)


def atomic_text(path: Path, value: str, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(value)
        publish_output(Path(name), path, overwrite=overwrite)
    finally:
        Path(name).unlink(missing_ok=True)


def save_segments(
    path: Path, segments: list[Segment], *, overwrite: bool = True, **metadata: Any
) -> None:
    atomic_json(path, {**metadata, "segments": [asdict(s) for s in segments]}, overwrite=overwrite)

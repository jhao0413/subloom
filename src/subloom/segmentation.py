from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace

from .models import BridgeError, Segment, Word
from .settings import Settings


@dataclass(frozen=True)
class Unit:
    start: float
    end: float
    begin: int
    stop: int
    text: str
    word: Word


def display_width(text: str) -> int:
    return sum(
        0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1
        for c in text
    )


def sentence_end(text: str) -> bool:
    value = text.rstrip().rstrip("\"'”’）)]")
    if not re.search(r"[.!?。！？]$", value):
        return False
    if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc)\.$", value, flags=re.IGNORECASE):
        return False
    if re.search(r"(?:\b[A-Za-z]\.){2,}$", value):
        return False
    return True


def timed_units(segment: Segment) -> list[Unit] | None:
    positions = []
    cursor = 0
    for word in segment.words:
        token = word.text.strip()
        match = re.search(re.escape(token), segment.text[cursor:], flags=re.IGNORECASE)
        if match is None:
            return None
        begin = cursor + match.start()
        if any(c.isalnum() for c in segment.text[cursor:begin]):
            return None
        positions.append(begin)
        cursor = cursor + match.end()
    if not positions or any(c.isalnum() for c in segment.text[cursor:]):
        return None
    positions[0] = 0
    stops = positions[1:] + [len(segment.text)]
    return [
        Unit(w.start, w.end, begin, stop, segment.text[begin:stop], w)
        for w, begin, stop in zip(segment.words, positions, stops)
    ]


def segment_transcript(
    segments: list[Segment], settings: Settings, emit: Callable[..., None] | None = None
) -> list[Segment]:
    if settings.segmenter == "none":
        return segments
    if any(s.translation for s in segments):
        raise BridgeError(
            "Segment the source transcript before translation. Use render for "
            "edited translations, or set segmenter = 'none' to keep existing cues."
        )
    output = []
    limited = 0
    for segment in segments:
        units = timed_units(segment)
        if units is None:
            limited += 1
            output.append(replace(segment, id=len(output)))
            continue
        start = 0
        while start < len(units):
            last_clause = None
            stop = len(units)
            latest_end = units[start].start
            for index in range(start, len(units)):
                unit = units[index]
                length = display_width(segment.text[units[start].begin : unit.stop].strip())
                duration = max(latest_end, unit.end) - units[start].start
                if index > start and (
                    length > settings.segment_max_chars or duration > settings.segment_max_duration
                ):
                    if units[index].start >= latest_end:
                        stop = last_clause if last_clause is not None else index
                        break
                latest_end = max(latest_end, unit.end)
                next_unit = units[index + 1] if index + 1 < len(units) else None
                boundary = next_unit is None or next_unit.start >= latest_end
                if boundary and next_unit is not None:
                    pause = next_unit.start - latest_end >= settings.segment_pause_seconds
                    sentence = sentence_end(unit.text) and duration >= settings.segment_min_duration
                    if pause or sentence:
                        stop = index + 1
                        break
                    if (
                        re.search(r"[,;:，；：、][\"'”’）)\]]*\s*$", unit.text)
                        and duration >= settings.segment_min_duration
                    ):
                        last_clause = index + 1
            group = units[start:stop]
            output.append(
                Segment(
                    len(output),
                    group[0].start,
                    max(u.end for u in group),
                    segment.text[group[0].begin : group[-1].stop].strip(),
                    words=tuple(u.word for u in group),
                    speaker=segment.speaker,
                )
            )
            start = stop
    if limited and emit:
        emit(
            "warning",
            stage="segment",
            preserved_cues=limited,
            message="Some cues have no usable word timing; preserved their original text and "
            "timing. Word timestamps are required to split these cues accurately.",
        )
    return output

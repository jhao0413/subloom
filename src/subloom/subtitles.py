from __future__ import annotations

import html
import re

from .models import BridgeError, Segment


def timestamp(seconds: float) -> str:
    total = round(seconds * 1000)
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    secs, millis = divmod(total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clean(text: str) -> str:
    return html.escape(re.sub(r"\s+", " ", text).strip(), quote=False)


def srt(segments: list[Segment], style: str) -> str:
    cues = []
    for index, segment in enumerate(segments, 1):
        if style == "original":
            lines = [segment.text]
        else:
            if not segment.translation.strip():
                raise BridgeError(f"Segment {segment.id} has no translation.")
            lines = [segment.translation]
            if style == "bilingual":
                lines.append(segment.text)
            elif style != "translated":
                raise BridgeError(f"Unknown subtitle style: {style}")
        end = max(round(segment.end * 1000), round(segment.start * 1000) + 1) / 1000
        cues.append(
            f"{index}\n{timestamp(segment.start)} --> {timestamp(end)}\n"
            + "\n".join(clean(line) for line in lines)
        )
    return "\n\n".join(cues) + "\n"

import wave

import pytest

from subloom.models import BridgeError
from subloom.providers import transcribe
from subloom.settings import Settings


def audio_file(path, seconds):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * (16000 * seconds))
    return path


@pytest.mark.parametrize(
    "start,end",
    [
        ("NaN", "NaN"),
        (0, float("inf")),
        (0.9, 0.5),
        (0.5, 0.5),
        (2, 3),
        (True, 2),
    ],
)
def test_reject_bad_asr_timestamps_without_dropping_cues(tmp_path, monkeypatch, start, end):
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.setattr(
        "subloom.providers.request",
        lambda *args, **kwargs: {
            "segments": [
                {"start": 0, "end": 0.5, "text": "Valid cue"},
                {"start": start, "end": end, "text": "Invalid cue"},
            ]
        },
    )
    with pytest.raises(BridgeError, match="timestamps"):
        transcribe(audio_file(tmp_path / "audio.wav", 1), Settings(), None, lambda *a, **kw: None)
    assert not (tmp_path / "asr-chunk.wav").exists()


def test_chunk_offsets_and_clipping(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.setattr(
        "subloom.providers.request",
        lambda *args, **kwargs: {
            "segments": [
                {"start": 0, "end": 3, "text": "A sentence"},
            ],
            "words": [
                {"start": 0, "end": 1, "word": "A"},
                {"start": 1, "end": 3, "word": "sentence"},
            ],
        },
    )
    rows = transcribe(
        audio_file(tmp_path / "audio.wav", 12),
        Settings(chunk_seconds=10),
        None,
        lambda *a, **kw: None,
    )
    assert [(s.id, s.start, s.end) for s in rows] == [(0, 0, 3), (1, 10, 12)]
    assert [(w.start, w.end) for w in rows[1].words] == [(10, 11), (11, 12)]

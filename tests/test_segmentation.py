import json
from dataclasses import replace

import pytest

from subloom.cli import main
from subloom.models import BridgeError, Segment, Word, load_segments, save_segments
from subloom.segmentation import display_width, segment_transcript, sentence_end
from subloom.settings import Settings


def cue(tokens, *, text=None, gap=0, duration=1):
    words = tuple(
        Word(i * (duration + gap), i * (duration + gap) + duration, token)
        for i, token in enumerate(tokens)
    )
    return Segment(9, 0, words[-1].end, text or " ".join(tokens), words=words)


def split(segment, **config):
    return segment_transcript([segment], Settings(**config))


def test_sentence_boundaries_preserve_source_punctuation_and_timing():
    segment = cue(["Hello", "world", "Next", "sentence"], text="Hello, world! Next sentence.")
    rows = split(segment)
    assert [(s.id, s.start, s.end, s.text) for s in rows] == [
        (0, 0, 2, "Hello, world!"),
        (1, 2, 4, "Next sentence."),
    ]
    assert tuple(w for s in rows for w in s.words) == segment.words


def test_abbreviation_does_not_end_sentence():
    rows = split(cue(["Dr.", "Smith", "spoke.", "We", "listened."]))
    assert [s.text for s in rows] == ["Dr. Smith spoke.", "We listened."]
    assert not sentence_end("U.S.")


def test_pause_cuts_at_real_speech_boundaries():
    rows = split(cue(["hello", "world"], gap=1))
    assert [(s.start, s.end, s.text) for s in rows] == [(0, 1, "hello"), (2, 3, "world")]


def test_cjk_punctuation_and_width():
    rows = split(cue(["你好", "世界"], text="你好。世界！"))
    assert [s.text for s in rows] == ["你好。", "世界！"]
    assert display_width("你好ab") == 6
    assert display_width("e\u0301") == 1
    rows = split(
        cue(["一二三四五", "六七八九十"], text="一二三四五六七八九十"), segment_max_chars=10
    )
    assert [s.text for s in rows] == ["一二三四五", "六七八九十"]


def test_maximum_duration_and_clause_preference():
    rows = split(cue(["one", "two", "three"]), segment_max_duration=2)
    assert [s.text for s in rows] == ["one two", "three"]
    rows = split(
        cue(["In", "this", "case,", "we", "should", "continue", "carefully."]),
        segment_max_chars=28,
        segment_max_duration=30,
    )
    assert rows[0].text == "In this case,"
    assert all(display_width(s.text) <= 28 for s in rows)


def test_no_word_timing_never_interpolates_timestamps():
    events = []
    source = Segment(8, 0, 30, "A long sentence. Another sentence.")
    rows = segment_transcript([source], Settings(), lambda *a, **kw: events.append((a, kw)))
    assert [(s.start, s.end, s.text) for s in rows] == [(0, 30, source.text)]
    assert events[0][1]["preserved_cues"] == 1


def test_incomplete_word_alignment_preserves_original():
    source = cue(["hello", "world"], text="hello missing world")
    assert split(source)[0].text == source.text
    source = cue(["different"], text="original")
    assert split(source)[0].text == "original"


def test_no_split_in_overlapping_words():
    source = Segment(
        0,
        0,
        2.5,
        "hello world again",
        words=(Word(0, 1.5, "hello"), Word(1, 2, "world"), Word(2, 2.5, "again")),
    )
    rows = split(source, segment_max_duration=1)
    assert [(s.start, s.end) for s in rows] == [(0, 2), (2, 2.5)]


def test_translated_cues_cannot_be_silently_resegmented():
    source = replace(cue(["Hello.", "World."]), translation="Existing edit")
    with pytest.raises(BridgeError, match="before translation"):
        split(source)
    assert split(source, segmenter="none") == [source]


def test_segment_json_roundtrip_and_cli(tmp_path, capsys):
    source = tmp_path / "transcript.json"
    output = tmp_path / "segments.json"
    original = cue(["First.", "Second."])
    save_segments(source, [original])
    assert load_segments(source) == [original]
    assert main(["segment", str(source), "--output", str(output), "--json-events"]) == 0
    assert [s.text for s in load_segments(output)] == ["First.", "Second."]
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["event"] == "result" and result["count"] == 2
    assert main(["segment", str(source), "--output", str(source), "--json-events"]) == 1


@pytest.mark.parametrize(
    "words",
    [
        [{"start": 0, "end": 3, "text": "outside"}],
        [{"start": 0, "end": 0, "text": "empty duration"}],
        [None],
    ],
)
def test_word_validation(words):
    with pytest.raises(BridgeError):
        Segment(0, 0, 2, "hello", words=words)

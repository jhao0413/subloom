import json
import os
import signal
import subprocess
import sys
import time
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from subloom import native
from subloom.models import BridgeError, Segment, Word
from subloom.segmentation import segment_transcript
from subloom.settings import Settings


def whisper_payload():
    return {
        "transcription": [
            {
                "text": " hello wonderful world.",
                "offsets": {"from": 0, "to": 1000},
                "tokens": [
                    {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}},
                    {"text": " hello", "offsets": {"from": 0, "to": 200}},
                    {"text": " wonder", "offsets": {"from": 200, "to": 400}},
                    {"text": "ful", "offsets": {"from": 400, "to": 600}},
                    {"text": " world", "offsets": {"from": 600, "to": 800}},
                    {"text": ".", "offsets": {"from": 800, "to": 800}},
                    {"text": "[_TT_100]", "offsets": {"from": 1000, "to": 1000}},
                ],
            }
        ]
    }


def test_whisper_bpe_words_and_milliseconds():
    rows = native.parse_whisper(whisper_payload())
    assert [(w.start, w.end, w.text) for w in rows[0].words] == [
        (0, 0.2, " hello"),
        (0.2, 0.6, " wonderful"),
        (0.6, 0.8, " world."),
    ]
    assert rows[0].text == "hello wonderful world."
    split = segment_transcript(rows, Settings(segment_max_chars=10))
    assert [s.text for s in split] == ["hello", "wonderful", "world."]


def test_cjk_timed_tokens_and_missing_timing():
    tokens = [
        {"text": "你", "offsets": {"from": 0, "to": 300}},
        {"text": "好。", "offsets": {"from": 300, "to": 600}},
    ]
    assert [w.text for w in native.token_words(tokens, 0, 1)] == ["你", "好。"]
    assert native.token_words([{"text": "untimed"}], 0, 1) == ()
    assert native.token_words([{"text": "bad", "offsets": {"from": 0, "to": 0}}], 0, 1) == ()


@pytest.mark.parametrize("start,end", [(0, 0), (1, 0), (-1, 2), (0, float("nan")), (True, 2)])
def test_native_rejects_bad_intervals(start, end):
    with pytest.raises(BridgeError):
        native.interval(start, end)


def test_moss_timing_speakers_and_no_fake_words(tmp_path):
    audio = tmp_path / "sample with spaces.wav"
    payload = {
        "file": str(audio),
        "text": "hello",
        "segments": [{"t0_ms": 120, "t1_ms": 950, "text": "hello", "speaker_id": 2}],
    }
    rows = native.parse_moss(
        json.dumps({"type": "batch_header"}) + "\n" + json.dumps(payload), audio
    )
    assert rows == [Segment(0, 0.12, 0.95, "hello", speaker=2)]
    assert not rows[0].words
    assert segment_transcript(rows, Settings())[0].speaker == 2
    timed = Segment(
        0, 0, 2, "First. Second.", words=(Word(0, 1, "First."), Word(1, 2, "Second.")), speaker=2
    )
    assert [s.speaker for s in segment_transcript([timed], Settings())] == [2, 2]


@pytest.mark.parametrize(
    "change", [{"error": "OUTPUT_TRUNCATED"}, {"segments": None}, {"file": "/different.wav"}]
)
def test_moss_rejects_partial_or_unmatched_results(tmp_path, change):
    audio = tmp_path / "sample.wav"
    payload = {"file": str(audio), "segments": [{"t0_ms": 0, "t1_ms": 500, "text": "hi"}], **change}
    with pytest.raises(BridgeError):
        native.parse_moss(json.dumps(payload), audio)


def fake_settings(tmp_path, code=""):
    binary = tmp_path / "fake-whisper-cli"
    binary.write_text(f"#!{sys.executable}\n" + code)
    binary.chmod(0o755)
    model = tmp_path / "model.bin"
    model.write_bytes(b"lmggtest-fixture")
    return Settings(
        asr_provider="whisper-cpp", whisper_cpp_cli=str(binary), whisper_cpp_model=str(model)
    )


def audio_file(path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)
    return path


def test_native_invocation_and_stale_output_protection(tmp_path):
    code = (
        "import sys,json\nfrom pathlib import Path\n"
        "prefix=sys.argv[sys.argv.index('-of')+1]\n"
        f"Path(prefix+'.json').write_text(json.dumps({whisper_payload()!r}))\n"
    )
    settings = fake_settings(tmp_path, code)
    audio = audio_file(tmp_path / "audio.wav")
    rows = native.transcribe_native(audio, settings, "en", lambda *a, **kw: None)
    assert rows[0].words[1].text == " wonderful"
    Path(settings.whisper_cpp_cli).write_text(f"#!{sys.executable}\n")
    with pytest.raises(BridgeError, match="did not produce valid JSON"):
        native.transcribe_native(audio, settings, "en", lambda *a, **kw: None)
    assert not (tmp_path / "whisper-cpp.raw.json").exists()


def test_model_or_runtime_change_invalidates_native_fingerprint(tmp_path):
    settings = fake_settings(tmp_path)
    first = native.fingerprint(settings)
    model = Path(settings.whisper_cpp_model)
    model.write_bytes(b"lmggreplacement-fixture")
    assert native.fingerprint(settings) != first
    assert native.status(settings)["ready"]
    model.write_bytes(b"GGUFwrong-model")
    with pytest.raises(BridgeError, match="format"):
        native.fingerprint(settings)
    assert not native.status(replace(settings, whisper_cpp_cli="/nonexistent/tool"))["ready"]


def test_engine_timeout_terminates_process(tmp_path):
    start = time.monotonic()
    with pytest.raises(BridgeError, match="timed out"):
        native.run_engine(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path / "out",
            tmp_path / "err",
            1,
        )
    assert time.monotonic() - start < 4


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_engine_timeout_kills_workers_after_leader_exits(tmp_path):
    worker_pid = tmp_path / "worker.pid"
    group_pid = tmp_path / "group.pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(worker_pid)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"Path({str(group_pid)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    )
    try:
        with pytest.raises(BridgeError, match="timed out"):
            native.run_engine([sys.executable, "-c", parent], tmp_path / "out", tmp_path / "err", 2)
        pid = int(worker_pid.read_text())
        state = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False
        ).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        if group_pid.exists():
            try:
                os.killpg(int(group_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.getenv("SUBLOOM_TEST_NATIVE") != "1", reason="opt-in real local models")
@pytest.mark.parametrize("provider", ["whisper-cpp", "moss"])
def test_real_vimpo_models_through_cli(tmp_path, provider):
    from subloom.cli import main
    from subloom.models import load_segments

    sample = Path(__file__).resolve().parents[1] / ".tools/src/whisper.cpp/samples/jfk.wav"
    settings = Settings(asr_provider=provider)
    if not sample.is_file() or not native.status(settings)["ready"]:
        pytest.skip("build native engines and install the Vimpo models first")
    output = tmp_path / (provider + ".json")
    assert (
        main(
            [
                "transcribe",
                str(sample),
                "--output",
                str(output),
                "--asr",
                provider,
                "--language",
                "en",
                "--json-events",
            ]
        )
        == 0
    )
    rows = load_segments(output)
    assert "country" in " ".join(s.text.lower() for s in rows)
    if provider == "whisper-cpp":
        assert sum(len(s.words) for s in rows) > 15
        assert len(segment_transcript(rows, settings)) > len(rows)
    else:
        assert any(s.speaker is not None for s in rows)
        assert not any(s.words for s in rows)

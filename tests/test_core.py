import json

import httpx
import pytest

from subloom.models import BridgeError, Segment, atomic_text, load_segments
from subloom.pipeline import Job, job_lock, rerender
from subloom.providers import parse_translations, request
from subloom.settings import Settings
from subloom.subtitles import srt, timestamp


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (0, float("nan")), (True, 2)])
def test_invalid_timestamps(start, end):
    with pytest.raises(BridgeError):
        Segment(0, start, end, "hello")


def test_subtitle_timestamp_carry_and_escaping():
    assert timestamp(59.9996) == "00:01:00,000"
    text = srt([Segment(0, 0, 1.5, "<i>Hello</i>\n\nworld", "你好")], "bilingual")
    assert "你好\n&lt;i&gt;Hello&lt;/i&gt; world" in text
    assert "00:00:00,000 --> 00:00:01,500" in text
    with pytest.raises(BridgeError):
        srt([Segment(0, 0, 1, "hello")], "translated")


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"id": 0, "text": ""}],
        [{"id": "0", "text": "a"}],
        [{"id": 0, "text": "a"}, {"id": 0, "text": "b"}],
        [{"id": True, "text": "a"}],
        [{"id": 1, "text": "a"}],
    ],
)
def test_rejects_misaligned_translations(rows):
    with pytest.raises(BridgeError):
        parse_translations(json.dumps({"translations": rows}), [0])


def test_translation_json_fence_and_reordering():
    value = {"translations": [{"id": 9, "text": "nine"}, {"id": 2, "text": "two"}]}
    assert parse_translations("```json\n" + json.dumps(value) + "\n```", [2, 9]) == {
        9: "nine",
        2: "two",
    }


def test_duplicate_import_ids(tmp_path):
    row = {"id": 1, "start": 0, "end": 1, "text": "hello"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([row, row]))
    with pytest.raises(BridgeError, match="unique"):
        load_segments(path)


def test_stage_resume_and_changed_inputs(tmp_path):
    calls = []
    events = []
    output = tmp_path / "result.txt"

    def action():
        calls.append(True)
        atomic_text(output, "data")
        return [output]

    job = Job(tmp_path, lambda event, **data: events.append(event))
    job.stage("test", ["first"], action)
    Job(tmp_path, job.emit).stage("test", ["first"], action)
    assert len(calls) == 1
    assert events[-1] == "cached"
    job.stage("test", ["second"], action)
    assert len(calls) == 2
    output.unlink()
    job.stage("test", ["second"], action)
    assert len(calls) == 3


def test_stage_failure_can_resume(tmp_path):
    job = Job(tmp_path, lambda *args, **kwargs: None)
    with pytest.raises(BridgeError):
        job.stage("test", [], lambda: (_ for _ in ()).throw(BridgeError("failure")))
    assert json.loads((tmp_path / "job.json").read_text())["stages"]["test"]["status"] == "failed"
    output = tmp_path / "result"
    atomic_text(output, "ok")
    job.stage("test", [], lambda: [output])
    assert job.state["stages"]["test"]["status"] == "completed"


def test_job_lock(tmp_path):
    with job_lock(tmp_path):
        with pytest.raises(BridgeError, match="locked"):
            with job_lock(tmp_path):
                pass
    assert not (tmp_path / ".lock").exists()


def test_multipart_retry_rewinds_stream(monkeypatch, tmp_path):
    monkeypatch.setattr("subloom.providers.time.sleep", lambda _: None)
    bodies = []

    def handle(request):
        bodies.append(request.read())
        return httpx.Response(503 if len(bodies) == 1 else 200, json={"ok": True})

    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF-test-audio")
    with httpx.Client(transport=httpx.MockTransport(handle)) as client, path.open("rb") as audio:
        assert request(
            client,
            "https://test.invalid/v1/audio/transcriptions",
            files={"file": ("audio.wav", audio, "audio/wav")},
        ) == {"ok": True}
    assert all(b"RIFF-test-audio" in body for body in bodies)


@pytest.mark.parametrize("output_name", ["source.mp4", "segments.json", "output.txt"])
def test_rerender_rejects_input_overwrite_and_wrong_extension(tmp_path, output_name):
    with pytest.raises(BridgeError):
        rerender(
            tmp_path / "source.mp4",
            tmp_path / "segments.json",
            tmp_path / output_name,
            settings=Settings(),
            mode="soft",
            style="bilingual",
            target="zh-CN",
        )


def test_api_errors_do_not_expose_response_secrets():
    def handle(request):
        return httpx.Response(401, text="secret-key-and-private-data")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(BridgeError, match="HTTP 401") as error:
            request(client, "https://test.invalid")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_seconds": 601},
        {"batch_size": 0},
        {"font": 42},
        {"asr_key_env": ""},
        {"asr_provider": "unknown"},
        {"translation_base_url": "https://user:password@example.com/v1"},
    ],
)
def test_settings_validation(kwargs):
    with pytest.raises(BridgeError):
        Settings(**kwargs)

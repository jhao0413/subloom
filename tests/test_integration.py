import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from subloom.media import command


@pytest.fixture
def service(tmp_path):
    for executable in ("ffmpeg", "ffprobe", "yt-dlp"):
        if not shutil.which(executable):
            pytest.skip(f"requires {executable}")
    video = tmp_path / "input 'with spaces'.mp4"
    command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=320x180:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    calls = {"asr": 0, "translation": 0, "download": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(video.stat().st_size))
            self.end_headers()

        def do_GET(self):
            calls["download"] += 1
            self.do_HEAD()
            self.wfile.write(video.read_bytes())

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path.endswith("/audio/transcriptions"):
                calls["asr"] += 1
                assert b"RIFF" in body
                payload = {
                    "segments": [
                        {"start": 0, "end": 0.9, "text": "Hello, world."},
                        {"start": 1, "end": 1.9, "text": "A second sentence."},
                    ],
                    "words": [
                        {"start": 0, "end": 0.4, "word": "Hello"},
                        {"start": 0.5, "end": 0.9, "word": "world"},
                        {"start": 1, "end": 1.2, "word": "A"},
                        {"start": 1.2, "end": 1.5, "word": "second"},
                        {"start": 1.5, "end": 1.9, "word": "sentence"},
                    ],
                }
            else:
                calls["translation"] += 1
                request = json.loads(body)
                prompt = json.loads(request["messages"][1]["content"])
                rows = [
                    {"id": s["id"], "text": "你好，世界。" if s["id"] == 0 else "第二句话。"}
                    for s in reversed(prompt["segments"])
                ]
                payload = {
                    "choices": [{"message": {"content": json.dumps({"translations": rows})}}]
                }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    config = tmp_path / "config.toml"
    config.write_text(f'asr_base_url = "{base}/v1"\ntranslation_base_url = "{base}/v1"\n')
    try:
        yield video, calls, config, base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def cli(*args):
    result = subprocess.run(
        [sys.executable, "-m", "subloom", *map(str, args)],
        capture_output=True,
        text=True,
        input="q\n",
        check=False,
        env={
            **os.environ,
            "OPENAI_API_KEY": "local-test-key",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        },
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [json.loads(line) for line in result.stdout.splitlines()]


@pytest.mark.parametrize("remote", [False, True])
def test_complete_cli_pipeline_and_resume(service, tmp_path, remote):
    video, calls, config, base = service
    source = base + "/source.mp4" if remote else str(video)
    args = ["run", source, "--workspace", tmp_path / "jobs", "--config", config, "--json-events"]
    events = cli(*args)
    result = events[-1]
    output = Path(result["video"])
    assert output.is_file()
    assert calls["asr"] == calls["translation"] == 1
    if remote:
        assert calls["download"] > 0
    info = json.loads(
        command(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)])
    )
    assert {stream["codec_type"] for stream in info["streams"]} == {"video", "audio", "subtitle"}
    for stream in info["streams"]:
        if stream["codec_type"] in {"video", "audio"}:
            assert float(stream["duration"]) == pytest.approx(2, abs=0.15)
    extracted = command(
        ["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:s:0", "-f", "srt", "-"]
    )
    assert "你好，世界。" in extracted and "Hello, world." in extracted
    events = cli(*args)
    assert len([event for event in events if event["event"] == "cached"]) == 7
    assert calls["asr"] == calls["translation"] == 1
    folder = Path(result["job"])
    timed = json.loads((folder / "transcript.json").read_text())["segments"]
    assert timed[0]["words"][0]["text"] == "Hello"
    assert (folder / "segments.json").exists()
    manifest = (folder / "job.json").read_text()
    assert "local-test-key" not in manifest
    translated = folder / "translated.json"
    data = json.loads(translated.read_text())
    data["segments"][0]["translation"] = "人工修改。"
    translated.write_text(json.dumps(data, ensure_ascii=False))
    cli(*args)
    assert "人工修改。" in (folder / "captions.srt").read_text()
    assert calls["asr"] == calls["translation"] == 1
    edited = tmp_path / "edited video.mp4"
    cli("render", "--video", video, "--segments", translated, "--output", edited, "--json-events")
    assert edited.is_file()
    assert calls["asr"] == calls["translation"] == 1


def test_resegmentation_reuses_asr_and_retranslates_changed_cues(service, tmp_path):
    video, calls, config, _ = service
    args = ["run", video, "--workspace", tmp_path / "jobs", "--config", config, "--json-events"]
    first = cli(*args)[-1]
    assert calls["asr"] == calls["translation"] == 1
    with config.open("a") as file:
        file.write("segment_max_duration = 0.8\nsegment_min_duration = 0.1\n")
    events = cli(*args)
    assert calls["asr"] == 1
    assert calls["translation"] == 2
    assert any(e.get("stage") == "transcribe" and e["event"] == "cached" for e in events)
    rows = json.loads((Path(first["job"]) / "segments.json").read_text())["segments"]
    assert len(rows) == 4
    assert all(s["end"] - s["start"] <= 0.8 for s in rows)


def test_force_refreshes_completed_url_download(service, tmp_path):
    video, _, config, base = service
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps({"segments": [{"id": 0, "start": 0, "end": 1, "text": "Sample"}]})
    )
    args = [
        "run",
        base + "/source.mp4",
        "--workspace",
        tmp_path / "jobs",
        "--config",
        config,
        "--transcript",
        transcript,
        "--style",
        "original",
        "--json-events",
    ]
    first = cli(*args)[-1]
    command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=s=320x180:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=16000",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    second = cli(*args, "--force")[-1]
    assert second["video"] == first["video"]
    info = json.loads(
        command(["ffprobe", "-v", "error", "-show_streams", "-of", "json", second["video"]])
    )
    assert float(next(s for s in info["streams"] if s["codec_type"] == "video")["duration"]) == 3


def test_piped_quit_does_not_truncate_audio_or_video(service, tmp_path):
    video, _, config, _ = service
    command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=s=64x64:r=1",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=16000",
            "-t",
            "120",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    result = cli(
        "run", video, "--workspace", tmp_path / "jobs", "--config", config, "--json-events"
    )[-1]
    for path in (Path(result["job"]) / "audio.wav", Path(result["video"])):
        info = json.loads(
            command(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
        )
        for stream in info["streams"]:
            if stream["codec_type"] in {"audio", "video"}:
                assert float(stream["duration"]) == pytest.approx(120, abs=0.2)


def test_import_original_requires_no_model(service, tmp_path):
    video, calls, _, _ = service
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps({"segments": [{"id": 0, "start": 0, "end": 1.5, "text": "Original text"}]})
    )
    events = cli(
        "run",
        video,
        "--workspace",
        tmp_path / "jobs",
        "--transcript",
        transcript,
        "--style",
        "original",
        "--json-events",
    )
    assert Path(events[-1]["video"]).is_file()
    assert calls["asr"] == calls["translation"] == 0

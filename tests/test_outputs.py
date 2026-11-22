import json
import shutil

import pytest

from subloom.cli import main
from subloom.media import command
from subloom.models import BridgeError, Segment, load_segments, prepare_output, save_segments


@pytest.mark.parametrize("operation", ["transcribe", "segment", "render"])
def test_standalone_commands_preserve_existing_outputs(tmp_path, operation):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"input-video")
    transcript = tmp_path / "transcript.json"
    save_segments(transcript, [Segment(0, 0, 1, "Hello", "你好")])
    output = tmp_path / ("valuable.mp4" if operation == "render" else "valuable.json")
    output.write_bytes(b"original-content")
    if operation == "transcribe":
        args = [operation, str(video)]
    elif operation == "segment":
        args = [operation, str(transcript)]
    else:
        args = [operation, "--video", str(video), "--segments", str(transcript)]
    assert main([*args, "--output", str(output), "--json-events"]) == 1
    assert output.read_bytes() == b"original-content"
    assert video.read_bytes() == b"input-video"


def test_segment_requires_json_even_with_overwrite(tmp_path):
    source = tmp_path / "source.json"
    save_segments(source, [Segment(0, 0, 1, "Hello")])
    output = tmp_path / "valuable.mp4"
    output.write_bytes(b"original-video")
    assert main(["segment", str(source), "--output", str(output), "--overwrite"]) == 1
    assert output.read_bytes() == b"original-video"


def test_segment_overwrite_is_explicit_and_never_replaces_input(tmp_path):
    source, output = tmp_path / "source.json", tmp_path / "segments.json"
    save_segments(source, [Segment(0, 0, 1, "Hello")])
    output.write_text("existing")
    args = ["segment", str(source), "--output", str(output)]
    assert main(args) == 1
    assert main([*args, "--overwrite"]) == 0
    assert load_segments(output)[0].text == "Hello"
    original = source.read_bytes()
    assert main(["segment", str(source), "--output", str(source), "--overwrite"]) == 1
    assert source.read_bytes() == original


def test_atomic_output_rejects_file_created_after_preflight(tmp_path):
    output = prepare_output(tmp_path / "result.json", ".json")
    output.write_bytes(b"created-by-another-process")
    with pytest.raises(BridgeError, match="already exists"):
        save_segments(output, [Segment(0, 0, 1, "Hello")], overwrite=False)
    assert output.read_bytes() == b"created-by-another-process"
    assert list(tmp_path.iterdir()) == [output]
    output.unlink()
    save_segments(output, [Segment(0, 0, 1, "Hello")], overwrite=False)
    assert load_segments(output)[0].text == "Hello"


def test_render_overwrites_only_with_explicit_flag(tmp_path):
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        pytest.skip("requires FFmpeg")
    video, segments, output = (
        tmp_path / "source.mp4",
        tmp_path / "segments.json",
        tmp_path / "edited.mp4",
    )
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
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ]
    )
    save_segments(segments, [Segment(0, 0, 0.8, "Hello", "你好")])
    output.write_bytes(b"valuable-original")
    args = ["render", "--video", str(video), "--segments", str(segments), "--output", str(output)]
    assert main(args) == 1
    assert output.read_bytes() == b"valuable-original"
    assert main([*args, "--overwrite"]) == 0
    streams = json.loads(
        command(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)])
    )
    assert {s["codec_type"] for s in streams["streams"]} == {"video", "audio", "subtitle"}
    assert not list(tmp_path.glob("*.partial.mp4"))

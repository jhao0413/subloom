from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from .models import BridgeError, publish_output
from .settings import Settings


def command(args: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise BridgeError(f"Missing executable: {args[0]}. Run subloom doctor.") from exc
    except subprocess.TimeoutExpired as exc:
        raise BridgeError(f"Timed out running {args[0]}.") from exc
    if result.returncode:
        detail = result.stderr[-3000:]
        raise BridgeError(f"{Path(args[0]).name} failed (exit {result.returncode}): {detail}")
    return result.stdout


def probe(path: Path, settings: Settings) -> dict:
    return json.loads(
        command(
            [
                settings.ffprobe,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def duration(path: Path, settings: Settings) -> float:
    try:
        value = float(probe(path, settings)["format"]["duration"])
        if value <= 0:
            raise ValueError("duration is zero")
        return value
    except (KeyError, ValueError) as exc:
        raise BridgeError(f"Cannot determine duration of {path}.") from exc


def is_url(source: str) -> bool:
    return urlsplit(source).scheme in {"http", "https"}


def acquire(
    source: str,
    folder: Path,
    settings: Settings,
    cookies_browser: str | None,
    *,
    force: bool = False,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if is_url(source):
        args = [
            settings.yt_dlp,
            "--no-playlist",
            "--no-progress",
            "--continue",
            "--ffmpeg-location",
            str(Path(shutil.which(settings.ffmpeg) or settings.ffmpeg).parent),
            "-f",
            "bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "--remux-video",
            "mp4",
            "-o",
            str(folder / "source.%(ext)s"),
            "--print",
            "after_move:filepath",
        ]
        if force:
            args.append("--force-overwrites")
        if cookies_browser:
            args += ["--cookies-from-browser", cookies_browser]
        output = command([*args, "--", source])
        candidates = [Path(line) for line in output.splitlines() if Path(line).is_file()]
        if not candidates:
            raise BridgeError("yt-dlp did not return the downloaded video path.")
        result = candidates[-1].resolve()
    else:
        original = Path(source).expanduser().resolve()
        if not original.is_file():
            raise BridgeError(f"Video does not exist: {original}")
        result = folder / ("source" + original.suffix.lower())
        temporary = result.with_name(result.name + ".tmp")
        shutil.copyfile(original, temporary)
        os.replace(temporary, result)
    info = probe(result, settings)
    if not any(s["codec_type"] == "video" for s in info["streams"]):
        raise BridgeError("Input must contain a video stream.")
    if not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise BridgeError("Input has no audio stream to transcribe.")
    return result


def extract_audio(video: Path, audio: Path, settings: Settings) -> None:
    temporary = audio.with_name("audio.partial.wav")
    command(
        [
            settings.ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
    )
    os.replace(temporary, audio)


def has_libass(settings: Settings) -> bool:
    filters = command([settings.ffmpeg, "-hide_banner", "-filters"], timeout=15)
    return any(
        len(parts := line.split()) > 1 and parts[1] == "subtitles" for line in filters.splitlines()
    )


def render(
    video: Path,
    subtitle: Path,
    output: Path,
    settings: Settings,
    mode: str,
    target: str,
    *,
    overwrite: bool = True,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [settings.ffmpeg, "-y", "-v", "error", "-i", str(video)]
    if mode == "soft":
        args += [
            "-i",
            str(subtitle),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map",
            "1:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"title={target} subtitles",
            "-disposition:s:0",
            "default",
        ]
    elif mode == "hard":
        if not has_libass(settings):
            raise BridgeError(
                "Hard subtitles need FFmpeg with libass. Use --mode soft, or configure "
                "an ffmpeg-full executable in config.local.toml."
            )
        font = settings.font.replace("'", "").replace(",", "").replace(":", "")
        style = f"FontName={font},FontSize={settings.font_size},Outline=1,MarginV=24"
        args += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"subtitles=captions.srt:force_style='{style}'",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
        ]
    else:
        raise BridgeError(f"Unsupported render mode: {mode}")
    fd, name = tempfile.mkstemp(
        prefix=f".{output.stem[:80]}.", suffix=".partial.mp4", dir=output.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        command([*args, "-movflags", "+faststart", str(temporary)], cwd=subtitle.parent)
        publish_output(temporary, output, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)

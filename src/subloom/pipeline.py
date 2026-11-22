from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import media, native, providers
from .models import (
    BridgeError,
    atomic_json,
    atomic_text,
    load_segments,
    prepare_output,
    save_segments,
)
from .segmentation import segment_transcript
from .settings import Settings
from .subtitles import srt

EventSink = Callable[..., None]


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def job_lock(folder: Path) -> Iterator[None]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ".lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BridgeError(
            f"Job is locked: {path}. If no process is running, remove this lock."
        ) from exc
    try:
        with os.fdopen(fd, "w") as file:
            file.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


class Job:
    def __init__(self, folder: Path, emit: EventSink, force: bool = False):
        self.folder = folder
        self.emit = emit
        self.force = force
        self.state_path = folder / "job.json"
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text())
                if self.state.get("schema_version") != 1 or not isinstance(
                    self.state.get("stages"), dict
                ):
                    raise ValueError("unsupported job schema")
            except (ValueError, OSError, AttributeError) as exc:
                raise BridgeError(f"Cannot read job manifest: {self.state_path}") from exc
        else:
            self.state = {"schema_version": 1, "stages": {}}

    def stage(self, name: str, inputs: Any, action: Callable[[], list[Path]]) -> list[Path]:
        key = signature({"version": 1, "inputs": inputs})
        previous = self.state["stages"].get(name, {})
        files = [self.folder / p for p in previous.get("outputs", [])]
        if (
            not self.force
            and previous.get("signature") == key
            and previous.get("status") == "completed"
            and files
            and all(p.is_file() and p.stat().st_size for p in files)
        ):
            self.emit("cached", stage=name)
            return files
        self.state["stages"][name] = {"status": "running", "signature": key}
        atomic_json(self.state_path, self.state)
        self.emit("started", stage=name)
        try:
            files = action()
            if not files or not all(p.is_file() and p.stat().st_size for p in files):
                raise BridgeError(f"Stage {name} did not create all expected outputs.")
        except BaseException:
            self.state["stages"][name]["status"] = "failed"
            atomic_json(self.state_path, self.state)
            self.emit("failed", stage=name)
            raise
        self.state["stages"][name].update(
            status="completed", outputs=[str(p.relative_to(self.folder)) for p in files]
        )
        atomic_json(self.state_path, self.state)
        self.emit("completed", stage=name)
        return files


def run(
    source: str,
    *,
    workspace: Path,
    settings: Settings,
    emit: EventSink,
    target: str = "zh-CN",
    language: str | None = None,
    mode: str = "soft",
    style: str = "bilingual",
    transcript: Path | None = None,
    cookies_browser: str | None = None,
    force: bool = False,
) -> Path:
    local = None if media.is_url(source) else Path(source).expanduser().resolve()
    if local and not local.is_file():
        raise BridgeError(f"Video does not exist: {local}")
    if transcript:
        transcript = transcript.expanduser().resolve()
        load_segments(transcript)
    if mode == "hard" and not media.has_libass(settings):
        raise BridgeError("Hard subtitles need FFmpeg with libass. Run doctor or use --mode soft.")
    native_identity = (
        native.fingerprint(settings)
        if not transcript and settings.asr_provider in native.NATIVE_PROVIDERS
        else None
    )
    identity = str(local) if local else source
    folder = workspace.expanduser().resolve() / signature([identity, target])[:16]
    with job_lock(folder):
        job = Job(folder, emit, force)
        emit("job", path=str(folder))
        job.state.update(source=identity, target_language=target)
        source_stamp = digest(local) if local else identity
        video = job.stage(
            "download",
            [source_stamp],
            lambda: [media.acquire(identity, folder, settings, cookies_browser, force=force)],
        )[0]
        audio = folder / "audio.wav"

        def audio_action() -> list[Path]:
            media.extract_audio(video, audio, settings)
            return [audio]

        transcribed = folder / "transcript.json"
        if transcript:

            def import_action() -> list[Path]:
                save_segments(transcribed, load_segments(transcript), language=language)
                return [transcribed]

            job.stage("transcribe", ["import", digest(transcript)], import_action)
        else:
            job.stage("audio", [digest(video), settings.ffmpeg], audio_action)

            def transcribe_action() -> list[Path]:
                rows = providers.transcribe(audio, settings, language, emit)
                save_segments(
                    transcribed,
                    rows,
                    language=language,
                    provider=settings.asr_provider,
                    native=native_identity,
                )
                return [transcribed]

            job.stage(
                "transcribe",
                [
                    "word-timestamps-v1",
                    digest(audio),
                    language,
                    settings.asr_provider,
                    settings.asr_base_url,
                    settings.asr_model,
                    settings.local_model,
                    settings.local_device,
                    settings.local_compute_type,
                    settings.chunk_seconds,
                    native_identity,
                ],
                transcribe_action,
            )
        segmented = folder / "segments.json"

        def segment_action() -> list[Path]:
            rows = segment_transcript(load_segments(transcribed), settings, emit)
            save_segments(segmented, rows, language=language, segmenter=settings.segmenter)
            return [segmented]

        job.stage(
            "segment",
            [
                digest(transcribed),
                settings.segmenter,
                settings.segment_max_chars,
                settings.segment_max_duration,
                settings.segment_min_duration,
                settings.segment_pause_seconds,
            ],
            segment_action,
        )
        translated = folder / "translated.json"

        def translation_action() -> list[Path]:
            rows = providers.translate(load_segments(segmented), target, settings, emit)
            save_segments(translated, rows, target_language=target)
            return [translated]

        if style != "original":
            job.stage(
                "translate",
                [
                    digest(segmented),
                    target,
                    settings.translation_base_url,
                    settings.translation_model,
                    settings.batch_size,
                    settings.context_segments,
                ],
                translation_action,
            )
        segments = translated if style != "original" else segmented

        def subtitle_action() -> list[Path]:
            rows = load_segments(segments)
            paths = []
            for filename, selected in (("original.srt", "original"), ("captions.srt", style)):
                path = folder / filename
                atomic_text(path, srt(rows, selected))
                paths.append(path)
            if style != "original":
                path = folder / "translated.srt"
                atomic_text(path, srt(rows, "translated"))
                paths.append(path)
            return paths

        captions = folder / "captions.srt"
        job.stage("subtitles", [digest(segments), style], subtitle_action)
        output = folder / "output.mp4"

        def render_action() -> list[Path]:
            media.render(video, captions, output, settings, mode, target)
            return [output]

        job.stage(
            "render",
            [
                digest(video),
                digest(captions),
                mode,
                settings.ffmpeg,
                settings.font,
                settings.font_size,
            ],
            render_action,
        )
        emit("result", video=str(output), job=str(folder), subtitles=str(captions))
        return output


def rerender(
    video: Path,
    segments: Path,
    output: Path,
    *,
    settings: Settings,
    mode: str,
    style: str,
    target: str,
    overwrite: bool = False,
) -> None:
    import tempfile

    video, output = video.expanduser().resolve(), output.expanduser().resolve()
    segments = segments.expanduser().resolve()
    output = prepare_output(output, ".mp4", (video, segments), overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix="subloom-render-") as name:
        subtitle = Path(name) / "captions.srt"
        atomic_text(subtitle, srt(load_segments(segments), style))
        media.render(video, subtitle, output, settings, mode, target, overwrite=overwrite)

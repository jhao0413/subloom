from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__, media, native, providers
from .models import BridgeError, load_segments, prepare_output, save_segments
from .pipeline import job_lock, rerender, run
from .segmentation import segment_transcript
from .settings import Settings


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="subloom", description="Download, transcribe, translate and subtitle videos."
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    process = commands.add_parser("run", help="Run or resume the video pipeline")
    process.add_argument("source", help="Video file or URL supported by yt-dlp")
    process.add_argument("--workspace", type=Path, default=Path("jobs"))
    process.add_argument("--target", default="zh-CN", help="Translation language (default: zh-CN)")
    process.add_argument(
        "--language", help="ASR source language code, e.g. en; default: auto-detect"
    )
    process.add_argument("--transcript", type=Path, help="Import timestamped JSON instead of ASR")
    process.add_argument(
        "--cookies-from-browser", metavar="BROWSER", help="Pass to yt-dlp, e.g. chrome"
    )
    process.add_argument("--force", action="store_true", help="Rerun completed stages")
    render = commands.add_parser("render", help="Render edited segment JSON without model calls")
    render.add_argument("--video", type=Path, required=True)
    render.add_argument("--segments", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--target", default="zh-CN")
    for command in (process, render):
        command.add_argument("--mode", choices=("soft", "hard"), default="soft")
        command.add_argument(
            "--style", choices=("bilingual", "translated", "original"), default="bilingual"
        )
        command.add_argument("--json-events", action="store_true", help="Emit JSON Lines to stdout")
    transcribe = commands.add_parser(
        "transcribe", help="Transcribe local audio/video to timed JSON"
    )
    transcribe.add_argument("source", type=Path)
    transcribe.add_argument("--output", type=Path, required=True)
    transcribe.add_argument("--language")
    transcribe.add_argument("--json-events", action="store_true")
    segment = commands.add_parser("segment", help="Resegment word-timed transcript JSON")
    segment.add_argument("transcript", type=Path)
    segment.add_argument("--output", type=Path, required=True)
    segment.add_argument("--json-events", action="store_true")
    for command in (render, transcribe, segment):
        command.add_argument(
            "--overwrite", action="store_true", help="Replace an existing output file"
        )
    doctor = commands.add_parser("doctor", help="Check tools, subtitle support and provider setup")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable report")
    for command in (process, transcribe, doctor):
        command.add_argument(
            "--asr",
            choices=("whisper-cpp", "moss", "local", "openai"),
            help="Override transcription provider",
        )
    for command in (process, render, segment, transcribe, doctor):
        command.add_argument("--config", type=Path, help="TOML configuration file")
    return root


def doctor(settings: Settings) -> dict:
    tools = {
        name: shutil.which(getattr(settings, name)) for name in ("ffmpeg", "ffprobe", "yt_dlp")
    }
    libass = media.has_libass(settings) if tools["ffmpeg"] else False
    providers = {
        "asr": {
            "provider": settings.asr_provider,
            "model": settings.local_model
            if settings.asr_provider == "local"
            else settings.asr_model,
            "ready": importlib.util.find_spec("faster_whisper") is not None
            if settings.asr_provider == "local"
            else bool(os.environ.get(settings.asr_key_env)),
        },
        "translation": {
            "model": settings.translation_model,
            "ready": bool(os.environ.get(settings.translation_key_env)),
        },
    }
    if settings.asr_provider in native.NATIVE_PROVIDERS:
        providers["asr"] = native.status(settings)
    return {
        "tools": tools,
        "soft_subtitles": bool(tools["ffmpeg"]),
        "hard_subtitles": libass,
        "providers": providers,
        "ready": all(tools.values()) and all(value["ready"] for value in providers.values()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    json_events = getattr(args, "json_events", False)

    def emit(event: str, **data) -> None:
        if json_events:
            print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)
        elif event == "result":
            print(data.get("video", data.get("segments", data.get("transcript", ""))), flush=True)
        else:
            details = " ".join(f"{key}={value}" for key, value in data.items())
            print(f"[{event}] {details}", file=sys.stderr, flush=True)

    try:
        settings = Settings.load(args.config)
        if getattr(args, "asr", None):
            settings = replace(settings, asr_provider=args.asr)
        if args.command == "doctor":
            report = doctor(settings)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                for name, location in report["tools"].items():
                    print(f"{name}: {location or 'MISSING'}")
                print(f"Hard subtitles (libass): {report['hard_subtitles']}")
                for name, info in report["providers"].items():
                    print(
                        f"{name}: {info['model']} — {'ready' if info['ready'] else 'not configured'}"
                    )
                if not report["hard_subtitles"]:
                    print("Use --mode soft, or configure FFmpeg with libass for --mode hard.")
            return 0 if report["ready"] else 1
        if args.command == "transcribe":
            source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
            if not source.is_file():
                raise BridgeError(f"Input does not exist: {source}")
            output = prepare_output(output, ".json", (source,), overwrite=args.overwrite)
            work = output.with_suffix(".work")
            if source.is_relative_to(work):
                raise BridgeError("Input cannot be inside the transcript work directory.")
            identity = (
                native.fingerprint(settings)
                if settings.asr_provider in native.NATIVE_PROVIDERS
                else None
            )
            with job_lock(work):
                audio = work / "audio.wav"
                media.extract_audio(source, audio, settings)
                rows = providers.transcribe(audio, settings, args.language, emit)
                save_segments(
                    output,
                    rows,
                    language=args.language,
                    provider=settings.asr_provider,
                    native=identity,
                    overwrite=args.overwrite,
                )
            emit("result", transcript=str(output), count=len(rows))
        elif args.command == "segment":
            source = args.transcript.expanduser().resolve()
            output = prepare_output(args.output, ".json", (source,), overwrite=args.overwrite)
            rows = segment_transcript(load_segments(source), settings, emit)
            save_segments(output, rows, segmenter=settings.segmenter, overwrite=args.overwrite)
            emit("result", segments=str(output), count=len(rows))
        elif args.command == "run":
            run(
                args.source,
                workspace=args.workspace,
                settings=settings,
                emit=emit,
                target=args.target,
                language=args.language,
                mode=args.mode,
                style=args.style,
                transcript=args.transcript,
                cookies_browser=args.cookies_from_browser,
                force=args.force,
            )
        else:
            rerender(
                args.video,
                args.segments,
                args.output,
                settings=settings,
                mode=args.mode,
                style=args.style,
                target=args.target,
                overwrite=args.overwrite,
            )
            emit("result", video=str(args.output.expanduser().resolve()))
        return 0
    except KeyboardInterrupt:
        emit("error", message="Interrupted. Rerun the same command to resume completed stages.")
        return 130
    except (BridgeError, OSError, ValueError) as exc:
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path

ENGINES = {
    "whisper-cpp": {
        "repo": "https://github.com/ggml-org/whisper.cpp.git",
        "tag": "v1.9.5",
        "commit": "d1be6fde11ac6e0407606b4e42fe72d34add8037",
        "source": "whisper.cpp",
        "target": "whisper-cli",
    },
    "moss": {
        "repo": "https://github.com/handy-computer/transcribe.cpp.git",
        "tag": "v0.4.2",
        "commit": "275cd50d2c4104c204009a197ac091b9687668e5",
        "source": "transcribe.cpp",
        "target": "transcribe-cli",
    },
}


def run(args, log):
    result = subprocess.run(
        args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log.name}")


def main():
    parser = argparse.ArgumentParser(description="Build pinned native transcription engines.")
    parser.add_argument("--engine", choices=("all", "whisper-cpp", "moss"), default="all")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cmake, ninja, git = (shutil.which(name) for name in ("cmake", "ninja", "git"))
    if not all((cmake, ninja, git)):
        parser.error("Need git, cmake and ninja. Run with uv run --extra native-build.")
    root = Path(__file__).resolve().parents[1] / ".tools"
    root.mkdir(exist_ok=True)
    manifest = root / "engines.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    for name, spec in ENGINES.items():
        if args.engine not in {"all", name}:
            continue
        source, build = root / "src" / spec["source"], root / "build" / spec["source"]
        log_path = root / f"{name}.build.log"
        with log_path.open("w") as log:
            if not source.exists():
                source.parent.mkdir(exist_ok=True)
                print(f"[{name}] Downloading source {spec['tag']}", flush=True)
                run(
                    [
                        git,
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        spec["tag"],
                        spec["repo"],
                        str(source),
                    ],
                    log,
                )
            commit = subprocess.check_output(
                [git, "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if commit != spec["commit"]:
                raise RuntimeError(
                    f"Unexpected source commit in {source}: {commit}; expected {spec['commit']}"
                )
            metal = "ON" if platform.system() == "Darwin" else "OFF"
            flags = [f"-DGGML_METAL={metal}", "-DGGML_METAL_EMBED_LIBRARY=ON"]
            flags += (
                ["-DBUILD_SHARED_LIBS=OFF", "-DWHISPER_BUILD_TESTS=OFF"]
                if name == "whisper-cpp"
                else [
                    f"-DTRANSCRIBE_METAL={metal}",
                    "-DTRANSCRIBE_BUILD_SHARED=OFF",
                    "-DTRANSCRIBE_BUILD_EXAMPLES=ON",
                    "-DTRANSCRIBE_BUILD_TESTS=OFF",
                ]
            )
            print(f"[{name}] Configuring and building; log: {log_path}", flush=True)
            run(
                [
                    cmake,
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-G",
                    "Ninja",
                    f"-DCMAKE_MAKE_PROGRAM={ninja}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    *flags,
                ],
                log,
            )
            run(
                [cmake, "--build", str(build), "--target", spec["target"], "-j", str(args.jobs)],
                log,
            )
        binary = build / "bin" / spec["target"]
        if not binary.is_file():
            raise RuntimeError(f"Build produced no executable: {binary}")
        records[name] = {**spec, "executable": str(binary)}
        manifest.write_text(json.dumps(records, indent=2) + "\n")
        print(f"[{name}] Ready: {binary}", flush=True)
    print("Model files are read from config; no model downloads were performed.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from exc

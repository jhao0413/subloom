from __future__ import annotations

import ast
import io
import os
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path

from pygments.lexers import get_lexer_for_filename
from pygments.token import Comment

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    ".tools",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "jobs",
}
CONFIG_SUFFIXES = {".toml", ".yaml", ".yml", ".sh"}


def diagnostic(path: Path, line: int, column: int, code: str, message: str) -> str:
    return f"{path}:{line}:{column}: {code} {message}"


def check_python(path: Path, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=str(path))
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (SyntaxError, tokenize.TokenError) as exc:
        return [
            diagnostic(
                path,
                getattr(exc, "lineno", 1) or 1,
                getattr(exc, "offset", 1) or 1,
                "LNT001",
                "Invalid Python syntax",
            )
        ]
    issues = [
        diagnostic(path, token.start[0], token.start[1] + 1, "NOC001", "Comments are not allowed")
        for token in tokens
        if token.type == tokenize.COMMENT
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and (
            isinstance(node.value, ast.JoinedStr)
            or (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        ):
            issues.append(
                diagnostic(
                    path,
                    node.lineno,
                    node.col_offset + 1,
                    "NOC002",
                    "Docstrings and standalone string comments are not allowed",
                )
            )
    return issues


def check_hash_comments(path: Path, source: str) -> list[str]:
    issues = []
    lexer = get_lexer_for_filename(path.name)
    for index, token_type, _ in lexer.get_tokens_unprocessed(source):
        if token_type in Comment:
            line = source.count("\n", 0, index) + 1
            column = index - source.rfind("\n", 0, index)
            issues.append(diagnostic(path, line, column, "NOC001", "Comments are not allowed"))
    return issues


def source_files(root: Path):
    for folder, directories, filenames in os.walk(root):
        directories[:] = sorted(
            name
            for name in directories
            if name not in EXCLUDED and not (Path(folder) / name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(folder) / name
            if not path.is_symlink() and (
                path.suffix == ".py" or path.suffix in CONFIG_SUFFIXES or path.name == "Makefile"
            ):
                yield path


def check_repository(root: Path) -> list[str]:
    issues = []
    for path in source_files(root):
        relative = path.relative_to(root)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            issues.append(diagnostic(relative, 1, 1, "LNT001", "Cannot read source as UTF-8"))
            continue
        if path.suffix == ".py":
            issues.extend(check_python(relative, source))
        else:
            if path.suffix == ".toml":
                try:
                    tomllib.loads(source)
                except tomllib.TOMLDecodeError as exc:
                    issues.append(
                        diagnostic(
                            relative,
                            getattr(exc, "lineno", 1),
                            getattr(exc, "colno", 1),
                            "LNT001",
                            "Invalid TOML syntax",
                        )
                    )
            issues.extend(check_hash_comments(relative, source))
    return issues


def main() -> int:
    issues = check_repository(ROOT)
    for issue in issues:
        print(issue, flush=True)
    if issues:
        print(f"No-comments check failed: {len(issues)} issue(s).", flush=True)
    codes = [
        subprocess.run([sys.executable, "-m", "ruff", *args], cwd=ROOT, check=False).returncode
        for args in (("check", "."), ("format", "--check", "."))
    ]
    if issues or any(codes):
        return 1
    print("Lint passed: Ruff, formatting, and no-comments checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

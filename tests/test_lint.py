import runpy
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def lint():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "lint.py"))


@pytest.mark.parametrize(
    "source",
    [
        "# explanation\nvalue = 1\n",
        "value = 1  # explanation\n",
        "value = 1  # noqa\n",
        "def example():\n    return 1  # type: ignore\n",
    ],
)
def test_python_comments_are_rejected(lint, source):
    issues = lint["check_python"](Path("example.py"), source)
    assert len(issues) == 1
    assert "NOC001" in issues[0]


@pytest.mark.parametrize(
    "source",
    [
        '"module documentation"\nvalue = 1\n',
        'class Example:\n    "class documentation"\n',
        'def example():\n    "function documentation"\n    return 1\n',
        'async def example():\n    "async documentation"\n    return 1\n',
        'value = 1\n"standalone explanation"\n',
        'value = 1\nf"standalone explanation {value}"\n',
    ],
)
def test_docstrings_and_string_comments_are_rejected(lint, source):
    issues = lint["check_python"](Path("example.py"), source)
    assert len(issues) == 1
    assert "NOC002" in issues[0]


def test_python_hashes_in_strings_are_allowed(lint):
    source = 'url = "https://example.com/#anchor"\ntext = """# literal content"""\n'
    assert lint["check_python"](Path("example.py"), source) == []


@pytest.mark.parametrize(
    "filename,source",
    [
        ("config.toml", 'value = "ok" # explanation\n'),
        ("workflow.yml", "name: Lint # explanation\n"),
        ("example.sh", 'echo "ok" # explanation\n'),
        ("Makefile", "lint:\n\tpython lint.py # explanation\n"),
    ],
)
def test_configuration_comments_are_rejected(lint, filename, source):
    issues = lint["check_hash_comments"](Path(filename), source)
    assert len(issues) == 1
    assert "NOC001" in issues[0]


@pytest.mark.parametrize(
    "filename,source",
    [
        ("config.toml", "url = \"https://example.com/#anchor\"\nliteral = 'C:\\'\n"),
        ("config.toml", 'text = """\n# literal content\n"""\n'),
        ("workflow.yml", "name: Don't allow comments # explanation\n"),
        ("example.sh", 'echo "# literal content"\n'),
    ],
)
def test_hash_detection_respects_language_syntax(lint, filename, source):
    issues = lint["check_hash_comments"](Path(filename), source)
    expected = 1 if filename == "workflow.yml" else 0
    assert len(issues) == expected


def test_repository_excludes_external_and_generated_files(lint, tmp_path):
    for folder in (".tools", ".venv", "jobs", "dist"):
        path = tmp_path / folder / "example.py"
        path.parent.mkdir()
        path.write_text("# external or generated\n")
    (tmp_path / "source.py").write_text("value = 1\n")
    assert lint["check_repository"](tmp_path) == []
    (tmp_path / "source.py").write_text("value = 1 # explanation\n")
    assert "source.py:1:11: NOC001" in lint["check_repository"](tmp_path)[0]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("value = 1 # explanation\n", 1),
        ('value = "# literal"\n', 0),
    ],
)
def test_lint_command_exit_status(tmp_path, source, expected):
    script = Path(__file__).resolve().parents[1] / "scripts" / "lint.py"
    target = tmp_path / "scripts" / "lint.py"
    target.parent.mkdir()
    target.write_text(script.read_text())
    (tmp_path / "example.py").write_text(source)
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    result = subprocess.run(
        [sys.executable, str(target)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if expected:
        assert "NOC001" in result.stdout
    else:
        assert "Lint passed" in result.stdout


def test_repository_reports_invalid_toml(lint, tmp_path):
    (tmp_path / "config.toml").write_text("value = [\n")
    assert "LNT001" in lint["check_repository"](tmp_path)[0]

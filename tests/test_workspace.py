import sys

import pytest

from agentflow.workspace import WorkspaceError, WorkspaceTools


def test_workspace_reads_and_writes_relative_text_files(tmp_path):
    workspace = WorkspaceTools(tmp_path)

    written = workspace.write_file({"path": "src/example.py", "content": "value = 42\n"})

    assert written["bytes_written"] == len(b"value = 42\n")
    assert workspace.read_file({"path": "src/example.py"})["content"] == "value = 42\n"
    assert "src/example.py" in workspace.list_files({})["files"]


def test_workspace_rejects_paths_that_escape_root(tmp_path):
    workspace = WorkspaceTools(tmp_path)

    with pytest.raises(WorkspaceError, match="escapes"):
        workspace.read_file({"path": "../outside.txt"})

    with pytest.raises(WorkspaceError, match="Absolute paths"):
        workspace.write_file({"path": str(tmp_path / "outside.txt"), "content": "no"})


def test_workspace_protects_git_and_generated_directories_from_writes(tmp_path):
    workspace = WorkspaceTools(tmp_path)

    with pytest.raises(WorkspaceError, match="protected"):
        workspace.write_file({"path": ".git/config", "content": "unsafe"})

    with pytest.raises(WorkspaceError, match="protected"):
        workspace.write_file({"path": "src/../.venv/config", "content": "unsafe"})


def test_test_execution_requires_opt_in(tmp_path):
    workspace = WorkspaceTools(tmp_path)

    with pytest.raises(WorkspaceError, match="--allow-tests"):
        workspace.run_tests(1)


def test_arbitrary_commands_require_opt_in_and_allowlist(tmp_path):
    command = [sys.executable, "-c", "print('ok')"]
    disabled = WorkspaceTools(tmp_path)
    with pytest.raises(WorkspaceError, match="--allow-shell"):
        disabled.run_command({"command": command}, 2)

    restricted = WorkspaceTools(tmp_path, allow_shell=True, allowed_commands=("pytest",))
    with pytest.raises(WorkspaceError, match="not in the configured allowlist"):
        restricted.run_command({"command": command}, 2)

    with pytest.raises(ValueError, match="non-empty executable allowlist"):
        WorkspaceTools(tmp_path, allow_shell=True)


def test_allowlisted_command_is_argv_only_and_obeys_timeout(tmp_path):
    workspace = WorkspaceTools(
        tmp_path, allow_shell=True, allowed_commands=(sys.executable,)
    )
    result = workspace.run_command(
        {"command": [sys.executable, "-c", "import time; time.sleep(2)"]}, 0.1
    )

    assert result.timed_out is True
    assert result.succeeded is False

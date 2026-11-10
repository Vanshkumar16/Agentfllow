"""Workspace-confined file operations and opt-in command execution."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class WorkspaceError(RuntimeError):
    """Raised when an operation is invalid or escapes the configured workspace."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}


class WorkspaceTools:
    def __init__(
        self,
        root: Path,
        *,
        allow_tests: bool = False,
        allow_shell: bool = False,
        allowed_commands: tuple[str, ...] = (),
        test_command: tuple[str, ...] | None = None,
        max_read_bytes: int = 64_000,
        max_write_bytes: int = 128_000,
    ) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace does not exist or is not a directory: {self.root}")
        if test_command is None:
            test_command = (sys.executable, "-m", "pytest", "-q")
        if (
            not isinstance(test_command, (list, tuple))
            or not test_command
            or any(not isinstance(part, str) or not part for part in test_command)
        ):
            raise ValueError("test_command must be a non-empty argument list.")
        if max_read_bytes < 1 or max_write_bytes < 1:
            raise ValueError("File size limits must be positive.")
        if allow_shell and not allowed_commands:
            raise ValueError("Arbitrary command execution requires a non-empty executable allowlist.")
        if any(not isinstance(command, str) or not command for command in allowed_commands):
            raise ValueError("Allowed commands must be non-empty executable names.")
        self.allow_tests = allow_tests
        self.allow_shell = allow_shell
        self.allowed_commands = {Path(command).stem.lower() for command in allowed_commands}
        self.test_command = tuple(test_command)
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    def _resolve(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise WorkspaceError("A non-empty relative path is required.")
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise WorkspaceError("Absolute paths are not allowed.")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceError("Path escapes the configured workspace.") from exc
        return resolved

    @staticmethod
    def _is_protected_path(relative_path: str) -> bool:
        return any(part.lower() in _IGNORED_DIRECTORIES for part in Path(relative_path).parts)

    def list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = arguments.get("path", ".")
        folder = self._resolve(relative)
        if not folder.is_dir():
            raise WorkspaceError(f"Not a directory: {relative}")
        files: list[str] = []
        for current, directories, filenames in os.walk(folder, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if name not in _IGNORED_DIRECTORIES
                and not (Path(current) / name).is_symlink()
            )
            for filename in sorted(filenames):
                path = Path(current) / filename
                if path.is_symlink():
                    continue
                files.append(path.relative_to(self.root).as_posix())
                if len(files) >= 500:
                    return {"files": files, "truncated": True}
        return {"files": files, "truncated": False}

    def snapshot(self, max_total_chars: int = 12_000) -> dict[str, Any]:
        if max_total_chars < 1:
            raise ValueError("max_total_chars must be positive.")
        files: dict[str, str] = {}
        remaining = max_total_chars
        listing = self.list_files({})
        truncated = listing["truncated"]
        for relative in listing["files"]:
            if Path(relative).suffix.lower() != ".py" and relative != "pyproject.toml":
                continue
            try:
                content = self.read_file({"path": relative})["content"]
            except WorkspaceError:
                truncated = True
                continue
            if len(content) > remaining:
                files[relative] = content[:remaining]
                truncated = True
                break
            files[relative] = content
            remaining -= len(content)
        return {"files": files, "truncated": truncated}

    def read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = arguments.get("path")
        path = self._resolve(relative)
        if not path.is_file():
            raise WorkspaceError(f"Not a file: {relative}")
        if path.stat().st_size > self.max_read_bytes:
            raise WorkspaceError(
                f"File exceeds the {self.max_read_bytes}-byte read limit: {relative}"
            )
        try:
            return {"path": path.relative_to(self.root).as_posix(), "content": path.read_text(encoding="utf-8")}
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"File is not UTF-8 text: {relative}") from exc

    def write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = arguments.get("path")
        content = arguments.get("content")
        path = self._resolve(relative)
        if not isinstance(content, str):
            raise WorkspaceError("'content' must be a string.")
        if self._is_protected_path(relative):
            raise WorkspaceError("Writes to protected workspace directories are not allowed.")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_write_bytes:
            raise WorkspaceError(
                f"Content exceeds the {self.max_write_bytes}-byte write limit."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not path.is_file():
            raise WorkspaceError(f"Destination is not a regular file: {relative}")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, delete=False
            ) as file:
                temporary_path = Path(file.name)
                file.write(encoded)
            temporary_path.replace(path)
        except OSError as exc:
            raise WorkspaceError(f"Could not write '{relative}': {exc}") from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return {"path": path.relative_to(self.root).as_posix(), "bytes_written": len(encoded)}

    def run_tests(self, timeout_seconds: float) -> CommandResult:
        if not self.allow_tests:
            raise WorkspaceError(
                "Test execution is disabled. Pass --allow-tests to explicitly enable it."
            )
        return self._run_process(self.test_command, timeout_seconds)

    def run_command(
        self, arguments: dict[str, Any], timeout_seconds: float
    ) -> CommandResult:
        if not self.allow_shell:
            raise WorkspaceError(
                "Arbitrary command execution is disabled. Pass --allow-shell to opt in."
            )
        command = arguments.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise WorkspaceError("'command' must be a non-empty list of strings.")
        executable = Path(command[0]).stem.lower()
        if executable not in self.allowed_commands:
            raise WorkspaceError(
                f"Command '{executable}' is not in the configured allowlist."
            )
        working_directory = self._resolve(arguments.get("cwd", "."))
        relative_cwd = str(arguments.get("cwd", "."))
        if self._is_protected_path(relative_cwd):
            raise WorkspaceError("Commands cannot run from a protected workspace directory.")
        if not working_directory.is_dir():
            raise WorkspaceError("Command working directory must be a workspace directory.")
        return self._run_process(command, timeout_seconds, cwd=working_directory)

    def _run_process(
        self, command: tuple[str, ...] | list[str], timeout_seconds: float, cwd: Path | None = None
    ) -> CommandResult:
        if timeout_seconds <= 0:
            raise WorkspaceError("The task time budget is exhausted.")
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd or self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                list(command),
                None,
                _as_text(exc.stdout),
                _as_text(exc.stderr),
                timed_out=True,
            )
        except OSError as exc:
            raise WorkspaceError(f"Could not start command '{command[0]}': {exc}") from exc
        return CommandResult(
            list(command),
            completed.returncode,
            completed.stdout[-16_000:],
            completed.stderr[-16_000:],
        )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-16_000:]
    return value[-16_000:]

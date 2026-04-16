"""Built-in tools for agent capabilities."""

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mybot.tools.base import tool

if TYPE_CHECKING:
    from mybot.core.agent import AgentSession


# --- Security helpers ---

# Dangerous shell command patterns that should be blocked
_DANGEROUS_COMMAND_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[a-zA-Z]*f|-[a-zA-Z]*r|--force|--recursive)\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*of=/dev/"),
    re.compile(r"\b:(){ :\|:& };:"),  # fork bomb
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r"\bchmod\s+(-[a-zA-Z]*R\s+)?[0-7]*777\b"),
    re.compile(r"\bcurl\b.*\|\s*(bash|sh|zsh)\b"),
    re.compile(r"\bwget\b.*\|\s*(bash|sh|zsh)\b"),
    re.compile(r"\bnc\s+-[a-zA-Z]*l"),  # netcat listen
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\binit\s+[06]\b"),
]


def _is_command_dangerous(command: str) -> str | None:
    """Check if a command matches dangerous patterns.

    Returns a warning message if dangerous, None if safe.
    """
    for pattern in _DANGEROUS_COMMAND_PATTERNS:
        if pattern.search(command):
            return (
                f"Blocked: command matches dangerous pattern ({pattern.pattern}). "
                "This command could cause irreversible damage."
            )
    return None


def _validate_path_in_workspace(path: str, workspace: Path) -> str | None:
    """Validate that a path is within the workspace directory.

    Returns an error message if the path is outside the workspace, None if safe.
    """
    try:
        resolved = Path(path).resolve()
        workspace_resolved = workspace.resolve()
        if not str(resolved).startswith(str(workspace_resolved)):
            return (
                f"Error: Access denied. Path '{path}' is outside the workspace "
                f"directory '{workspace_resolved}'. File operations are restricted "
                "to the workspace for security."
            )
    except (OSError, ValueError) as e:
        return f"Error: Invalid path '{path}': {e}"
    return None


def _get_workspace(session: "AgentSession") -> Path:
    """Get the workspace directory from the session context."""
    try:
        return session.shared_context.config.workspace
    except AttributeError:
        # Fallback for early tutorial steps without shared_context
        return Path.cwd()


# Filesystem tools


@tool(
    name="read",
    description="Read the contents of a text file (restricted to workspace)",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"},
        },
        "required": ["path"],
    },
)
async def read_file(path: str, session: "AgentSession") -> str:
    """Read and return the contents of a file at the given path."""
    workspace = _get_workspace(session)
    path_error = _validate_path_in_workspace(path, workspace)
    if path_error:
        return path_error
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied reading: {path}"
    except IsADirectoryError:
        return f"Error: Path is a directory, not a file: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool(
    name="write",
    description="Write content to a file (restricted to workspace)",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write"},
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
)
async def write_file(path: str, content: str, session: "AgentSession") -> str:
    """Write content to a file at the given path."""
    workspace = _get_workspace(session)
    path_error = _validate_path_in_workspace(path, workspace)
    if path_error:
        return path_error
    try:
        Path(path).write_text(content)
        return f"Successfully wrote to: {path}"
    except PermissionError:
        return f"Error: Permission denied writing to: {path}"
    except IsADirectoryError:
        return f"Error: Path is a directory, not a file: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(
    name="edit",
    description="Edit a file by replacing a string with new content (restricted to workspace)",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "old_text": {"type": "string", "description": "The text to replace"},
            "new_text": {
                "type": "string",
                "description": "The new text to replace with",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
)
async def edit_file(
    path: str, old_text: str, new_text: str, session: "AgentSession"
) -> str:
    """Edit a file by replacing old_text with new_text."""
    workspace = _get_workspace(session)
    path_error = _validate_path_in_workspace(path, workspace)
    if path_error:
        return path_error
    try:
        content = Path(path).read_text()
        if old_text not in content:
            return f"Error: '{old_text}' not found in {path}"
        new_content = content.replace(old_text, new_text)
        Path(path).write_text(new_content)
        return f"Successfully edited {path}"
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied editing: {path}"
    except Exception as e:
        return f"Error editing file: {e}"


# Shell tool


@tool(
    name="bash",
    description="Execute a bash shell command (dangerous commands are blocked)",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
        },
        "required": ["command"],
    },
)
async def bash(command: str, session: "AgentSession") -> str:
    """Execute a bash command and return the output."""
    danger = _is_command_dangerous(command)
    if danger:
        return danger
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode() if stdout else ""
        error = stderr.decode() if stderr else ""
        if output and error:
            return f"{output}\n{error}"
        return output or error or "Command completed with no output"
    except Exception as e:
        return f"Error executing command: {e}"

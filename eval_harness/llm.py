"""One isolated model call through the local Claude Code CLI (no API key needed).

By default the CLI is an agent: it can read files, run tools and load project instructions, which
would contaminate a grader or a baseline. Every call here runs it as a plain model: no tools, no
settings, no MCP servers, no slash commands, no saved session, from an empty temporary directory,
with the prompt on stdin.
"""
import subprocess
import tempfile

SYSTEM = "You answer using only the text in the user's message. You have no tools, files or memory."


def ask(prompt, model="haiku", system=SYSTEM, timeout=600):
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "", "--strict-mcp-config",
             "--disable-slash-commands", "--no-session-persistence", "--system-prompt", system],
            input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if proc.returncode != 0:
        return f"ERROR: {proc.stderr.strip()[:200]}"
    return proc.stdout.strip()

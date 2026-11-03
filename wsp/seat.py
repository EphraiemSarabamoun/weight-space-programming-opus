"""The model seat: one call of Claude Code in print mode, with every tool switched off.

This is the Claude counterpart of the authors' harness, which ran every turn as one `codex exec` call with all of
Codex's tools disabled and the one-line system instructions "Answer the user's request." (their
`harness/instructions.md`). Here every turn is one `claude -p` call:

  * `--tools ""` and `--strict-mcp-config` (with no MCP config): the model has no tools at all;
  * `--system-prompt` with the authors' exact one-line instructions;
  * `--setting-sources ""` and `disableAllHooks`: no user or project settings, no hooks;
  * CLAUDE_CODE_DISABLE_CLAUDE_MDS / _AUTO_MEMORY / _ATTACHMENTS / _GIT_INSTRUCTIONS: no CLAUDE.md, no memory;
  * a fresh, neutrally named working directory, so the environment block Claude Code adds names no task.

What still reaches the model besides the prompt (checked with a probe, see README): Claude Code's fixed one-line
SDK preamble before the system prompt, a short environment block (date, platform, the neutral working directory)
and an account note. None of it describes the task.

A turn's record is written to `out_dir`, mirroring the authors' files:
  prompt.txt     what was sent
  events.jsonl   the raw stream-json event stream
  answer.md      the model's reply (text blocks of the final assistant message)
  run.json       model, effort, command, usage, wall time, session id, and the tool audit
  stderr.log     the CLI's stderr
"""
import json
import os
import subprocess
import tempfile
import time

SYSTEM = "Answer the user's request."
MODEL = os.environ.get("WSP_MODEL", "claude-opus-5-5")

# Environment switches that keep any local Claude Code context (CLAUDE.md, memory, git notes) out of the prompt.
CLEAN_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_ATTACHMENTS": "1",
    "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
    "DISABLE_AUTOUPDATER": "1",
}


def _env():
    """The child environment: the caller's, plus the clean-context switches.

    ANTHROPIC_API_KEY is blanked unless WSP_ALLOW_API_KEY=1, so by default the run uses the Claude Code login
    (a subscription), not a metered key."""
    env = dict(os.environ)
    env.update(CLEAN_ENV)
    if env.get("WSP_ALLOW_API_KEY") != "1":
        env["ANTHROPIC_API_KEY"] = ""
    return env


def _command(effort, resume, persist):
    cmd = ["claude", "-p", "--model", MODEL, "--effort", effort,
           "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--settings", json.dumps({"disableAllHooks": True}),
           "--system-prompt", SYSTEM, "--output-format", "stream-json", "--verbose"]
    if resume:
        cmd += ["--resume", resume]
    if not persist:
        cmd.append("--no-session-persistence")
    return cmd


def _parse(lines):
    """Pull the reply's text blocks, the thinking, any tool use, the usage and the session id out of a stream-json
    event list."""
    text_blocks, thinking, tool_uses = [], [], []
    result, session, model = None, None, None
    for e in lines:
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init":
            session, model = e.get("session_id"), e.get("model")
        elif t == "assistant":
            msg_text = []
            for c in e.get("message", {}).get("content", []):
                if c.get("type") == "text":
                    msg_text.append(c.get("text", ""))
                elif c.get("type") == "thinking":
                    thinking.append(c.get("thinking", ""))
                elif c.get("type") in ("tool_use", "server_tool_use"):
                    tool_uses.append(c.get("name"))
            if msg_text:
                text_blocks.append("".join(msg_text))
        elif t == "result":
            result = e
    return text_blocks, thinking, tool_uses, result, session, model


def ask(prompt, effort, out_dir, resume=None, persist=False, cwd=None, timeout=6 * 3600, retries=2):
    """Send one message and record the turn in out_dir. Returns the run record (dict) with 'answer' added.

    `resume` continues an earlier conversation (its session id); `persist` must be True for any conversation that
    will be resumed later, and every turn of one conversation must pass the same neutral `cwd` (Claude Code files
    sessions by working directory). Without `cwd`, each call gets a fresh neutral directory. A reply that is an API error (is_error) is retried up to `retries` times after 60 s, and
    each failed attempt is kept as fail<k>_events.jsonl, as the authors did."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    cmd = _command(effort, resume, persist)
    for attempt in range(retries + 1):
        seat = cwd or tempfile.mkdtemp(prefix="seat")
        t0 = time.time()
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=seat, env=_env(), timeout=timeout)
        wall = time.time() - t0
        events = []
        for line in p.stdout.splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    events.append({"type": "unparsed", "line": line})
        texts, thinking, tools, result, session, model = _parse(events)
        failed = result is None or result.get("is_error") or not texts
        if failed and attempt < retries:
            with open(os.path.join(out_dir, f"fail{attempt + 1}_events.jsonl"), "w") as f:
                f.write(p.stdout)
            time.sleep(60)
            continue
        break
    # The reply is every text block of the response, in order. With interleaved thinking, Claude Code streams one
    # reply as several text blocks; keeping only the last one would drop anything written before a thinking block.
    answer = "\n".join(texts)
    with open(os.path.join(out_dir, "events.jsonl"), "w") as f:
        f.write(p.stdout)
    with open(os.path.join(out_dir, "stderr.log"), "w") as f:
        f.write(p.stderr)
    with open(os.path.join(out_dir, "answer.md"), "w") as f:
        f.write(answer)
    if thinking and any(thinking):
        with open(os.path.join(out_dir, "thinking.md"), "w") as f:
            f.write("\n\n---\n\n".join(thinking))
    usage = (result or {}).get("usage", {})
    run = {
        "model_requested": MODEL,
        "model_reported": model,
        "model_usage": list(((result or {}).get("modelUsage") or {}).keys()),
        "effort": effort,
        "system_prompt": SYSTEM,
        "command": cmd,
        "resume": resume,
        "session_id": session,
        "wall_s": round(wall, 1),
        "is_error": bool((result or {}).get("is_error")) or result is None,
        "result_text_if_error": (result or {}).get("result") if (result or {}).get("is_error") else None,
        "usage": {k: usage.get(k) for k in ("input_tokens", "cache_creation_input_tokens",
                                             "cache_read_input_tokens", "output_tokens")},
        "audit": {"tool_free": not tools, "tool_uses": tools, "assistant_messages": len(texts)},
        "attempts": attempt + 1,
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    run["answer"] = answer
    return run

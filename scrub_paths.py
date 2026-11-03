"""Prepare results/ for publishing: replace this machine's paths with placeholders (as the authors did in their logs),
and drop the opaque `signature` field of thinking blocks from the raw event streams.

  <home>     the user's home directory
  <scratch>  the session scratch folder the multi-turn seats and sandboxes ran in
  <tmp>      the system temporary folder (one-shot seats' working directories)

Thinking signatures: Claude Code returned every thinking block with empty text and an encrypted, base64 `signature`
that no reader can use; random substrings of that base64 match secret-scanner token shapes (e.g. a Google refresh
token), so the field is removed from events.jsonl. Nothing readable is lost.

Only text files are rewritten (.json, .jsonl, .md, .txt, .log); network files (.npy, .npz) are left alone. Nothing
else changes: no number, and no reply text beyond the paths themselves.
Run it once before committing results; running it again is a no-op.

  python3 scrub_paths.py [--check]     (--check: report what would change, change nothing)
"""
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
HOME = os.path.expanduser("~")
USER_KEY = HOME.replace("/", "-")                     # e.g. -Users-name, as it appears in Claude Code's paths
RULES = [
    (re.compile(r"/private/tmp/claude-\d+/" + re.escape(USER_KEY) + r"/[0-9a-f-]{36}/scratchpad"), "<scratch>"),
    (re.compile(r"/private/var/folders/[A-Za-z0-9_]+/[A-Za-z0-9_]+/T"), "<tmp>"),
    (re.compile(re.escape(HOME)), "<home>"),
    (re.compile(re.escape(USER_KEY)), "-<home>"),
]
TEXT = (".json", ".jsonl", ".md", ".txt", ".log")


def drop_signatures(jsonl):
    """Remove `signature` from every thinking block in a stream-json event file; other lines pass unchanged."""
    out = []
    for line in jsonl.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue
        content = e.get("message", {}).get("content") if isinstance(e.get("message"), dict) else None
        if isinstance(content, list) and any(isinstance(c, dict) and "signature" in c for c in content):
            for c in content:
                if isinstance(c, dict):
                    c.pop("signature", None)
            line = json.dumps(e, ensure_ascii=False, separators=(",", ":"))
        out.append(line)
    return "\n".join(out) + ("\n" if jsonl.endswith("\n") else "")


def scrub(text, name=""):
    for pattern, placeholder in RULES:
        text = pattern.sub(placeholder, text)
    if name.endswith(".jsonl") and "events" in name:
        text = drop_signatures(text)
    return text


def main():
    check = "--check" in sys.argv
    changed = 0
    for d, _, files in os.walk(ROOT):
        for f in files:
            if not f.endswith(TEXT):
                continue
            p = os.path.join(d, f)
            old = open(p, encoding="utf-8", errors="surrogateescape").read()
            new = scrub(old, f)
            if new != old:
                changed += 1
                if not check:
                    open(p, "w", encoding="utf-8", errors="surrogateescape").write(new)
    print(f"{'would change' if check else 'changed'} {changed} files")
    left = sum(HOME in open(os.path.join(d, f), errors="ignore").read()
               for d, _, fs in os.walk(ROOT) for f in fs if f.endswith(TEXT))
    print(f"files still naming the home directory: {left}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

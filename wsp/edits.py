"""The `edits` language of Experiments 4 and 5: read the LAST ```edits block of a reply and apply it to tensors.

The language is specified in the authors' prompts (Experiment 5, "HOW TO WRITE THE WEIGHTS"):

    W1[12, 0] = 2.5                      one entry
    W1[0:6, 3] = [1, 0, -1, 0, 2, 0]     explicit values along a range (as many values as entries)
    b1[0:4] = -1                         fill a range; ':' alone is the whole axis
    W1[3:600:60, 0] = [...]              strided range a:b:s
    W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]   a 2-D list for a 2-D block
    W3[:, 2] *= 2                        scale (also += and -=)

Experiment 4 uses the same language with dotted tensor names (`actor0.W`) and one more statement, `ZERO <tensor>`.
'#' starts a comment, a long list may continue over the next lines, statements apply in order, a statement that
cannot be read is skipped while the others still apply, and |v| <= 1e6. Their reader was fixed to accept a space
after an opening bracket (`[ [ ...`); this one accepts any whitespace.

The authors' own reader is not released. This one is checked against it in `tests/test_edits.py`: reading the
authors' three Experiment 5 replies reproduces their scored networks exactly.
"""
import ast
import re

import numpy as np

BLOCK = re.compile(r"```[ \t]*edits[ \t]*\n(.*?)```", re.S)
STATEMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(\[[^\]]*\])?\s*(=|\*=|\+=|-=)\s*(.+?)\s*$", re.S)
ZERO = re.compile(r"^\s*ZERO\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$")
MAX_ABS = 1e6


def last_block(reply):
    """The text of the last ```edits block, or None. An unclosed final block is read to the end of the reply."""
    blocks = BLOCK.findall(reply)
    if blocks:
        return blocks[-1]
    m = re.search(r"```[ \t]*edits[ \t]*\n(.*)$", reply, re.S)
    return m.group(1) if m else None


def _strip_comment(line):
    return line.split("#", 1)[0]


def statements(block):
    """Split a block into statements: one per line, except that a line continues while brackets are unbalanced."""
    out, buf, depth = [], "", 0
    for raw in block.splitlines():
        line = _strip_comment(raw)
        if not line.strip() and not buf:
            continue
        buf = (buf + " " + line) if buf else line
        depth = buf.count("[") - buf.count("]")
        # The index bracket on the left also counts, so a statement is complete when every bracket is closed.
        if depth <= 0:
            if buf.strip():
                out.append(buf.strip())
            buf, depth = "", 0
    if buf.strip():
        out.append(buf.strip())
    return out


def _index(text, shape):
    """Parse '[a, b:c, ::s]' into a tuple of ints and slices for an array of `shape`."""
    inner = text.strip()[1:-1]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != len(shape):
        raise ValueError(f"index {text} has {len(parts)} axes, tensor has {len(shape)}")
    idx = []
    for p in parts:
        if ":" in p:
            fields = [f.strip() for f in p.split(":")]
            if len(fields) > 3:
                raise ValueError(f"bad slice {p}")
            nums = [int(f) if f else None for f in fields] + [None] * (3 - len(fields))
            idx.append(slice(*nums))
        else:
            idx.append(int(p))
    return tuple(idx)


def _value(text):
    v = ast.literal_eval(text.strip())
    a = np.asarray(v, dtype=np.float64)
    if not np.isfinite(a).all() or (np.abs(a) > MAX_ABS).any():
        raise ValueError("value out of range")
    return a


def apply(reply, tensors):
    """Apply the reply's last edits block to `tensors` (dict name -> float array, edited in place).

    Returns a report like the authors' `edit_report`: whether a block was found, statements read and applied,
    entries written, the errors of skipped statements, and the tensors touched."""
    block = last_block(reply)
    report = {"block": block is not None, "statements": 0, "applied": 0, "entries_written": 0, "errors": [],
              "tensors_touched": []}
    if block is None:
        return report
    touched = set()
    for st in statements(block):
        report["statements"] += 1
        try:
            z = ZERO.match(st)
            if z:  # Experiment 4's extra statement: set a whole tensor to 0
                if z.group(1) not in tensors:
                    raise ValueError(f"unknown tensor {z.group(1)}")
                tensors[z.group(1)][...] = 0
                report["applied"] += 1
                report["entries_written"] += int(tensors[z.group(1)].size)
                touched.add(z.group(1))
                continue
            m = STATEMENT.match(st)
            if not m:
                raise ValueError("unreadable statement")
            name, idx_text, op, val_text = m.groups()
            if name not in tensors:
                raise ValueError(f"unknown tensor {name}")
            arr = tensors[name]
            idx = _index(idx_text, arr.shape) if idx_text else tuple(slice(None) for _ in arr.shape)
            target = arr[idx]
            val = _value(val_text)
            if val.ndim and val.shape != np.shape(target):
                raise ValueError(f"value shape {val.shape} != target shape {np.shape(target)}")
            if op == "=":
                arr[idx] = val
            elif op == "*=":
                arr[idx] = target * val
            elif op == "+=":
                arr[idx] = target + val
            else:
                arr[idx] = target - val
            report["applied"] += 1
            report["entries_written"] += int(np.size(target))
            touched.add(name)
        except Exception as e:  # a statement that cannot be read is skipped; the others still apply
            report["errors"].append({"statement": st[:200], "error": str(e)})
    report["tensors_touched"] = sorted(touched)
    return report

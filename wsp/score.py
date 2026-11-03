"""Scoring helpers shared by the experiments: read a labelled line out of a reply and map it to a class by keyword.

The authors' scorers are described, not released: Experiment 1 maps the `Game:` line to a game by keywords (Frogger
counts as Freeway), Experiment 2 maps the `Images:` line to a shape (circle / disc / round / oval; square /
rectangle / box; triangle / pyramid / wedge). The keyword lists below implement those descriptions. When a line
names more than one class, the class named first wins, and the scorer marks the case `ambiguous` so it is read by
hand.
"""
import re

GAME_KEYWORDS = {
    "breakout": ["breakout", "break-out", "brick", "arkanoid"],
    "pong": ["pong"],
    "pacman": ["pac-man", "pacman", "pac man", "ms. pac", "ms pac"],
    "freeway": ["freeway", "frogger", "road-crossing", "road crossing", "traffic-crossing", "traffic crossing",
                "crossy road", "chicken"],
}

SHAPE_KEYWORDS = {
    "circle": ["circle", "circular", "disc", "disk", "round", "oval", "ring", "ellipse"],
    "square": ["square", "rectangle", "rectangular", "box"],
    "triangle": ["triangle", "triangular", "pyramid", "wedge"],
}


def field(answer, name):
    """The text of the reply's `name` field; '' if absent.

    Two layouts occur: `Name: text` on one line (Astra's, and usually Opus's), and a heading line `**Name**` with
    the text in the paragraph below it (Opus sometimes formats the fields as markdown sections). The last
    occurrence wins, since the requested format puts the answer fields at the end."""
    hits = re.findall(rf"^[\s>*#-]*\**{re.escape(name)}\**\s*:\**[ \t]*(\S.*)$", answer, flags=re.I | re.M)
    heading = list(re.finditer(rf"^[\s>#-]*\**{re.escape(name)}\**\s*:?\**\s*$", answer, flags=re.I | re.M))
    if heading and (not hits or heading[-1].start() > answer.rfind(hits[-1])):
        rest = answer[heading[-1].end():].strip("\n")
        para = re.split(r"\n\s*\n", rest, maxsplit=1)[0]
        return " ".join(line.strip(" -*") for line in para.splitlines()).strip()
    return hits[-1].strip() if hits else ""


def confidence(answer):
    """The integer percentage on the `Confidence:` line, or None."""
    m = re.search(r"(\d{1,3})", field(answer, "Confidence"))
    return int(m.group(1)) if m else None


def classify(line, keywords):
    """Map a line to (class, ambiguous). Class is the keyword class appearing earliest in the line, or None."""
    low = line.lower()
    found = {}
    for cls, words in keywords.items():
        # Whole words only (a word may take a plural s), so "round" does not match inside "background".
        pos = [m.start() for w in words for m in re.finditer(rf"\b{re.escape(w)}s?\b", low)]
        if pos:
            found[cls] = min(pos)
    if not found:
        return None, False
    best = min(found, key=found.get)
    return best, len(found) > 1

"""Generate the example scripts from the example notebooks.

The code cells are copied unchanged, so each script runs exactly the same code as its
notebook. Markdown cells become comments. Edit the notebook, then regenerate:

    python scripts/nb_to_script.py                       # all notebooks
    python scripts/nb_to_script.py pyutils_basics ...    # selected notebooks
"""
import json
import re
import sys
import textwrap
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOKS = [
    "pyutils_basics",
    "pyutils_on_EAF",
    "pyutils_multifile",
    "pyplot_demo",
    "pycut_demo",
    "pyutils_mctruth",
    "pyutils_ensemble_analysis",
]
WIDTH = 88
RULE = "# " + "=" * (WIDTH - 2)


def clean(line):
    # Drop HTML anchors and markdown emphasis that read badly as plain text
    line = re.sub(r"<a [^>]*></a>", "", line)
    line = re.sub(r"^\s*>\s?", "", line)
    return line.replace("**", "").rstrip()


def markdown_lines(text):
    # Yield plain text lines, wrapped, keeping tables and code blocks as they are
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.lstrip().startswith("|"):
            yield line.rstrip()
            continue
        line = clean(line)
        indent = re.match(r"\s*(?:[-*]|\d+\.)?\s*", line).group(0)
        yield from textwrap.wrap(line, WIDTH - 2, subsequent_indent=" " * len(indent)) or [""]


def markdown_to_comment(text):
    out = []
    for line in markdown_lines(text):
        heading = re.match(r"#+\s*(.*)", line)
        if heading:
            out += ["", RULE, f"# {heading.group(1)}", RULE]
        else:
            out.append(f"# {line}".rstrip())
    # Remove blank comment lines at either end
    while out and out[0] in ("", "#"):
        out.pop(0)
    while out and out[-1] == "#":
        out.pop()
    return "\n".join(out)


def header(name, text):
    # Module docstring from the first markdown cell
    lines = list(markdown_lines(text))
    title = re.sub(r"^#+\s*", "", lines[0]).replace("`", "")
    body = "\n".join(re.sub(r"^#+\s*", "", line) for line in lines[1:]).strip()
    return (
        f'"""\n{title}\n{"=" * len(title)}\n\n{body}\n\n'
        f"Generated from examples/notebooks/{name}.ipynb by scripts/nb_to_script.py.\n"
        f'Edit the notebook and regenerate, rather than editing this script.\n"""'
    )


def convert(name):
    nb = json.loads((EXAMPLES / "notebooks" / f"{name}.ipynb").read_text())
    cells = [("".join(c["source"]).strip("\n"), c["cell_type"]) for c in nb["cells"]]
    cells = [(text, kind) for text, kind in cells if text.strip()]

    blocks = []
    if cells and cells[0][1] == "markdown":
        blocks.append(header(name, cells.pop(0)[0]))
    for text, kind in cells:
        blocks.append(markdown_to_comment(text) if kind == "markdown" else text)

    out_path = EXAMPLES / "scripts" / f"{name}.py"
    out_path.write_text("\n\n".join(blocks) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    for name in sys.argv[1:] or NOTEBOOKS:
        convert(name)

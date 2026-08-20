"""Render a working paper to PDF via headless Chrome.

Lives here rather than in a scratch dir because it has been rewritten three times after
the scratchpad was cleared mid-session.

    uv run --with markdown python tools/events_working_papers/md2pdf.py IN.md OUT.pdf
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: letter; margin: 20mm 18mm; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.5;
       color: #111; max-width: none; }
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 0.6em; }
h2 { font-size: 13pt; margin: 1.6em 0 0.5em; border-bottom: 1px solid #ccc;
     padding-bottom: 0.2em; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 1.2em 0 0.4em; page-break-after: avoid; }
p { margin: 0 0 0.7em; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 0.87em;
       background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }
pre { background: #f6f6f6; padding: 0.7em 0.9em; border-radius: 4px; overflow-x: auto;
      font-size: 0.85em; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.92em;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 0.3em 0.6em; text-align: left; }
th { background: #f0f0f0; }
td:nth-child(n+2) { text-align: right; }
blockquote { margin: 0.8em 0; padding: 0.3em 0 0.3em 1em; border-left: 3px solid #ccc;
             color: #444; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
li { margin: 0.25em 0; }
"""


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    html = markdown.markdown(src.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{html}")
        tmp = fh.name
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={dst}", f"file://{tmp}"], check=True, capture_output=True)
    print(f"wrote {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

"""Render the Divine Blueprint LaTeX paper to the canonical PDF."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "THE_DIVINE_BLUEPRINT.tex"
BUILD_DIR = ROOT / "tmp" / "tex"
ROOT_PDF = ROOT / "THE_DIVINE_BLUEPRINT.pdf"


def render() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex is required to render THE_DIVINE_BLUEPRINT.pdf")

    command = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD_DIR}",
        str(SOURCE),
    ]
    for _ in range(2):
        subprocess.run(command, cwd=ROOT, check=True)

    built_pdf = BUILD_DIR / "THE_DIVINE_BLUEPRINT.pdf"
    if not built_pdf.exists():
        raise RuntimeError(f"Expected PDF was not generated: {built_pdf}")
    shutil.copy2(built_pdf, ROOT_PDF)
    print(f"Updated {ROOT_PDF}")


if __name__ == "__main__":
    render()

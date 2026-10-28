"""Load the ShopHelp return/shipping handbook as retrieval chunks."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

HANDBOOK_DIR = Path(__file__).resolve().parents[1] / "shophelp" / "handbook"


class Passage(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    text: str
    source: str
    tags: tuple[str, ...] = Field(default_factory=tuple)


def load_handbook(path: Path | None = None) -> tuple[Passage, ...]:
    root = path or HANDBOOK_DIR
    passages: list[Passage] = []
    for md in sorted(root.glob("*.md")):
        passages.extend(_split_markdown(md))
    if not passages:
        raise FileNotFoundError(f"no handbook markdown in {root}")
    return tuple(passages)


def _split_markdown(path: Path) -> list[Passage]:
    lines = path.read_text(encoding="utf-8").splitlines()
    doc_title = path.stem.replace("_", " ").title()
    chunks: list[list[str]] = []
    current: list[str] = []
    heading = doc_title
    headings: list[str] = []
    for line in lines:
        if line.startswith("# ") and not current and not chunks:
            doc_title = line[2:].strip()
            heading = doc_title
            continue
        if line.startswith("## "):
            if current:
                chunks.append(current)
                headings.append(heading)
            heading = line[3:].strip()
            current = [line]
            continue
        current.append(line)
    if current:
        chunks.append(current)
        headings.append(heading)
    out: list[Passage] = []
    for i, (title, body_lines) in enumerate(zip(headings, chunks, strict=True)):
        text = "\n".join(body_lines).strip()
        if len(text) < 40:
            continue
        out.append(
            Passage(
                id=f"{path.stem}-{i}",
                title=title,
                text=text,
                source=path.name,
                tags=(path.stem,),
            )
        )
    return out

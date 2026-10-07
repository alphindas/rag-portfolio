"""
ingest/chunk.py

Splits source documents into retrieval-ready chunks.

Why not just split by character count or word count?
------------------------------------------------------
BGE (like most transformer encoders) has a hard token limit (512 for
bge-base-en). If we chunk by characters/words, we're guessing at the
token count indirectly -- and text like tables, code blocks, or dense
technical prose can pack far more tokens per word than plain English.
Guess wrong and the embedding model silently truncates your chunk,
which means the tail of it is invisible to retrieval. That's a subtle
bug that's easy to ship and hard to notice, so we count real tokens
using the actual model tokenizer instead.

Why structure-aware splitting first?
------------------------------------------------------
If we just slide a token window across raw text, we routinely slice
a chunk in half mid-sentence or mid-list-item. That hurts both the
embedding (it now represents a fragment, not a coherent idea) and the
final citation (a chunk that starts mid-sentence reads badly when shown
to a user as a cited source). So we first split on markdown structure
(headers, paragraphs), then pack those structural units into
token-budgeted chunks, only falling back to a hard token-window split
if a single paragraph is itself too large.

Output
------------------------------------------------------
Each chunk is a dict:
{
    "text": str,                # the chunk's actual text
    "source": str,               # source file path
    "chunk_index": int,          # position within the source doc
    "header_path": str,          # e.g. "Installation > Requirements"
    "token_count": int,          # token count per the embedding tokenizer
}

This metadata matters later: header_path and source are what make a
generated answer's citation ("see Installation > Requirements") mean
something to a user, instead of just "chunk #47".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Must match the embedding model we'll use in embed.py -- the tokenizer
# defines what "512 tokens" actually means for BGE, and that number
# differs from, say, a GPT tokenizer's count for the same text.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en"

# Leave headroom below BGE's 512 limit. BGE prepends a [CLS] token and
# appends [SEP], and we'd rather have a small safety margin than chunks
# that land right at the boundary and get silently truncated by a
# one-token miscount.
MAX_TOKENS = 480

# ~15-20% overlap. Enough that a query whose answer spans a chunk
# boundary still has a decent chance of hitting a chunk that contains
# the full answer, without bloating the index with heavily duplicated
# content.
OVERLAP_TOKENS = 80

_tokenizer = None


def _get_tokenizer():
    """Lazy-load the tokenizer once per process -- it's not free to construct."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return _tokenizer


def count_tokens(text: str) -> int:
    """Real token count per the BGE tokenizer, not an estimate."""
    tokenizer = _get_tokenizer()
    # add_special_tokens=False: we're counting content tokens; the
    # MAX_TOKENS budget above already reserves room for [CLS]/[SEP].
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Structural parsing
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """A structural unit of the document: a paragraph under a header path."""
    text: str
    header_path: str


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def parse_markdown_blocks(text: str) -> list[Block]:
    """
    Split markdown into (paragraph, header_path) blocks.

    We track a stack of active headers by level so that a paragraph
    under an H3 nested inside an H2 inside an H1 gets a header_path like
    "Setup > Installation > Requirements" -- that breadcrumb is what
    gets attached to every chunk under it.
    """
    lines = text.split("\n")
    header_stack: list[tuple[int, str]] = []  # (level, title)
    blocks: list[Block] = []
    current_para_lines: list[str] = []

    def flush_paragraph():
        para = "\n".join(current_para_lines).strip()
        if para:
            header_path = " > ".join(title for _, title in header_stack)
            blocks.append(Block(text=para, header_path=header_path))
        current_para_lines.clear()

    for line in lines:
        header_match = _HEADER_RE.match(line)
        if header_match:
            flush_paragraph()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            # Pop any headers at this level or deeper -- a new H2 ends
            # the previous H2's (and any nested H3's) scope.
            header_stack = [h for h in header_stack if h[0] < level]
            header_stack.append((level, title))
        elif line.strip() == "":
            # Blank line: paragraph boundary.
            flush_paragraph()
        else:
            current_para_lines.append(line)

    flush_paragraph()
    return blocks


# ---------------------------------------------------------------------------
# Packing blocks into token-budgeted chunks
# ---------------------------------------------------------------------------

def _split_oversized_block(block: Block) -> list[Block]:
    """
    Fallback for a single paragraph that alone exceeds MAX_TOKENS
    (e.g. a giant code sample or table with no blank lines). We do a
    hard token-window split here since there's no more structure left
    to respect -- this should be rare in clean docs, common in messy
    ones.
    """
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(block.text, add_special_tokens=False)
    if len(token_ids) <= MAX_TOKENS:
        return [block]

    sub_blocks = []
    start = 0
    while start < len(token_ids):
        end = min(start + MAX_TOKENS, len(token_ids))
        sub_text = tokenizer.decode(token_ids[start:end])
        sub_blocks.append(Block(text=sub_text, header_path=block.header_path))
        if end == len(token_ids):
            break
        start = end - OVERLAP_TOKENS  # slide back for overlap
    return sub_blocks


def pack_blocks(blocks: list[Block]) -> list[Block]:
    """
    Greedily pack consecutive blocks (that share reasonable context)
    into chunks up to MAX_TOKENS, splitting any oversized single block
    on its own. Overlap is applied between packed chunks by carrying
    the tail of one chunk's blocks into the start of the next.
    """
    # First, make sure no individual block exceeds the budget.
    normalized: list[Block] = []
    for b in blocks:
        normalized.extend(_split_oversized_block(b))

    packed: list[Block] = []

    # current_units holds (text, token_count, header_path) triples for the
    # chunk being built. Tracking header_path per-unit (not as one shared
    # variable) is what lets us correctly label a chunk that starts with
    # carried-over overlap content from a different section than the
    # fresh content that follows it.
    current_units: list[tuple[str, int, str]] = []
    current_tokens = 0

    def flush() -> list[tuple[str, int, str]]:
        """Emit the current chunk and return the trailing units to seed
        the next chunk's overlap (by token budget, taken from the end).
        The chunk's header_path is taken from its FIRST unit: a chunk
        commonly spans multiple headers (e.g. the tail of "Installation"
        plus the start of "Usage"), and the first unit's header tells a
        reader/citation where this chunk actually starts -- what matters
        when someone clicks through to "see source"."""
        nonlocal current_units, current_tokens
        if current_units:
            packed.append(
                Block(
                    text="\n\n".join(t for t, _, _ in current_units),
                    header_path=current_units[0][2],
                )
            )
        # Walk backward from the end, collecting units until we've
        # covered roughly OVERLAP_TOKENS worth of trailing context.
        # Each carried unit keeps its own original header_path, so the
        # next chunk's flush() will correctly label itself from whatever
        # section the carried content actually came from.
        carry: list[tuple[str, int, str]] = []
        carried_tokens = 0
        for t, tok, hp in reversed(current_units):
            if carried_tokens >= OVERLAP_TOKENS:
                break
            carry.insert(0, (t, tok, hp))
            carried_tokens += tok
        current_units = []
        current_tokens = 0
        return carry

    for b in normalized:
        b_tokens = count_tokens(b.text)

        if current_tokens + b_tokens > MAX_TOKENS and current_units:
            carry = flush()
            current_units = carry
            current_tokens = sum(tok for _, tok, _ in carry)

        current_units.append((b.text, b_tokens, b.header_path))
        current_tokens += b_tokens

    if current_units:
        packed.append(
            Block(
                text="\n\n".join(t for t, _, _ in current_units),
                header_path=current_units[0][2],
            )
        )
    return packed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_document(filepath: str) -> list[dict]:
    """
    Read a markdown file and return a list of chunk dicts, ready to be
    embedded and indexed.
    """
    text = Path(filepath).read_text(encoding="utf-8")

    # Resolve to an absolute, canonical path BEFORE it becomes part of
    # source (and therefore part of the chunk_id in embed.py / bm25_index.py).
    # Without this, the same physical file gets a different source string
    # depending on the caller CWD and how the path was typed, which would
    # silently break RRF fusion in hybrid.py.
    normalized_source = str(Path(filepath).resolve())

    blocks = parse_markdown_blocks(text)
    packed = pack_blocks(blocks)

    chunks = []
    for i, block in enumerate(packed):
        chunks.append(
            {
                "text": block.text,
                "source": normalized_source,
                "chunk_index": i,
                "header_path": block.header_path,
                "token_count": count_tokens(block.text),
            }
        )
    return chunks


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python chunk.py <path-to-markdown-file>")
        sys.exit(1)

    result = chunk_document(sys.argv[1])
    print(f"Produced {len(result)} chunks from {sys.argv[1]}\n")
    for c in result:
        print(f"--- chunk {c['chunk_index']} ({c['token_count']} tokens) ---")
        print(f"header_path: {c['header_path']}")
        print(c["text"][:200].replace("\n", " ") + ("..." if len(c["text"]) > 200 else ""))
        print()
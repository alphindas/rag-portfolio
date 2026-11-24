"""
generate/citations.py

Checks the [n] citations in an answer, in code, instead of trusting the prompt.
An answer is flagged when it
  - cites a number that does not exist ("[4]" when only 2 chunks were given), or
  - has no citations at all.
A plain "Not documented in the provided context." is allowed to have none.
"""

import re

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1], [2], [1, 2]
DECLINE_PHRASE = "not documented in the provided context"


def check_citations(answer: str, n_chunks: int) -> dict:
    """
    status: "ok" | "declined" | "no_citations" | "invalid"
    n_cited: how many [n] markers the answer contains
    invalid: cited numbers outside 1..n_chunks
    """
    numbers = [int(n) for group in _CITE_RE.findall(answer) for n in group.split(",")]
    invalid = sorted({n for n in numbers if not 1 <= n <= n_chunks})
    if invalid:
        status = "invalid"
    elif numbers:
        status = "ok"
    elif DECLINE_PHRASE in answer.lower():
        status = "declined"
    else:
        status = "no_citations"
    return {"status": status, "n_cited": len(numbers), "invalid": invalid}

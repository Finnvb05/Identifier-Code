#!/usr/bin/env python3
"""specimen_registry.py -- look up and create specimens in the PaperSteel app.

Sits on top of identifier_push.py (same stark-ingest route, same token) and adds
the two things the measuring station needs beyond pushing a reading:

    list_specimens()            every specimen code, for the search box
    next_code("PS-V4")          the next free number in a series
    create_specimen("PS-V4-39") insert it, letting the DB reject a collision

WHY THE DATABASE MINTS, NOT US
The code is a shared, human-readable identifier. The press, the tensile bench
and the app can all create specimens. If this station picks a number from a
list it fetched a minute ago, two machines can pick the same one, and afterwards
there is no way to tell the two specimens apart. So we propose a code and let
the insert fail if someone got there first, then try the next number.

That has a consequence worth stating plainly: a NEW specimen cannot be created
while the Pi is offline. Measurements against an EXISTING code still queue in
outbox/ and go up later, as before.
"""
from __future__ import annotations

import re
import time

from identifier_push import PS_CODE, ingest, load_conf

SERIES = re.compile(r"^(PS-[A-Z0-9]+)-(\d+)$")
MAX_ATTEMPTS = 8


def list_specimens(limit: int = 2000) -> list[str]:
    """Every specimen code the app knows, newest first where the DB allows it."""
    conf = load_conf()
    rows = ingest(conf, {"op": "select", "table": "specimens",
                         "columns": "specimen_id", "limit": limit})
    return [r["specimen_id"] for r in rows if r.get("specimen_id")]


def series_of(code: str) -> tuple[str, int] | None:
    """'PS-V4-038' -> ('PS-V4', 38). None if it is not a numbered series."""
    m = SERIES.match(code)
    return (m.group(1), int(m.group(2))) if m else None


def next_code(prefix: str, known: list[str] | None = None, width: int = 3) -> str:
    """The lowest unused number in `prefix`, zero-padded to `width`.

    `known` lets a caller reuse a list it already fetched. Treat the result as a
    PROPOSAL -- only a successful insert makes it yours.
    """
    prefix = prefix.strip().upper().rstrip("-")
    full = SERIES.match(prefix)
    if full:
        prefix = full.group(1)     # a whole code was pasted; take its series
    if not re.match(r"^PS-[A-Z0-9]+$", prefix):
        raise ValueError(f"series must look like PS-V4, got {prefix!r}")
    codes = list_specimens() if known is None else known
    used = {n for p, n in filter(None, (series_of(c) for c in codes)) if p == prefix}
    n = 1
    while n in used:
        n += 1
    return f"{prefix}-{n:0{width}d}"


def create_specimen(code: str, **fields) -> str:
    """Insert one specimen. Raises if the code is taken or the insert is refused."""
    if not PS_CODE.match(code):
        raise ValueError(f"bad specimen code: {code!r} (expected e.g. PS-V4-038)")
    conf = load_conf()
    row = {"specimen_id": code, **fields}
    ingest(conf, {"op": "insert", "table": "specimens", "rows": [row]})
    return code


def claim_next(prefix: str, **fields) -> str:
    """Mint a code and claim it, retrying past anyone who beats us to it.

    The database's uniqueness on specimen_id is the arbiter -- not our view of
    the list, which is always a little stale.
    """
    codes = list_specimens()
    for attempt in range(MAX_ATTEMPTS):
        code = next_code(prefix, known=codes)
        try:
            return create_specimen(code, **fields)
        except RuntimeError as e:
            msg = str(e).lower()
            if not any(w in msg for w in ("duplicate", "unique", "conflict", "already")):
                raise                      # a real failure, not a race
            codes.append(code)             # someone took it; move on
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"could not claim a code in {prefix} after "
                       f"{MAX_ATTEMPTS} tries - is another station looping?")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        codes = list_specimens()
        print(f"{len(codes)} specimens")
        for c in codes[:40]:
            print(" ", c)
    elif sys.argv[1] == "next" and len(sys.argv) == 3:
        print(next_code(sys.argv[2]))
    elif sys.argv[1] == "claim" and len(sys.argv) == 3:
        print("claimed:", claim_next(sys.argv[2]))
    else:
        sys.exit("usage: specimen_registry.py [next PS-V4 | claim PS-V4]")
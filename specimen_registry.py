#!/usr/bin/env python3
"""specimen_registry.py -- look up specimen codes in the PaperSteel app.

Sits on top of identifier_push.py (same route, same token) and adds the one
thing the measuring station needs beyond pushing a reading: the list of
specimen codes, for the search box.

WHY THERE IS NO create_specimen HERE ANY MORE
Until 2 November this module could also mint a code and insert a row into
`specimens`. That is now explicitly out of scope: measurements go only to
`identifier_measurements`, and the `identifier-ingest` function the station
talks to cannot write to `specimens` at all -- it can only select from it.
Specimens are created in the production app.

The three PS-TEST specimens created on 26 October came from the old path and
are being marked invalid at Hugo's end.
"""
from __future__ import annotations

import re

from identifier_push import ingest, load_conf

SERIES = re.compile(r"^(PS-[A-Z0-9]+)-(\d+)$")


def list_specimens(limit: int = 2000) -> list[str]:
    """Every specimen code the app knows, for the station's search box."""
    conf = load_conf()
    rows = ingest(conf, {"op": "select", "table": "specimens",
                         "columns": "specimen_id", "limit": limit})
    return [r["specimen_id"] for r in rows if r.get("specimen_id")]


def series_of(code: str) -> tuple[str, int] | None:
    """'PS-V4-038' -> ('PS-V4', 38). None if it is not a numbered series."""
    m = SERIES.match(code)
    return (m.group(1), int(m.group(2))) if m else None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit("usage: specimen_registry.py        (lists the known codes)\n"
                 "creating specimens from the station is no longer supported")
    codes = list_specimens()
    print(f"{len(codes)} specimens")
    for c in codes[:40]:
        print(" ", c)
    if len(codes) > 40:
        print(f"  ... and {len(codes) - 40} more")
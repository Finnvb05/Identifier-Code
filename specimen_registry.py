#!/usr/bin/env python3
"""specimen_registry.py -- look up specimen codes in the PaperSteel app.

Sits on top of identifier_push.py (same route, same token) and answers the one
question the measuring station needs beyond pushing a reading: does this
specimen code exist in the app?

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

from identifier_push import load_conf, specimen_exists

SERIES = re.compile(r"^(PS-[A-Z0-9]+)-(\d+)$")


def exists(code: str) -> bool:
    """Is this specimen code known to the app?

    WHY THERE IS NO list_specimens() ANY MORE
    identifier-ingest only accepts a select that names ONE specimen -- its
    select requires filter.specimen_id and answers "HTTP 400 filter.specimen_id
    (text) required" otherwise. Listing every code is deliberately not
    available, so the station cannot offer a searchable dropdown; it checks the
    code you type instead.
    """
    return specimen_exists(load_conf(), code.strip().upper())


def series_of(code: str) -> tuple[str, int] | None:
    """'PS-V4-038' -> ('PS-V4', 38). None if it is not a numbered series."""
    m = SERIES.match(code)
    return (m.group(1), int(m.group(2))) if m else None


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: specimen_registry.py PS-V4-038     (is it in the app?)")
    code = sys.argv[1]
    print(f"{code}: {'found' if exists(code) else 'NOT in specimens'}")
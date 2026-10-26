"""identifier_push.py - stuurt een meting van het identificatiestation naar de PaperSteel-app.

Startpunt voor Finn (alleen Python-standaardbibliotheek, geen pip install nodig).
Zelfde route als de Galdabini-trekbank: HTTPS POST naar de 'stark-ingest' edge function.

Gebruik vanuit jouw meetcode:
    from identifier_push import send_measurement
    send_measurement("PS-V4-038", length_mm=300.2, width_mm=20.1, thickness_mm=4.05,
                     photo_path="/home/pi/captures/PS-V4-038.jpg")

Of los testen:  python3 identifier_push.py PS-V4-038 300.2 20.1 4.05

Config: stark_ingest.conf naast dit script (NIET in Git zetten):
    INGEST_URL=https://<project>.supabase.co/functions/v1/stark-ingest
    TOKEN=<krijg je apart van Hugo>
Elke meting gaat eerst naar outbox/ op de Pi en wordt pas verwijderd als de app hem heeft.
Geen wifi? Dan blijft hij staan en gaat hij mee met de volgende verzending.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONF = HERE / "stark_ingest.conf"
OUTBOX = HERE / "outbox"
TABLE = "identifier_measurements"   # nieuwe tabel - wordt aangemaakt zodra de velden vaststaan
SOFTWARE_VERSION = "0.1"
PS_CODE = re.compile(r"^PS-[A-Z0-9]+-\d{3,}$")   # bv. PS-V4-038 (tekstcode, NIET de uuid)


def load_conf() -> dict[str, str]:
    conf = {}
    for line in CONF.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            conf[k.strip()] = v.strip()
    if not conf.get("INGEST_URL") or not conf.get("TOKEN"):
        sys.exit("stark_ingest.conf moet INGEST_URL en TOKEN bevatten")
    return conf


def ingest(conf: dict[str, str], body: dict) -> list[dict]:
    req = urllib.request.Request(
        conf["INGEST_URL"], data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {conf['TOKEN']}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:   # laat de echte reden zien, niet alleen "400"
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp.get("data", [])


def specimen_exists(conf: dict[str, str], code: str) -> bool:
    rows = ingest(conf, {"op": "select", "table": "specimens", "columns": "specimen_id",
                         "filter": {"specimen_id": code}, "limit": 1})
    return bool(rows)


def send_measurement(specimen_id: str, *, length_mm: float, width_mm: float, thickness_mm: float,
                     photo_path: str | None = None, notes: str = "") -> Path:
    """Schrijf de meting naar outbox/ en probeer meteen alles in de outbox te versturen."""
    if not PS_CODE.match(specimen_id):
        raise ValueError(f"onbekend specimen-ID formaat: {specimen_id!r} (verwacht bv. PS-V4-038)")
    row = {
        "specimen_id": specimen_id,
        "length_mm": round(length_mm, 2), "width_mm": round(width_mm, 2),
        "thickness_mm": round(thickness_mm, 3),
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "station": socket.gethostname(), "software_version": SOFTWARE_VERSION,
        "photo_path": photo_path, "notes": notes,
    }
    OUTBOX.mkdir(exist_ok=True)
    f = OUTBOX / f"{row['measured_at'].replace(':', '')}_{specimen_id}.json"
    f.write_text(json.dumps(row, indent=2), encoding="utf-8")
    flush_outbox()
    return f


def flush_outbox() -> int:
    """Verstuur alles wat nog in outbox/ staat. Geeft het aantal verzonden metingen terug.

    Geen (volledige) stark_ingest.conf is geen fout: de meting staat al veilig in
    outbox/ en gaat mee zodra het token er is. Zonder deze vangst crasht een
    testrun zonder token met een FileNotFoundError.
    """
    try:
        conf = load_conf()
    except (OSError, SystemExit) as e:
        print(f"nog niet verstuurd (geen bruikbare {CONF.name}): {e}")
        return 0
    sent = 0
    for f in sorted(OUTBOX.glob("*.json")):
        row = json.loads(f.read_text(encoding="utf-8"))
        try:
            if not specimen_exists(conf, row["specimen_id"]):
                print(f"LET OP: {row['specimen_id']} staat niet in de app - blijft in outbox")
                continue
            ingest(conf, {"op": "insert", "table": TABLE, "rows": [row]})
        except (OSError, RuntimeError) as e:   # geen wifi / app weigert: later opnieuw
            print(f"nog niet verstuurd ({f.name}): {e}")
            break
        f.unlink()
        sent += 1
    return sent


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("gebruik: python3 identifier_push.py <PS-code> <lengte_mm> <breedte_mm> <dikte_mm>")
    code, l, w, t = sys.argv[1], *map(float, sys.argv[2:])
    print("opgeslagen:", send_measurement(code, length_mm=l, width_mm=w, thickness_mm=t))
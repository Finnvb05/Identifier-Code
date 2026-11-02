"""identifier_push.py - stuurt een meting van het identificatiestation naar de PaperSteel-app.

Sinds 2 nov 2026 via de eigen, afgeschermde functie identifier-ingest: die kan
alleen specimen_id's opzoeken in specimens en rijen toevoegen aan
identifier_measurements. Niets wijzigen, niets verwijderen, en nooit meer naar
specimens schrijven.

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
TABLE = "identifier_measurements"
SOFTWARE_VERSION = "0.2"
MAX_ROWS = 50                       # identifier-ingest weigert meer per aanroep

# Precies de kolommen uit Hugo's mail van 29 oktober. De functie weigert de HELE
# aanroep bij een onbekende kolom, dus een enkel veld te veel blokkeert de hele
# outbox. id en created_at vult de database zelf.
# Een kolom erbij? Eerst naam + type naar Hugo, niet alvast versturen.
ALLOWED = (
    "specimen_id", "measured_at", "station", "software_version",
    "length_mm", "width_mm", "thickness_mm", "length_side_mm", "delta_mm",
    "length_px", "width_px", "thickness_px",
    "focal_px_0", "focal_px_1", "eff_dist_0", "eff_dist_1",
    "skew0_deg", "tilt0_deg", "tilt1_deg", "rms0_px", "rms1_px",
    "roi0", "roi1", "photo_path", "notes",
)
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


def row_for_app(row: dict) -> dict:
    """Alleen de kolommen die identifier-ingest accepteert.

    Oude outbox-bestanden kunnen velden bevatten van voor de kolomlijst. Omdat
    de functie bij een onbekende kolom de hele aanroep weigert, zou een enkel
    oud bestand de wachtrij voorgoed blokkeren. Hier filteren is dus geen
    cosmetica maar de reden dat de outbox leegloopt.
    """
    return {k: v for k, v in row.items() if k in ALLOWED}


def send_measurement(specimen_id: str, *, length_mm: float, width_mm: float, thickness_mm: float,
                     photo_path: str | None = None, notes: str = "",
                     **extra) -> Path:
    """Schrijf de meting naar outbox/ en probeer meteen alles in de outbox te versturen.

    In `extra` mogen alleen namen uit ALLOWED staan (de pixel- en
    kalibratiekolommen). Een onbekende naam is hier meteen een fout, zodat hij
    nooit stilletjes in de outbox belandt en daar de hele rij blokkeert.
    """
    unknown = sorted(set(extra) - set(ALLOWED))
    if unknown:
        raise ValueError(f"kolom(men) niet toegestaan door identifier-ingest: "
                         f"{', '.join(unknown)} - eerst naam + type naar Hugo")
    if not PS_CODE.match(specimen_id):
        raise ValueError(f"onbekend specimen-ID formaat: {specimen_id!r} (verwacht bv. PS-V4-038)")
    row = {
        "specimen_id": specimen_id,
        "length_mm": round(length_mm, 2), "width_mm": round(width_mm, 2),
        "thickness_mm": round(thickness_mm, 3),
        "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "station": socket.gethostname(), "software_version": SOFTWARE_VERSION,
        "photo_path": photo_path, "notes": notes,
        **{k: v for k, v in extra.items() if v is not None},
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
    sent, batch = 0, []

    def ship():
        """Eén aanroep voor maximaal MAX_ROWS metingen; pas daarna wissen."""
        if not batch:
            return 0
        ingest(conf, {"op": "insert", "table": TABLE,
                      "rows": [r for _, r in batch]})
        for f, _ in batch:
            f.unlink()                 # pas weg als de app hem echt heeft
        n = len(batch)
        batch.clear()
        return n

    for f in sorted(OUTBOX.glob("*.json")):
        try:
            row = json.loads(f.read_text(encoding="utf-8"))
            if not specimen_exists(conf, row["specimen_id"]):
                print(f"LET OP: {row['specimen_id']} staat niet in de app - blijft in outbox")
                continue
            batch.append((f, row_for_app(row)))
            if len(batch) >= MAX_ROWS:
                sent += ship()
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            print(f"nog niet verstuurd ({f.name}): {e}")
            break
    try:
        sent += ship()
    except (OSError, RuntimeError) as e:
        print(f"laatste portie nog niet verstuurd: {e}")
    return sent


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("gebruik: python3 identifier_push.py <PS-code> <lengte_mm> <breedte_mm> <dikte_mm>")
    code, l, w, t = sys.argv[1], *map(float, sys.argv[2:])
    print("opgeslagen:", send_measurement(code, length_mm=l, width_mm=w, thickness_mm=t))
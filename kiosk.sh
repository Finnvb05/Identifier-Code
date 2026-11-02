#!/bin/bash
# ~/kiosk.sh -- open the station's touchscreen page full screen, once the
# station is actually answering.
#
# Lives in the home directory on purpose: the repo path has spaces in it, and
# a .desktop Exec= line with spaces is a reliable source of silent failure.

URL="http://localhost:8080/touch"

# Wait for the server rather than racing it. Chromium does not retry a refused
# connection -- it shows its own error page and stays there, which looks
# exactly like the station being broken.
for _ in $(seq 1 90); do
    curl -sf -o /dev/null --max-time 2 "$URL" && break
    sleep 1
done

# After a power cut Chromium offers to restore pages, and nobody is there to
# dismiss it. Tell it the last run ended cleanly.
PREFS="$HOME/.config/chromium/Default/Preferences"
if [ -f "$PREFS" ]; then
    sed -i 's/"exit_type":"[^"]*"/"exit_type":"Normal"/; s/"exited_cleanly":false/"exited_cleanly":true/' "$PREFS"
fi

BROWSER="$(command -v chromium || command -v chromium-browser)"
[ -x "$BROWSER" ] || { echo "no chromium found" >&2; exit 1; }

exec "$BROWSER" --kiosk "$URL" \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-features=TranslateUI \
    --check-for-update-interval=31536000 \
    --password-store=basic

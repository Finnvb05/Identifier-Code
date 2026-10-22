#!/bin/bash
# Launch the gauge touch view on the touchscreen (HDMI-A-2).
# Placement is done by the labwc window rule matching --class=gauge.
# The profile is wiped each launch so the window never restores fullscreen.

PROFILE=/tmp/gaugekiosk
URL=http://localhost:8080/touch

pkill -f "user-data-dir=$PROFILE" 2>/dev/null
sleep 1
rm -rf "$PROFILE"

exec chromium \
    --app="$URL" \
    --class=gauge \
    --user-data-dir="$PROFILE" \
    --window-size=720,1560 \
    --no-first-run \
    --no-default-browser-check \
    --disable-session-crashed-bubble \
    --disable-infobars \
    --noerrdialogs \
    --disable-background-networking \
    --disable-sync \
    --disable-features=Translate,OptimizationGuideModelDownloading \
    --check-for-update-interval=31536000 \
    "$@"
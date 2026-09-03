#!/bin/bash
# Launch the gauge touch view on the touchscreen (HDMI-A-2).
#
# Placement is done by the labwc window rule in ~/.config/labwc/rc.xml, which
# matches the window title "gauge", moves it to HDMI-A-2 and then fullscreens it.
# Under Wayland the browser cannot place its own window, so the compositor has
# to do it -- --window-position is accepted and ignored.
#
# --class=gauge FORCES THE WAYLAND app_id, AND THAT IS THE POINT.
# labwc evaluates window rules when a window is first MAPPED. A rule matching on
# title is a race: Chromium sets the title only once the page has loaded, so a
# cold start (slow) matches and a warm start (fast) maps the window before the
# title exists and the rule silently does nothing. app_id is set at creation, so
# matching it is deterministic.
#
# The profile is also wiped each launch: labwc refuses to move a window that is
# already fullscreen, and a restored session can recreate one that way.

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
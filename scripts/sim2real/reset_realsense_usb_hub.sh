#!/usr/bin/env bash
# Re-enumerate (unload/reload) the USB hub the RealSense cameras hang off, at the
# usb-driver layer. This recovers cameras that have WEDGED or DROPPED off the bus
# (no /dev/bus/usb node), which librealsense hardware_reset() cannot touch.
#
# Needs root (writes to /sys/bus/usb/drivers/usb/{unbind,bind}).
# Run from the session with:   ! sudo bash scripts/sim2real/reset_realsense_usb_hub.sh
#
# Targets the parent hub(s) of any currently-enumerated RealSense, plus the known
# RealSense hub 2-1 (the external "USB3.2 Hub" all 3 cameras plug into) as a
# fallback in case every camera has fully dropped.

set -u

UNBIND=/sys/bus/usb/drivers/usb/unbind
BIND=/sys/bus/usb/drivers/usb/bind

# Which hub(s) to re-enumerate. A fully-dropped cam has no sysfs node, so we can't
# know its hub from the cam itself -> reset ALL the USB3.2 hubs the RealSense
# cameras share (this box has two: 2-1 and 2-8, Genesys Logic 05e3). Also fold in
# the parent hub of any RealSense that IS still enumerated.
declare -A HUBS=()
for d in /sys/bus/usb/devices/*/; do
    node=$(basename "$d")
    v=$(cat "$d/idVendor" 2>/dev/null || true)
    prod=$(cat "$d/product" 2>/dev/null || true)
    # USB3.2 hub (where the 5Gbps RealSense cams live) -> reset candidate.
    if [ "$v" = "05e3" ] && [ "${prod}" = "USB3.2 Hub" ]; then
        HUBS["$node"]=1
    fi
    # parent hub of a still-enumerated RealSense.
    if [ "$v" = "8086" ]; then
        hub=${node%.*}                   # strip trailing .port (2-1.3 -> 2-1)
        [ "$hub" != "$node" ] && HUBS["$hub"]=1
    fi
done
# Fallback if discovery found nothing.
[ ${#HUBS[@]} -eq 0 ] && HUBS=( ["2-1"]=1 ["2-8"]=1 )

echo "[hub-reset] RealSense before:"
lsusb | grep -iE 'realsense|8086:0b' || echo "    (none on bus)"

for hub in "${!HUBS[@]}"; do
    if [ ! -e "/sys/bus/usb/devices/$hub" ]; then
        echo "[hub-reset] hub $hub not present, skipping"
        continue
    fi
    echo "[hub-reset] unbind $hub ($(cat /sys/bus/usb/devices/$hub/product 2>/dev/null))"
    echo -n "$hub" > "$UNBIND" 2>/dev/null || { echo "    unbind FAILED (need root?)"; exit 1; }
    sleep 3
    echo "[hub-reset] bind   $hub"
    echo -n "$hub" > "$BIND" 2>/dev/null || echo "    bind FAILED"
done

echo "[hub-reset] waiting 6s for re-enumeration..."
sleep 6

echo "[hub-reset] RealSense after:"
lsusb | grep -iE 'realsense|8086:0b' || echo "    (none on bus — try a physical replug / powered-hub power cycle)"

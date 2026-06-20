"""Unload/reload (USB-reset) all connected Intel RealSense cameras.

Calls librealsense ``hardware_reset()`` on every enumerated device — this forces
a real USB disconnect + re-enumeration (the "unplug/replug" equivalent), which
clears the wedged-camera state that makes eval crash on startup. Run with NO
camera process holding the devices.

NOTE: hardware_reset only works on devices that are still enumerated. A camera
that has fully dropped off the bus (no /dev/bus/usb node) can't be reset this way
— recover those with a root USB unbind/bind (see the printed hint) or replug.

Usage:  conda run -n robodiff python scripts/sim2real/reset_realsense_usb.py
"""
import time
import pyrealsense2 as rs


def enumerate_devices():
    ctx = rs.context()
    devs = ctx.query_devices()
    out = []
    for d in devs:
        try:
            serial = d.get_info(rs.camera_info.serial_number)
        except Exception:
            serial = "?"
        try:
            name = d.get_info(rs.camera_info.name)
        except Exception:
            name = "RealSense"
        try:
            phys = d.get_info(rs.camera_info.physical_port)
        except Exception:
            phys = "?"
        out.append((d, name, serial, phys))
    return out


def main():
    before = enumerate_devices()
    print(f"[reset] {len(before)} RealSense device(s) enumerated before reset:")
    for _, name, serial, phys in before:
        print(f"    - {name}  serial={serial}  port={phys}")
    if not before:
        print("[reset] No RealSense devices enumerated — nothing to reset.")
        print("[reset] If a camera is wedged/dropped, recover at the USB layer (needs root):")
        print("        for p in /sys/bus/usb/drivers/usb/2-1.3 ...; do echo $(basename $p) | sudo tee "
              "/sys/bus/usb/drivers/usb/unbind; done; then ...echo... | sudo tee .../bind")
        return

    for d, name, serial, _ in before:
        try:
            d.hardware_reset()
            print(f"[reset] hardware_reset() sent -> {name} {serial}")
        except Exception as e:
            print(f"[reset] FAILED to reset {name} {serial}: {e}")

    # Re-enumeration takes a few seconds after a USB reset.
    print("[reset] waiting 8s for devices to re-enumerate...")
    time.sleep(8.0)

    after = enumerate_devices()
    print(f"[reset] {len(after)} RealSense device(s) enumerated after reset:")
    for _, name, serial, phys in after:
        print(f"    - {name}  serial={serial}  port={phys}")


if __name__ == "__main__":
    main()

#! /bin/sh
# Setting up CAN for the Jetson Orin Nano following NVIDIA's developer guide:
# https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/HR/ControllerAreaNetworkCan.html
# (2025-09-19)

set -eu

script_name="$0"

usage() {
  echo "This script sets up a virtual CAN interface on the Jetson Orin Nano."
  echo ""
  echo "Usage: $script_name [loopback|--help|-h]"
  echo ""
  echo "  loopback"
  echo "    If loopback is specified, the virtual interface will be set up as a loopback"
  echo "    interface, allowing you to receive the messages you send on the same"
  echo "    interface. This way you can easily check if the CAN controller is set up"
  echo "    correctly."
  echo ""
  echo "  -h, --help"
  echo "    Display this help message."
}

arg="${1:-}"

if [ "$arg" = "--help" ] || [ "$arg" = "-h" ] || [ "$#" -gt 1 ]; then
  usage
  exit 0
fi

if [ -n "$arg" ] && [ "$arg" != "loopback" ]; then
  usage
  exit 0
fi

controller_status=$(cat /proc/device-tree/mttcan\@c310000/status)
if [ "$controller_status" != "okay" ]; then
  echo "FATAL: CAN controller is not reporting okay!" >&2
  exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "FATAL: This script must be run as root so it can set up the CAN interface." >&2
  exit 1
fi

# can0_din
busybox devmem 0x0c303018 w 0x458
# can0_dout
busybox devmem 0x0c303010 w 0x400

modprobe can
modprobe can_raw
modprobe mttcan

echo "Setting up virtual CAN interface." >&2
if [ "$arg" = "loopback" ]; then
  # This will set up a loopback on the CAN interface so we can receive what we send on the same interface.
  ip link set can0 up type can bitrate 500000 loopback on
  echo "Loopback activated" >&2
else
  # TODO: Figure out the properties we want the CAN interface to have.
  ip link set can0 up type can bitrate 500000 dbitrate 1000000 berr-reporting on fd on
fi

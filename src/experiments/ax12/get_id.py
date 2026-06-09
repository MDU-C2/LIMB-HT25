from dynamixel_sdk import *
import time

ADDR_TORQUE_ENABLE      = 24
ADDR_GOAL_POSITION      = 30
ADDR_SPEED              = 32
ADDR_CURRENT_POSITION   = 36

PROTOCOL_VERSION = 1.0
DEVICENAME = '/dev/cu.usbserial-FT88YU6M'

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DXL_SPEED = 200
DXL1_START_POSITION_VALUE = 200

port = PortHandler(DEVICENAME)
packet_handler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not port.openPort():
    print("Failed to open the port!")
    exit(1)

print("Succeeded to open the port")


baud_rate = 1000000
port.setBaudRate(baud_rate)

ID = 1

# -- Enable torque
packet_handler.write1ByteTxOnly(port, ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
# --- Set speed ---
packet_handler.write2ByteTxOnly(port, ID, ADDR_SPEED, DXL_SPEED)
# -- Set start position
packet_handler.write2ByteTxOnly(port, ID, ADDR_GOAL_POSITION, DXL1_START_POSITION_VALUE) #200 startpoint
packet_handler.write2ByteTxOnly(port, ID, ADDR_GOAL_POSITION, 300) #200 startpoint
packet_handler.write2ByteTxOnly(port, ID, ADDR_GOAL_POSITION, 400) #200 startpoint
packet_handler.write2ByteTxOnly(port, ID, ADDR_GOAL_POSITION, 500) #200 startpoint



# Close port
port.closePort()
print("Port closed")

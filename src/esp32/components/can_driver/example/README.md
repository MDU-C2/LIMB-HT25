
## CAN Bus Setup


## Project Structure

```
Esp_32_CAN_BUS/
├── main/
│   ├── can_driver.c              # CAN driver implementation
│   ├── can_driver.h              # CAN driver header
│   ├── main.c                    # Current active code
│   ├── node1_master.c.example    # Master node
│   ├── node2_sensor.c.example    # Sensor node
│   ├── node3_actuator.c.example  # Actuator node
│   └── node4_monitor.c.example   # Monitor node
|   └── CMakeLists.txt
├── CMakeLists.txt
└── sdkconfig
```

---

## Hardware Requirements

### Per Node
- 1× ESP32-C3 Mini (or compatible)
- 1× WCMCU-230 CAN transceiver module
  - Important: This module has built-in 120Ω termination resistors
  - Remove the onboard resistors if the node is NOT at a bus endpoint
  - Keep the resistors only for nodes at the physical ends of the bus
- Power supply (3.3V or 5V depending on configuration)

### About WCMCU-230
The WCMCU-230 is a compact CAN transceiver module based on the SN65HVD230 chip with:
- Built-in 3.3V power supply
- Onboard 120Ω termination resistor (can be removed by desoldering)
- Compatible with 3.3V logic levels

### GPIO Configuration
- **GPIO 5** → CAN TX
- **GPIO 4** → CAN RX
- **3.3V** → Transceiver VCC
- **GND** → Common ground

---

## CAN Bus Setup

### WCMCU-230 to ESP32-C3 Connection

```
ESP32-C3              WCMCU-230 Module
┌──────────┐          ┌─────────────────┐
│          │          │                 │
│  GPIO 5  │─────────►│ CTX (TX)        │
│  GPIO 4  │◄─────────│ CRX (RX)        │
│  3.3V    │─────────►│ 3V3             │
│  GND     │─────────►│ GND             │
│          │          │                 │
└──────────┘          │  CANH ───┐      │
                      │  CANL ───┤ To   │
                      │  GND  ───┘ Bus  │
                      └─────────────────┘
                      │ [120Ω] Optional │
                      └─────────────────┘
```

### 4-Node Bus Topology

```
 WCMCU-230   WCMCU-230   WCMCU-230   WCMCU-230
   [120Ω]       [NO]        [NO]       [120Ω]
     │           │           │           │
  ┌──▼──┐     ┌──▼──┐     ┌──▼──┐     ┌──▼──┐
  │ESP#1│     │ESP#2│     │ESP#3│     │ESP#4│
  │Node1│     │Node2│     │Node3│     │Node4│
  └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
     │           │           │           │
CANH─┴───────────┴───────────┴───────────┴─── CANH
     │           │           │           │
CANL─┴───────────┴───────────┴───────────┴─── CANL
     │           │           │           │
GND──┴───────────┴───────────┴───────────┴─── GND

 ENDPOINT    MIDDLE      MIDDLE     ENDPOINT
 Keep 120Ω  Remove 120Ω Remove 120Ω Keep 120Ω

Important:
- 120Ω resistors only at bus endpoints (ESP#1 and ESP#4)
- Common ground is essential
- WCMCU-230 modules in middle positions (ESP#2, ESP#3) must have their termination resistors removed

## Quick Start

### 1. Clone or Download the Project

```bash
cd ~/esp
git clone <your-repo-url>
cd Esp_32_CAN_BUS
```

### 2. Configure ESP-IDF

```bash
idf.py set-target esp32c3
idf.py menuconfig
```

### 3. Select Node Configuration

For Node 1 (Master):
```bash
cd main
cp node1_master.c.example main.c
```

For Node 2 (Sensor):
```bash
cd main
cp node2_sensor.c.example main.c
```

And so on for Nodes 3 and 4.

### 4. Build and Flash

```bash
idf.py build
idf.py -p COMx flash monitor  # Replace COMx with your port
```

---

### 4-Node Priority System Example
- Node 1: Master (0x010) - System coordinator
- Node 2: Sensor (0x020) - Temperature & humidity
- Node 3: Actuator (0x030) - Fan & LED control
- Node 4: Monitor (0x040) - Traffic logger

## Configuration Options

```c

CanMsgFilter filter = {
    // The ID we want to accept.
    .id = 0x10,
    // Set bits of ID can differ, unset bits must match.
    // Allows accepting multiple IDs.
    .ignore_mask = 0x0F,
};

// TX=5, RX=4, 125 kbps, only accept messages based on filter
// (NULL accepts all messages).
can_init(5, 4, 125000, &filter);
```







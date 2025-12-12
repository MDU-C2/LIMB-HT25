# 🦾 BIONIC-ARM — Dual ESP32 Prosthetic Control System

Real-time embedded firmware for a bionic robotic arm distributed over two ESP32-C3 microcontrollers:

- **Upper Arm** ➜ Sensor acquisition + EMG/IMU processing + CAN transmission
- **Lower Arm** ➜ CAN reception + Local sensor acquisition + Bluetooth relay to AGX

This project is built with:

- ESP-IDF (low-level hardware drivers)
- FreeRTOS (multi-tasking real-time OS)
- CAN Bus / TWAI (inter-module communication)
- Bluetooth (data relay to main AGX system)

---

## 📁 Project Structure

```
Upper_lower_arm/
├── can/
│   ├── __init__.py
│   ├── can_interface.py          # Abstract CAN interface
│   ├── can_message_parser.py      # CAN message serialization
│   ├── can_socketcan.py           # SocketCAN implementation
│   └── test_socketcan.py
│
├── Lower_arm_reading/            # Forearm (Avant-bras)
│   ├── main/
│   │   ├── adc_emg_driver.c       # EMG acquisition (DMA ~4kHz)
│   │   ├── adc_emg_driver.h
│   │   ├── imu_driver.c           # IMU LSM6DSO32 @100Hz
│   │   ├── imu_driver.h
│   │   └── main.c                 # RX CAN + Local sensors + BLE relay
│   ├── CMakeLists.txt
│   ├── sdkconfig
│   └── build/
│
├── Upper_arm_reading/            # Upper arm (Bras)
│   ├── main/
│   │   ├── adc_emg_driver.c       # EMG acquisition (DMA ~4kHz)
│   │   ├── adc_emg_driver.h
│   │   ├── imu_driver.c           # IMU LSM6DSO32 @100Hz
│   │   ├── imu_driver.h
│   │   ├── upper_arm.c            # Angle filtering (complementary filter)
│   │   ├── upper_arm.h
│   │   └── main.c                 # Sensor reading + TX CAN
│   ├── CMakeLists.txt
│   ├── sdkconfig
│   └── build/
│
└── README.md
```

## 🚀 Build & Flash

### Requirements

- ESP-IDF v5.x (environment sourced)
- Python 3.x (optional: CAN logging utilities)
- 2 × ESP32-C3 boards (RISC-V cores)
- CAN transceiver module (e.g., MCP2551 or SN65HVD230)

### Build & Deploy

```bash
# Set target architecture
idf.py set-target esp32c3

# Build both projects
idf.py build

# Flash to board (monitor output)
idf.py flash monitor
```

### Boot Order

⚠️ **IMPORTANT:** Always boot the **Lower Arm** first before the Upper Arm!

- Lower Arm initializes the CAN bus listener
- Upper Arm connects and starts transmitting
- This avoids initialization race conditions

---

## ⚡ Hardware Configuration

### GPIO Pinout

Both ESP32-C3 boards share the same interface:

| Component    | Interface | GPIO   | Lower Arm | Upper Arm |
| ------------ | --------- | ------ | --------- | --------- |
| EMG Sensor   | ADC0      | GPIO 2 | ✅        | ✅        |
| Piezo Sensor | ADC1      | GPIO 3 | ✅        | —         |
| IMU SDA      | I2C       | GPIO 4 | ✅        | ✅        |
| IMU SCL      | I2C       | GPIO 5 | ✅        | ✅        |
| CAN RX       | TWAI      | GPIO 6 | ✅        | ✅        |
| CAN TX       | TWAI      | GPIO 7 | ✅        | ✅        |

### CAN Bus Wiring

Both boards must be connected to the **same CAN bus**:

```
ESP32-C3 (Upper Arm)          ESP32-C3 (Lower Arm)
     GPIO 6 (RX) ──────┐   ┌─ GPIO 6 (RX)
     GPIO 7 (TX) ──────┤   ├─ GPIO 7 (TX)
          GND ─────────┴───┴─ GND

         ↓ Via CAN Transceiver (MCP2551, SN65HVD230, etc.)

      CAN-H ──────────────── CAN-H
      CAN-L ──────────────── CAN-L
      GND ───────────────── GND
```

---

## 📡 Communication Protocol

### CAN Bus Configuration

- **Speed:** 500 kbps (standard automotive)
- **Frame Type:** Extended frames (29-bit IDs)
- **Update Rate:** ~50-100 Hz (depends on sensor acquisition rate)

### Message Layout

#### **Message 0x100 — EMG Data (Upper Arm → Lower Arm)**

| Byte | Type    | Description                       |
| ---- | ------- | --------------------------------- |
| 0-3  | float32 | EMG Channel 0 (muscle signal, mV) |
| 4-7  | —       | Reserved                          |

**DLC:** 4 bytes

#### **Message 0x101 — IMU Data (Upper Arm → Lower Arm)**

| Byte | Type    | Description                 |
| ---- | ------- | --------------------------- |
| 0-3  | float32 | Accelerometer X-axis (m/s²) |
| 4-7  | float32 | Accelerometer Y-axis (m/s²) |

**DLC:** 8 bytes

_(Additional IMU data (Z, gyro) to be fragmented across multiple messages if needed)_

---

## 🏗️ System Architecture

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        UPPER ARM (Bras)                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐         ┌──────────────┐                    │
│  │ EMG Driver  │         │ IMU Driver   │                    │
│  │ (4 kHz DMA) │         │ (100 Hz I2C) │                    │
│  └──────┬──────┘         └──────┬───────┘                    │
│         │                       │                            │
│         └───────────┬───────────┘                            │
│                     │                                        │
│           ┌─────────▼─────────┐                              │
│           │  Sync & Filter    │ (FreeRTOS Tasks)             │
│           │   - emg_task      │                              │
│           │   - imu_task      │                              │
│           │ - sync_send_task  │                              │
│           └─────────┬─────────┘                              │
│                     │                                        │
│              [0x100: EMG]                                    │
│              [0x101: IMU]                                    │
│                     │                                        │
│         ┌───────────▼───────────┐                            │
│         │   CAN TX (TWAI)       │                            │
│         │   500 kbps            │                            │
│         └───────────┬───────────┘                            │
└─────────────────────┼────────────────────────────────────────┘
                      │
                ~~~ CAN BUS ~~~
                      │
┌─────────────────────▼────────────────────────────────────────┐
│                   LOWER ARM (Avant-bras)                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌────────────┐  ┌───────────────┐         │
│  │ CAN RX Task  │  │ EMG Driver │  │ IMU Driver    │         │
│  │ (Priority 6) │  │ (4 kHz)    │  │ (100 Hz)      │         │
│  └──────┬───────┘  └────────┬───┘  └───────┬───────┘         │
│         │                   │              │                 │
│         │           ┌───────▼──────────────▼────┐            │
│         │           │  acquisition_task         │            │
│         │           │  (reads local sensors)    │            │
│         │           └───────┬──────────────┬────┘            │
│         │                   │              │                 │
│         └───────┬───────────┴──────────────┴───┐             │
│                 │                              │             │
│           ┌─────▼──────────────────────────────▼──┐          │
│           │  bluetooth_relay_task                 │          │
│           │  (Fuses local + Upper data)           │          │
│           └─────┬──────────────────────────────┬──┘          │
│                 │                              │             │
│         ┌───────▼────────────┬────────────────▼──┐           │
│         │    Bluetooth TX    │                   │           │
│         │    → AGX System    │                   │           │
│         └────────────────────┴───────────────────┘           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Task Priorities & Synchronization

**Upper Arm:**
| Task | Priority | Function |
|------|----------|----------|
| `emg_task` | 6 | Collects EMG data via DMA |
| `imu_task` | 3 | Reads IMU, applies complementary filter |
| `sync_send_task` | 4 | Synchronizes data, transmits via CAN |

**Lower Arm:**
| Task | Priority | Function |
|------|----------|----------|
| `can_rx_task` | 6 | High-priority: receives Upper Arm data |
| `acquisition_task` | 5 | Reads local sensors (EMG, Piezo, IMU) |
| `bluetooth_relay_task` | 4 | Fuses all data, sends via Bluetooth |

---

## 🔄 Data Flow Summary

1. **Upper Arm** acquires its own EMG & IMU data
2. **Upper Arm** transmits via CAN (0x100: EMG, 0x101: IMU)
3. **Lower Arm** receives CAN messages
4. **Lower Arm** also reads its own local sensors (EMG, Piezo, IMU)
5. **Lower Arm** fuses all data (Upper + Local) and relays via Bluetooth to AGX

---

## 🛠️ Driver Details

### EMG Driver (`adc_emg_driver.c`)

- **Interface:** ESP32 continuous ADC with DMA
- **Sampling Rate:** ~4000 Hz
- **Output:** 800-sample windows with 400-sample overlap (50% overlap)
- **Window Size:** 800 samples per packet
- **Step Size:** 400 samples (new packet every ~100 ms)

### IMU Driver (`imu_driver.c`)

- **Sensor:** LSM6DSO32 (6-axis accelerometer + gyroscope)
- **Interface:** I2C (400 kHz)
- **Output Rate:** ~100 Hz
- **Data:** Raw accelerometer (m/s²) + gyroscope (rad/s) values
- **Filtering:** Complementary filter applied at application level

---

## 📊 Performance Metrics

- **CAN Bus Latency:** < 10 ms per message
- **Total System Latency:** ~50-100 ms (EMG window + processing + relay)
- **Bluetooth Bandwidth:** Adjustable based on BLE payload size
- **Power Consumption:** ~500 mA (both boards + sensors, TBD)

---

## 🚨 Troubleshooting

### CAN Bus Issues

- **No messages received:** Ensure Lower Arm is powered on first
- **Garbled data:** Check CAN transceiver wiring and termination
- **Timeout errors:** Verify 500 kbps clock synchronization

### Sensor Issues

- **EMG reads as zero:** Check ADC calibration in `adc_emg_driver.c`
- **IMU not responding:** Verify I2C bus voltage (3.3V) and pull-up resistors
- **Missing data:** Check FreeRTOS task priorities and mutex locks

### Bluetooth Issues

- **Not connected to AGX:** Implement BLE pairing in `bluetooth_relay_task`
- **Data corruption:** Verify packet serialization format

---

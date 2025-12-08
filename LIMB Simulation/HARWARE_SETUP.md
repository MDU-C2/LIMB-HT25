=============================================================================
🦾 SENSOR-BASED EXOSKELETON CONTROL SYSTEM
Real-time Robotic Arm Motion Replication (2× IMU + 2× ESP32)
=============================================================================

Author : Paul Briand
Version : Dec 2025

This system enables **direct kinematics robotic arm control**,
where the robot joints replicate the user's movements in real-time,
using two IMU sensors installed on the **upper arm** and **forearm**.

It replaces the old **Inverse Kinematics** (target-endpoint positioning)
with a natural motion-mapping approach:
Human motion ➜ Sensor fusion ➜ Robot joint angles

---

## 🔧 HARDWARE CONFIGURATION

Components:

- 2x ESP32-C3 ZERO (Waveshare)
- 2x Adafruit LSM6DSO32 6-DoF IMU sensors (I²C)
- Right-arm robotic simulation (PyBullet)

Official ESP32-C3 ZERO documentation:
https://mischianti.org/waveshare-esp32-c3-zero-high-resolution-pinout-datasheet-and-specs/

🚨 IMPORTANT:
Each IMU is on **its own ESP32 board** → two USB cables → two COM ports

Connections (same wiring on both boards):

IMU Pin ➜ ESP32-C3 ZERO Pin

---

VIN ➜ 3.3V
GND ➜ GND
SDA ➜ GPIO4
SCL ➜ GPIO5
ADO ➜ 3.3V (pull-high to fix I²C address)

Sensor placement:

- Upper arm IMU → controls shoulder pitch/roll & yaw drift
- Forearm IMU → controls relative elbow bending

USB / COM PORTS:

- ESP32 (Arm) → COM5
- ESP32 (Forearm) → COM6

⚠️ Check ports before running:
→ Device Manager (Windows) → Update below if different:

PORT_BRAS = "COM5" meaning ARM
PORT_AVANTBRAS = "COM6" meaning FOREARM

## Personal note: I recommend soldering the components together once the wiring is complete, else you could have some serious issues by making a wire move.

## 🧠 SOFTWARE LOGIC

Python modules:
simulation_1_IMU.py → single-sensor test
simulation_2_imus.py → full system with elbow logic

Behavior:
• Live accelerometer vectors → atan2() → stable Pitch/Roll
• Gyroscope Z → integrated into Shoulder Yaw (Base rotation)
• Dynamic calibration at startup ("tare"):
When pressing L, current angles become zero reference
• Elbow motion = relative difference:
Elbow = Forearm_Angle - Arm_Angle

Safety:
• Joint limits applied (no robot over-rotation)
• Dead-zones prevent jitter around neutral
• Smoothing filter (EMA) avoids vibration

Controls:
Press L → Start/stop sensor streaming
ESC → Exit safely

---

## 📐 KINEMATIC MAPPING SUMMARY

Shoulder_Pitch = ΔPitch(Arm_IMU)
Shoulder_Roll = IntegratedGyroZ(Arm_IMU)
Elbow_Flexion = ΔPitch(Forearm_IMU) - ΔPitch(Arm_IMU)

Output angles are clamped to:
Shoulder_Z : -90° ➜ +90°
Shoulder_Y : -90° ➜ +160°
Elbow : 0° ➜ -120° (human-like)

---

## 🎮 DIAGNOSTIC INTERFACE (PyGame HUD)

Real-time visual feedback:
• Raw IMU values (debug left column)
• Robot joint angles (right column)
• Live/Pause status color indicator

---

## 🧪 DEVELOPMENT & TESTING STEPS

Step 1 — Validate a single IMU
→ Run: simulation_1_IMU.py
→ Sensor controls shoulder motion only

Step 2 — Activate full exoskeleton logic
→ Run: simulation_2_imus.py
→ Shoulder + Elbow work together

Step 3 — Tune parameters if necessary:
LISSAGE → stability (0.10 fast, 0.25 smooth)
GYRO*SENSITIVITY → base rotation speed
DIR*\* → invert motion direction if needed

---

## 🚀 ROADMAP (Next Features)

✔ Replace IK with direct kinematics (DONE)
✔ Dual-IMU mapping with dynamic calibration (DONE)
✔ Real-time motion replication (DONE)

🔜 Coming:
▸ Forearm Yaw (pronation/supination) from gyro
▸ EMG input for gripper control
▸ Sensor health monitoring (disconnection warnings)
▸ Full 7-DoF robotic arm support

---

## END OF READ-ME

## 🤖 LIMB – 7-DoF Robotic Arm Simulation

This project simulates a **7-DoF robotic arm** with a multi-jointed articulated hand
and dedicated **Inverse Kinematics (IK)** engine.
Built using **PyBullet** for physics simulation, **Pygame** for real-time control,
and **Tkinter** for sensor monitoring dashboard.

---

## 📁 1. Project Structure

LIMB3/
├── simulation.py              # Main script (PyBullet / Pygame / Tkinter logic)
├── arm_kinamatics.py          # Inverse Kinematics (IK) calculations and reachability algorithm
├── requirements.txt           # Python dependencies list
├── README.md                  # This documentation file
└── arm/                       # ROBOT ASSETS FOLDER
    ├── right_arm.urdf         # Robot description file
    ├── *.stl                  # Main arm segment meshes (e.g., bicep.stl)
    └── fullhand2/             # Detailed Hand Meshes
        └── *.stl              # Fingers, phalanges, and covers meshes

---

## 🚀 2. Setup & Launch

🔹 Requirements

- Python 3.x
- Libraries:
  - pybullet
  - pygame
  - tkinter (usually preinstalled)

🔹 Install dependencies
pip install -r requirements.txt

🔹 Run the simulation
python simulation.py

➤ Three windows will open:

1. PyBullet 3D Viewer (simulation environment)
2. Pygame Control HUD (manual control interface)
3. Tkinter Sensor Dashboard (real-time sensor feedback)

---

## 🎮 3. Control Mapping

⚠ IMPORTANT: Click inside the **Pygame window** before using commands,
otherwise controls won’t be registered.

🔧 Manual Joint Control (degrees-based)

| Joint    | Axis / Movement | Action        | Keys  |
| -------- | --------------- | ------------- | ----- |
| Shoulder | Y (Vertical)    | Up / Down     | ↑ / ↓ |
| Shoulder | X (Lateral)     | Left / Right  | ← / → |
| Shoulder | Z (Rotation)    | Azimuth       | C / V |
| Elbow    | X (Flexion)     | Close / Open  | Z / S |
| Wrist    | X (Flexion)     | Flex / Extend | Q / D |

✋ Hand & System Commands

| Command        | Key          | Description                            |
| -------------- | ------------ | -------------------------------------- |
| Speed Modifier | SHIFT / CTRL | SHIFT = fast · CTRL = slow/precise     |
| Hand Grip      | F / G        | F = Grab (if in contact) · G = Release |
| IK Mode        | H            | Enables inverse kinematics targeting   |
| Toggle HUD     | T            | Show / hide PyBullet tactical overlay  |
| Camera View    | TAB          | Switch Orbit / Shoulder view           |
| Reset          | SPACE        | Reset robot & cup to initial position  |
| Quit           | ESC          | Closes simulation                      |

---

## 🛰️ 4. Interface Interaction

🎥 PyBullet Camera Controls

| Action            | Control                                       |
| ----------------- | --------------------------------------------- |
| Rotate camera     | Ctrl + Left Click + drag                      |
| Pan (move camera) | Ctrl + Middle Click + drag                    |
| Default view      | Press **G** (after selecting PyBullet window) |

📊 Tkinter Sensor Dashboard
Displays:

- Joint **angle / torque**
- **IMU** orientation (Roll, Pitch, Yaw)
- **Gripping reachability:** GRAPPABLE / UNGRAPPABLE status (IK-based)

---

## 🧠 Notes & Tips

✔ GPU usage recommended for better simulation performance  
✔ If controls don’t work → ensure Pygame window has focus  
✔ For stability → run using terminal, not from an IDE

---

## 🚀 Ready to operate?

→ Run `simulation.py` and control the robotic arm in real time!

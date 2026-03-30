# The Canary Rover
### Autonomous Mobile Scout for Pre-Entry Hazardous Inspection

> **Mechanical Engineering Capstone Project**  
> Thapar Institute of Engineering and Technology, Patiala  

---

## Project Overview

The Canary Rover is an autonomous ground vehicle designed to perform **pre-entry hazardous inspection in underground coal mines** — eliminating the need for human entry into potentially fatal gas environments.

Named after the historical practice of using canary birds to detect toxic gases in mines, this rover serves as a modern, robotic equivalent: autonomously scouting tunnels for methane (CH₄), carbon monoxide (CO), and other noxious gases before workers enter.

### Problem Statement

Indian coal mines face critical and recurring safety hazards:

- **Undetected methane accumulation** leading to explosions
- **Toxic gas exposure** (CO, NOx) during post-blast inspections
- **High-risk manual inspection** required by regulation but lethal in practice
- **No real-time, localized gas data** in active mining zones

Recent incidents include the East Jaintia Hills disaster (Feb 2026, 27 deaths) and Putki Balihari Colliery CO leak (Dec 2025, 3 deaths).

---

## Repository Structure

```
canary-rover/
├── canary_rover_ws/                    <- ROS2 simulation workspace
│   └── src/
│       └── canary_rover_sim/
│           ├── canary_rover_sim/
│           │   ├── lidar_sim.py        <- RPLiDAR A1M8 simulation
│           │   ├── imu_sim.py          <- MPU6050 IMU simulation
│           │   └── motor_encoder_sim.py<- 4x BLDC motor + torque control
│           ├── launch/
│           │   └── full_sim.launch.py  <- launches all 3 nodes together
│           ├── package.xml
│           └── setup.py
├── rl_model/                           <- Reinforcement Learning navigation
├── cad/                                <- SolidWorks CAD models
├── electronics/                        <- PCB, circuit diagrams
├── docs/                               <- Reports, evaluation forms
├── .gitignore
└── README.md
```

---

## ROS2 Simulation

This module contains three ROS2 Humble simulation nodes that replicate the rover's physical sensor and actuator behaviour in software, allowing full input/output testing without hardware.

### System Architecture

```
                    ┌─────────────────────┐
                    │   imu_sim_node      │
                    │  (MPU6050 / BNO055) │
                    │  50 Hz              │
                    └──────────┬──────────┘
                               │ /imu/terrain_slope
                               │ /imu/data
                    ┌──────────▼──────────┐
                    │ motor_encoder_      │
                    │ sim_node            │◄── /cmd_vel
                    │ (4× BLDC, 50 Hz)   │
                    └──────────┬──────────┘
                               │ /motor/torque_feedback
                               │ /motor/encoder_data
                               │ /motor/wheel_velocity
                               │ /motor/status

                    ┌─────────────────────┐
                    │   lidar_sim_node    │
                    │  (RPLiDAR A1M8)    │──► /scan
                    │  10 Hz, 360°       │
                    └─────────────────────┘
```

### Node 1 — `lidar_sim.py` | RPLiDAR A1M8

Simulates a full 360° laser scan inside a mine tunnel environment.

**Why it works day AND night:** RPLiDAR uses active infrared laser pulses — it does not rely on ambient light. It generates its own illumination, making it fully operational in zero-visibility mine tunnels.

| Parameter | Value |
|-----------|-------|
| Angular resolution | 1° (360 rays per scan) |
| Range | 0.15 m – 12.0 m |
| Publish rate | 10 Hz |
| Sensor noise | ±1.5 cm RMS (matched to A1M8 datasheet) |
| Topic | `/scan` (`sensor_msgs/LaserScan`) |

**Simulated environment includes:**
- Static tunnel walls at realistic mine corridor widths (~1.5 m)
- Dynamic obstacle (simulated person/gas cloud) oscillating near 90°
- Rocky debris causing irregular range returns in 200–240° zone

---

### Node 2 — `imu_sim.py` | MPU6050 IMU

Simulates a 6-DOF inertial measurement unit cycling through 10 realistic mine terrain phases.

| Parameter | Value |
|-----------|-------|
| Publish rate | 50 Hz |
| Gyroscope noise | 0.005 rad/s RMS |
| Accelerometer noise | 0.02 m/s² RMS |
| Topics | `/imu/data` (`sensor_msgs/Imu`), `/imu/terrain_slope` (`std_msgs/Float32`) |

**Terrain phases simulated:**

| Phase | Pitch | Roll | Description |
|-------|-------|------|-------------|
| 1 | 0° | 0° | Flat ground — mine entry |
| 2 | 12° | 2° | Ascending slope |
| 3 | 26° | 5° | Steep incline (near design limit 25–30°) |
| 4 | 10° | -8° | Rocky bump — sharp right tilt |
| 5 | 10° | +9° | Rocky bump — sharp left tilt |
| 6 | -10° | 1° | Descending slope |
| 7 | 0° | 0° | Flat gallery section |
| 8 | 20° | 0° | Ramp climb |
| 9 | 20° | -6° | Ramp with side camber |
| 10 | 0° | 0° | Flat — inspection zone |

---

### Node 3 — `motor_encoder_sim.py` | 4× BLDC Encoded Motors

Simulates four independently driven BLDC motors with quadrature encoders. Reads terrain slope from the IMU node and **automatically adjusts torque per wheel** to compensate for inclines and rocky terrain.

| Parameter | Value |
|-----------|-------|
| Encoder resolution | 1000 ticks/rev |
| Tyre diameter | 20–30 cm (for uneven and rocky terrain) |
| Wheel radius (sim) | 100–150 mm |
| Track width | 350 mm |
| Rover mass | 12–15 kg |
| Max torque | 5.0 Nm per motor |
| Max RPM | 150 RPM |
| Control rate | 50 Hz |
| Topics | `/motor/encoder_data`, `/motor/torque_feedback`, `/motor/wheel_velocity`, `/motor/status` |

**Torque compensation model:**
- Computes normal force per wheel based on IMU slope angle
- Adds gravity compensation term for uphill driving
- Applies rocky-terrain vibration multiplier when slope > 10°
- Uses first-order motor lag dynamics (τ = 0.15 s)

---

## Getting Started

### Prerequisites

- Ubuntu 22.04 LTS
- ROS2 Humble Hawksbill
- Python 3.10+
- `numpy`

### Installation

```bash
# Clone the repository
git clone https://github.com/Kunal-Somani/canary-rover.git
cd canary-rover/canary_rover_ws

# Build the ROS2 package
source /opt/ros/humble/setup.bash
colcon build --packages-select canary_rover_sim

# Source the workspace
source install/setup.bash
```

### Running the Simulation

**Launch all 3 nodes together:**
```bash
ros2 launch canary_rover_sim full_sim.launch.py
```

**Or run individually in separate terminals:**
```bash
# Terminal 1 — IMU (start first)
ros2 run canary_rover_sim imu_sim

# Terminal 2 — LiDAR
ros2 run canary_rover_sim lidar_sim

# Terminal 3 — Motor Encoder
ros2 run canary_rover_sim motor_encoder_sim
```

### Verifying Output

```bash
# List all active topics
ros2 topic list -t

# Check LiDAR scan rate (~10 Hz)
ros2 topic hz /scan

# Watch terrain slope live
ros2 topic echo /imu/terrain_slope --field data

# Watch motor torque adapting to slope
ros2 topic echo /motor/torque_feedback

# View node graph
rqt_graph
```

### Expected Topics

```
/imu/data                [sensor_msgs/msg/Imu]
/imu/terrain_slope       [std_msgs/msg/Float32]
/motor/encoder_data      [std_msgs/msg/Int32MultiArray]
/motor/torque_feedback   [std_msgs/msg/Float32MultiArray]
/motor/wheel_velocity    [std_msgs/msg/Float32MultiArray]
/motor/status            [std_msgs/msg/String]
/scan                    [sensor_msgs/msg/LaserScan]
```

---

## Design Specifications (from Survey)

| Requirement | Specification |
|-------------|---------------|
| Climb angle | 25°–30° |
| Tyre diameter | 20–30 cm |
| Rover mass | 12–15 kg |
| Ground clearance | ~15 cm on rocky terrain |
| Communication range | >= 200 m line-of-sight |
| Surface temperature limit | < 85°C |
| Enclosure rating | IP54 or higher |
| Safe methane operation | Up to 5% vol |
| Gas detection | CH4, CO, NOx |
| Certification | IS/IEC 60079-11 (Intrinsic Safety) |

---

## Mechanical Design

Three design iterations were evaluated using a weighted decision matrix:

| Design | Score |
|--------|-------|
| Rocker-Bogie (NASA Mars Rover inspired) | 3.57 |
| Tank Climber (triangular tracked base) | 3.71 |
| **ATV 4-Wheeler with bevel gear differential** ✅ | **4.08** |

The **4-wheel all-terrain UGV with 4WD** was selected for best balance of cost, ease of implementation, certification feasibility, and terrain performance. Large diameter tyres (20–30 cm) are used specifically to handle rocky mine terrain and maintain the required ~15 cm ground clearance under a 12–15 kg rover.

---

## Regulatory Compliance

| Standard | Requirement |
|----------|-------------|
| IS/IEC 60079-11 | Intrinsically safe electrical design |
| IS/IEC 60079-1 | Flameproof enclosure for non-IS components |
| OSH Code 2020, Section 6 | Workplace free from foreseeable hazards |
| DGMS / Coal Mines Regulations | Mine safety authority compliance |

---

## License

Licensed under the [Apache License 2.0](LICENSE).

---

## Contributors

### **Kunal**
GitHub: [Kunal-Somani](https://github.com/Kunal-Somani)  
LinkedIn: [kunal-somani-227373344](https://www.linkedin.com/in/kunal-somani-227373344)

### **Ujjwal**
GitHub: [ujjwx1](https://github.com/ujjwx1)  

### **Chhavi**
GitHub: [Chhavi223](https://github.com/Chhavi223) 

---



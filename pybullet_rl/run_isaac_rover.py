
from omni.isaac.kit import SimulationApp

# =========================
# START ISAAC
# =========================
simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting"
})

# =========================
# IMPORTS
# =========================
import numpy as np
import time
from stable_baselines3 import PPO

import omni.kit.app

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.core.utils.viewports import set_camera_view

from pxr import UsdLux

# =========================
# 🔴 PATHS (EDIT THESE)
# =========================
MODEL_PATH = "/home/pradyumnasingh/Desktop/Capstone/rover_refined_final_v2.zip"
URDF_PATH  = "/home/pradyumnasingh/Desktop/Capstone/vleg_rover.urdf"

# =========================
# LOAD MODEL
# =========================
model = PPO.load(
    MODEL_PATH,
    device="cpu",
    custom_objects={
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": 0.2
    }
)
model.policy.eval()

# =========================
# WORLD
# =========================
world = World(stage_units_in_meters=1.0)
stage = world.stage

# =========================
# LIGHTING
# =========================
light = UsdLux.DistantLight.Define(stage, "/World/Light")
light.CreateIntensityAttr(3000)
light.CreateAngleAttr(0.5)

# =========================
# TERRAIN
# =========================

def ground(x_start, x_end, width):
    length = x_end - x_start
    center = x_start + length / 2
    create_prim(
        prim_path=f"/World/ground_{int(x_start)}",
        prim_type="Cube",
        position=np.array([center, 0, 0.0]),
        scale=np.array([length, width * 2, 0.2]),
    )

def wall(x_start, x_end, width):
    length = x_end - x_start
    center = x_start + length / 2

    for side, label in [(-1, "neg"), (1, "pos")]:
        create_prim(
            prim_path=f"/World/wall_{int(x_start)}_{label}",
            prim_type="Cube",
            position=np.array([center, side * (width + 0.2), 0.8]),
            scale=np.array([length, 0.2, 1.5]),
        )

def bumps(x_start, x_end, width):
    xs = np.linspace(x_start, x_end, 30)

    for i, x in enumerate(xs):
        h = 0.05 * np.sin(i * 0.6)

        create_prim(
            prim_path=f"/World/bump_{int(x_start)}_{i}",
            prim_type="Cube",
            position=np.array([x, 0, h]),
            scale=np.array([0.8, width * 2, 0.1]),
        )

# build track
ground(0, 25, 1.0)
ground(25, 50, 1.0)
bumps(25, 50, 1.0)

ground(50, 75, 2.0)
wall(50, 75, 2.0)

ground(75, 100, 1.6)
wall(75, 100, 1.6)
bumps(75, 100, 1.6)

ground(100, 125, 1.0)
wall(100, 125, 1.0)
bumps(100, 125, 1.0)

# =========================
# ENABLE URDF IMPORTER
# =========================
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

time.sleep(1)

# =========================
# IMPORT ROVER (CORRECT API)
# =========================
from isaacsim.asset.importer.urdf import _urdf
import os


urdf_interface = _urdf.acquire_urdf_interface()

import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = True

robot_desc=_urdf.UrdfRobot()
import_config.fix_base = False

# split path correctly
asset_root = os.path.dirname(URDF_PATH)
asset_name = os.path.basename(URDF_PATH)

urdf_interface.import_robot(
    asset_root,
    asset_name,
    robot_desc,
    import_config,
    "/World/Rover"
)
# =========================
# INIT WORLD
# =========================
world.reset()

robot = SingleArticulation("/World/Rover")
world.scene.add(robot)

# =========================
# CAMERA
# =========================

def update_camera():
    pos, _ = robot.get_world_pose()

    cam_pos = [
        pos[0] - 4.0,
        pos[1],
        pos[2] + 2.0
    ]

    target = [
        pos[0],
        pos[1],
        pos[2] + 0.5
    ]

    set_camera_view(cam_pos, target)

# =========================
# OBS
# =========================

def get_obs():
    pos, rot = robot.get_world_pose()
    vel = robot.get_linear_velocity()
    ang = robot.get_angular_velocity()

    roll, pitch, yaw = quat_to_euler_angles(rot)

    return np.array([
        pos[0], pos[1], pos[2],
        roll, pitch,
        vel[0], vel[1],
        ang[2]
    ], dtype=np.float32)

# =========================
# ACTION
# =========================

def apply_action(action):
    left, right = action

    for j in robot.joint_names:
        if "left" in j:
            robot.set_joint_efforts({j: float(left * 40)})
        elif "right" in j:
            robot.set_joint_efforts({j: float(right * 40)})

# =========================
# RUN LOOP
# =========================

print("🚀 Simulation started")

while simulation_app.is_running():

    obs = get_obs()
    action, _ = model.predict(obs, deterministic=True)

    apply_action(action)

    update_camera()

    world.step(render=True)

simulation_app.close()


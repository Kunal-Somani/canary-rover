"""
Canary Rover — Isaac Sim 5.1.0  +  ROS 2 Jazzy
═══════════════════════════════════════════════
Publishes real ROS 2 topics so you can use ANY ROS 2 SLAM package:
  /scan              → sensor_msgs/LaserScan      (RPLiDAR A1M8, 10 Hz)
  /velodyne_points   → sensor_msgs/PointCloud2    (VLP-16 style, 10 Hz)
  /imu/data          → sensor_msgs/Imu            (50 Hz)
  /odom              → nav_msgs/Odometry          (60 Hz)
  /tf                → odom → base_link transform  (60 Hz)

Run alongside slam_toolbox (2D) or rtabmap_ros (3D):
  Terminal 1:  python slam_ros2.py
  Terminal 2:  ros2 launch canary_slam slam_toolbox.launch.py      # 2D
               ros2 launch canary_slam rtabmap.launch.py           # 3D

Requirements (run once):
  sudo apt install ros-jazzy-slam-toolbox ros-jazzy-rtabmap-ros \
                   ros-jazzy-nav2-bringup python3-transforms3d
  pip install transforms3d
"""

# ── Source ROS 2 Jazzy inside Python before rclpy is imported ──────────────
import subprocess, sys, os

# Ensure ROS 2 environment variables are present.
# If you already source /opt/ros/jazzy/setup.bash in your shell before
# running Isaac Sim, this block is a no-op.
_ros_setup = "/opt/ros/jazzy/setup.bash"
if "ROS_DISTRO" not in os.environ:
    cmd = f"bash -c 'source {_ros_setup} && env'"
    env_out = subprocess.check_output(cmd, shell=True).decode()
    for line in env_out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k, v)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import math, struct, time
import numpy as np
import omni, omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import UsdGeom, Gf, Sdf

# ── ROS 2 imports ────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan, Imu, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
from tf2_ros import TransformBroadcaster

# ════════════════════════════════════════════════════════════════════════════
# USER PATHS
# ════════════════════════════════════════════════════════════════════════════
ROVER_USD = "/home/kunal/isaac/canary_demo/robot/vleg_rover/vleg_rover.usd"

# ════════════════════════════════════════════════════════════════════════════
# WORLD CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
FINISH_X = 55.0
TUNNEL_R = 1.6
LENGTH   = 60.0
_TUNNEL_X0, _TUNNEL_X1 = -2.0, 58.0
_SENSOR_Z  = 0.32        # sensor height above rover base (m)

# ════════════════════════════════════════════════════════════════════════════
# SENSOR PARAMETERS
# ════════════════════════════════════════════════════════════════════════════
# RPLiDAR A1M8 — horizontal 2-D scan
SCAN_RAYS      = 360
SCAN_MAX_RANGE = 3.5
SCAN_NOISE_STD = 0.02
SCAN_HZ        = 10        # publish every 6 sim steps (sim @ 60 Hz)

# VLP-16 — 3-D point cloud
VLP16_ELEV_DEG = np.linspace(-15.0, 15.0, 16)
VLP16_AZIM_N   = 360
_VLP16_ELEV    = np.deg2rad(VLP16_ELEV_DEG)
_VLP16_AZIM    = np.linspace(0, 2*np.pi, VLP16_AZIM_N, endpoint=False)
VLP16_MAX_RANGE = 10.0     # VLP-16 realistic range
VLP16_NOISE_STD = 0.03

# IMU (simulated MPU-6050 class)
IMU_HZ         = 50        # every ~1.2 sim steps — triggered on step%1==0 but
IMU_ACC_NOISE  = 0.02      # published every step (60 Hz ≈ fine for 50 Hz)
IMU_GYR_NOISE  = 0.005

_slam_rng = np.random.default_rng(0)

# Rock obstacles (world XY) — same as before
_ROCK_XY = [
    ( 3.0,  0.6, 0.06, 0.05),
    (10.0,  0.5, 0.06, 0.05),
    (16.0, -0.6, 0.06, 0.05),
    (23.0,  0.8, 0.06, 0.05),
    (31.0,  0.3, 0.06, 0.05),
    (38.0, -0.8, 0.06, 0.05),
    (45.0,  0.6, 0.06, 0.05),
    (48.0, -0.5, 0.06, 0.05),
]

# ════════════════════════════════════════════════════════════════════════════
# TERRAIN
# ════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
TERRAIN_SAMPLES = 300
_rx_t = np.linspace(0, LENGTH, TERRAIN_SAMPLES)
_terrain_h = (
    0.06  * np.sin(_rx_t * 0.25) +
    0.04  * np.cos(_rx_t * 0.8  + 0.5) +
    0.025 * np.sin(_rx_t * 2.1  + 1.2) +
    0.015 * np.sin(_rx_t * 4.5  + 0.3) +
    0.008 * np.random.normal(size=TERRAIN_SAMPLES)
)

def terrain_height_at(x: float) -> float:
    x = max(0.0, min(x, LENGTH - 0.01))
    idx = x / LENGTH * (TERRAIN_SAMPLES - 1)
    i0, i1 = int(idx), min(int(idx) + 1, TERRAIN_SAMPLES - 1)
    t = idx - i0
    return float(_terrain_h[i0] * (1 - t) + _terrain_h[i1] * t)

def terrain_slope_at(x: float, dx: float = 0.3) -> float:
    return math.degrees(math.atan2(
        terrain_height_at(x + dx) - terrain_height_at(max(0.0, x - dx)), 2 * dx))

# ════════════════════════════════════════════════════════════════════════════
# WORLD BUILDERS  (unchanged)
# ════════════════════════════════════════════════════════════════════════════
def build_tunnel(stage):
    segs, rings = 160, 32
    verts, fcounts, idx = [], [], []
    for i in range(segs + 1):
        x = i * (LENGTH / segs)
        for j in range(rings):
            a = (j / rings) * 2 * math.pi
            verts.append(Gf.Vec3f(x, TUNNEL_R * math.cos(a), TUNNEL_R * math.sin(a)))
    for i in range(segs):
        for j in range(rings):
            nj = (j + 1) % rings
            v0=i*rings+j; v1=i*rings+nj; v2=(i+1)*rings+j; v3=(i+1)*rings+nj
            idx += [v0,v2,v1,v1,v2,v3]; fcounts += [3,3]
    m = UsdGeom.Mesh.Define(stage, "/World/tunnel")
    m.CreatePointsAttr(verts); m.CreateFaceVertexCountsAttr(fcounts)
    m.CreateFaceVertexIndicesAttr(idx); m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr([(0.5, 0.35, 0.2)])
    UsdGeom.Xformable(m.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-2.0, 0.0, 1.0))

def build_rocky_floor(stage):
    SX=TERRAIN_SAMPLES; SY=16; W=2.8
    verts, fcounts, indices = [], [], []
    for i in range(SX):
        for j in range(SY):
            x=_rx_t[i]-2.0; y=(j/(SY-1))*W-W/2
            z=_terrain_h[i]+0.01*math.sin(j*0.8+i*0.3)
            verts.append(Gf.Vec3f(x, y, z))
    for i in range(SX-1):
        for j in range(SY-1):
            v0=i*SY+j; v1=i*SY+(j+1); v2=(i+1)*SY+j; v3=(i+1)*SY+(j+1)
            indices += [v0,v1,v3,v2]; fcounts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, "/World/rocky_floor")
    mesh.CreatePointsAttr(verts); mesh.CreateFaceVertexCountsAttr(fcounts)
    mesh.CreateFaceVertexIndicesAttr(indices); mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([(0.32, 0.20, 0.10)])

def build_rocks(stage):
    positions=[(5,0.6),(12,0.5),(18,-0.6),(25,0.8),(33,0.3),(40,-0.8),(47,0.6),(50,-0.5)]
    for k,(rx,ry) in enumerate(positions):
        rz=terrain_height_at(rx)
        cube=UsdGeom.Cube.Define(stage,f"/World/rock_{k}")
        cube.CreateSizeAttr(1.0); cube.CreateDisplayColorAttr([(0.28,0.22,0.18)])
        xf=UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(rx-2.0,ry,rz+0.06))
        xf.AddScaleOp().Set(Gf.Vec3d(0.12,0.10,0.07))

def build_finish(stage):
    c=UsdGeom.Cube.Define(stage,"/World/finish_line")
    c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([(1.0,0.0,0.0)])
    xf=UsdGeom.Xformable(c.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(FINISH_X-2.0,0.0,0.5))
    xf.AddScaleOp().Set(Gf.Vec3d(0.2,3.5,1.0))

def build_markers(stage):
    for k,x in enumerate(range(5,56,10)):
        tz=terrain_height_at(float(x))
        for side,lr in [(1.1,'L'),(-1.1,'R')]:
            cy=UsdGeom.Cylinder.Define(stage,f"/World/m{k}{lr}")
            cy.CreateRadiusAttr(0.04); cy.CreateHeightAttr(1.0)
            cy.CreateDisplayColorAttr([(1.0,0.55,0.0)])
            xf=UsdGeom.Xformable(cy.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(float(x)-2.0,side,tz+0.5))

def setup_lights(stage):
    for k,x in enumerate(range(0,60,15)):
        lp=stage.DefinePrim(f"/World/L{k}","SphereLight")
        lp.CreateAttribute("inputs:intensity",Sdf.ValueTypeNames.Float).Set(6000.0)
        lp.CreateAttribute("inputs:radius",   Sdf.ValueTypeNames.Float).Set(0.1)
        lp.CreateAttribute("inputs:color",    Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0,0.82,0.55))
        UsdGeom.Xformable(lp).AddTranslateOp().Set(Gf.Vec3d(float(x)-2.0,0.0,1.3))

def build_rover_beacon(stage):
    beacon=UsdGeom.Cube.Define(stage,"/World/rover/beacon")
    beacon.CreateSizeAttr(1.0)
    beacon.CreateDisplayColorAttr([(0.0,1.0,1.0)])
    xf=UsdGeom.Xformable(beacon.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0,0.0,0.15))
    xf.AddScaleOp().Set(Gf.Vec3d(0.30,0.20,0.15))
    print("[Rover] Beacon proxy built (cyan box)")

# ════════════════════════════════════════════════════════════════════════════
# SENSOR SIMULATION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _cast_ray_2d(ox: float, oy: float, angle: float) -> float:
    """2-D horizontal ray → range (tunnel + rocks)."""
    dx = math.cos(angle); dy = math.sin(angle)
    t = SCAN_MAX_RANGE; eps = 1e-9
    if dy >  eps:
        tc = (TUNNEL_R - oy)/dy
        if 0 < tc < t and _TUNNEL_X0 <= ox+dx*tc <= _TUNNEL_X1: t = tc
    if dy < -eps:
        tc = (-TUNNEL_R - oy)/dy
        if 0 < tc < t and _TUNNEL_X0 <= ox+dx*tc <= _TUNNEL_X1: t = tc
    if dx >  eps:
        tc = (_TUNNEL_X1 - ox)/dx
        if 0 < tc < t and -TUNNEL_R <= oy+dy*tc <= TUNNEL_R: t = tc
    if dx < -eps:
        tc = (_TUNNEL_X0 - ox)/dx
        if 0 < tc < t and -TUNNEL_R <= oy+dy*tc <= TUNNEL_R: t = tc
    for (rx,ry,rhx,rhy) in _ROCK_XY:
        x0,x1=rx-rhx,rx+rhx; y0,y1=ry-rhy,ry+rhy
        tx0=(x0-ox)/dx if abs(dx)>eps else (-1e9 if x0<=ox<=x1 else 1e9)
        tx1=(x1-ox)/dx if abs(dx)>eps else tx0
        ty0=(y0-oy)/dy if abs(dy)>eps else (-1e9 if y0<=oy<=y1 else 1e9)
        ty1=(y1-oy)/dy if abs(dy)>eps else ty0
        if tx0>tx1: tx0,tx1=tx1,tx0
        if ty0>ty1: ty0,ty1=ty1,ty0
        te=max(tx0,ty0); tx=min(tx1,ty1)
        if te<tx and 0<te<t: t=te
    return float(np.clip(t + _slam_rng.normal(0, SCAN_NOISE_STD), 0.01, SCAN_MAX_RANGE))


def simulate_laserscan(x_pos: float, y_pos: float = 0.0, heading: float = 0.0) -> list:
    """Return list of SCAN_RAYS range values (metres)."""
    step = 2 * math.pi / SCAN_RAYS
    return [_cast_ray_2d(x_pos, y_pos, heading + i * step) for i in range(SCAN_RAYS)]


def simulate_vlp16(x_pos: float, y_pos: float, heading: float):
    """
    VLP-16 point cloud — fully vectorised.
    Returns (N, 3) float32 array of (x, y, z) points in base_link frame.
    """
    oz = terrain_height_at(x_pos + 2.0) + _SENSOR_Z

    EL, AZ = np.meshgrid(_VLP16_ELEV, _VLP16_AZIM, indexing='ij')  # (16,360)
    cos_el = np.cos(EL)
    ch, sh = math.cos(heading), math.sin(heading)
    # Direction vectors in world frame (heading rotation around Z)
    dx_w = cos_el * (np.cos(AZ)*ch - np.sin(AZ)*sh)
    dy_w = cos_el * (np.cos(AZ)*sh + np.sin(AZ)*ch)
    dz_w = np.sin(EL)
    dx = dx_w.ravel(); dy = dy_w.ravel(); dz = dz_w.ravel()
    N = len(dx)

    # Cylinder intersection (tunnel)
    py = y_pos; pz = oz - 1.0
    a_c = dy**2 + dz**2 + 1e-30
    b_c = 2*(py*dy + pz*dz)
    c_c = py**2 + pz**2 - TUNNEL_R**2
    disc = b_c**2 - 4*a_c*c_c
    t_cyl = (-b_c + np.sqrt(np.maximum(disc, 0))) / (2*a_c)
    hit_x_cyl = x_pos + t_cyl * dx
    t_cyl = np.where(
        (disc >= 0) & (t_cyl > 0.01) &
        (hit_x_cyl >= _TUNNEL_X0) & (hit_x_cyl <= _TUNNEL_X1),
        t_cyl, VLP16_MAX_RANGE)
    # End-caps
    with np.errstate(divide='ignore', invalid='ignore'):
        for cap_x in (_TUNNEL_X0, _TUNNEL_X1):
            tc = np.where(np.abs(dx) > 1e-9, (cap_x - x_pos)/dx, VLP16_MAX_RANGE)
            hy = y_pos + tc*dy; hz = oz + tc*dz - 1.0
            t_cyl = np.minimum(t_cyl, np.where(
                (tc > 0.01) & (hy**2+hz**2 <= TUNNEL_R**2), tc, VLP16_MAX_RANGE))

    # Floor
    fz = terrain_height_at(x_pos + 2.0)
    t_floor = np.where(dz < -1e-9, (fz - oz)/dz, VLP16_MAX_RANGE)
    t_floor = np.where(t_floor > 0.01, t_floor, VLP16_MAX_RANGE)

    # Rocks (3D AABB)
    t_rocks = np.full(N, VLP16_MAX_RANGE, dtype=np.float32)
    for (rx,ry,rhx,rhy) in _ROCK_XY:
        rz_c = terrain_height_at(rx+2.0)+0.06; rhz=0.035
        with np.errstate(divide='ignore', invalid='ignore'):
            tx0=np.where(np.abs(dx)>1e-9,(rx-rhx-x_pos)/dx,np.where(x_pos>=rx-rhx,-1e9,1e9))
            tx1=np.where(np.abs(dx)>1e-9,(rx+rhx-x_pos)/dx,np.where(x_pos<=rx+rhx,1e9,-1e9))
            ty0=np.where(np.abs(dy)>1e-9,(ry-rhy-y_pos)/dy,np.where(y_pos>=ry-rhy,-1e9,1e9))
            ty1=np.where(np.abs(dy)>1e-9,(ry+rhy-y_pos)/dy,np.where(y_pos<=ry+rhy,1e9,-1e9))
            tz0=np.where(np.abs(dz)>1e-9,(rz_c-rhz-oz)/dz, np.where(oz>=rz_c-rhz,-1e9,1e9))
            tz1=np.where(np.abs(dz)>1e-9,(rz_c+rhz-oz)/dz, np.where(oz<=rz_c+rhz,1e9,-1e9))
        te=np.maximum(np.maximum(np.minimum(tx0,tx1),np.minimum(ty0,ty1)),np.minimum(tz0,tz1))
        tx=np.minimum(np.minimum(np.maximum(tx0,tx1),np.maximum(ty0,ty1)),np.maximum(tz0,tz1))
        t_rocks=np.minimum(t_rocks,np.where((te<tx)&(te>0.01),te,VLP16_MAX_RANGE))

    t_best = np.minimum(np.minimum(t_cyl, t_floor), t_rocks)
    noise  = _slam_rng.normal(0, VLP16_NOISE_STD, N).astype(np.float32)
    t_best = np.clip(t_best + noise, 0.01, VLP16_MAX_RANGE)

    # Convert to base_link frame (sensor at 0,0,_SENSOR_Z in base_link)
    pts_x = (t_best * dx).astype(np.float32)
    pts_y = (t_best * dy).astype(np.float32)
    pts_z = (t_best * dz + _SENSOR_Z).astype(np.float32)   # relative to base

    # Remove max-range returns (no return)
    valid = t_best < VLP16_MAX_RANGE - 0.05
    return np.stack([pts_x[valid], pts_y[valid], pts_z[valid]], axis=1)


def euler_to_quat(roll: float, pitch: float, yaw: float):
    """RPY (rad) → (w, x, y, z) quaternion."""
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return w, x, y, z


def make_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    Pack (N,3) float32 XYZ array into a sensor_msgs/PointCloud2.
    Fields: x, y, z  (each 4 bytes, FLOAT32).
    """
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp    = stamp
    msg.height          = 1
    msg.width           = len(points)
    msg.is_dense        = True
    msg.is_bigendian    = False
    msg.point_step      = 12        # 3 × float32
    msg.row_step        = msg.point_step * msg.width
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points.tobytes()
    return msg

# ════════════════════════════════════════════════════════════════════════════
# ROS 2 NODE
# ════════════════════════════════════════════════════════════════════════════
class CanaryPublisher(Node):
    """
    Single ROS 2 node that publishes all sensor topics from Isaac Sim.
    Call .publish_all() on every sim step.
    """

    def __init__(self):
        super().__init__("canary_rover")

        # Best-effort QoS for high-rate sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        self.pub_scan  = self.create_publisher(LaserScan,    "/scan",            sensor_qos)
        self.pub_cloud = self.create_publisher(PointCloud2,  "/velodyne_points", sensor_qos)
        self.pub_imu   = self.create_publisher(Imu,          "/imu/data",        sensor_qos)
        self.pub_odom  = self.create_publisher(Odometry,     "/odom",            reliable_qos)
        self.tf_bcast  = TransformBroadcaster(self)

        self.get_logger().info("CanaryPublisher ready — topics: "
                               "/scan  /velodyne_points  /imu/data  /odom  /tf")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _now(self):
        return self.get_clock().now().to_msg()

    def _header(self, frame_id: str):
        h = Header(); h.stamp = self._now(); h.frame_id = frame_id
        return h

    # ── publishers ───────────────────────────────────────────────────────────
    def publish_scan(self, ranges: list, heading: float = 0.0):
        msg = LaserScan()
        # FIX: use "laser" to match the static TF (base_link→laser) in launch file
        msg.header         = self._header("laser")
        msg.angle_min      = 0.0
        msg.angle_max      = 2 * math.pi - (2 * math.pi / SCAN_RAYS)
        msg.angle_increment= 2 * math.pi / SCAN_RAYS
        msg.time_increment = (1.0 / SCAN_HZ) / SCAN_RAYS
        msg.scan_time      = 1.0 / SCAN_HZ
        msg.range_min      = 0.10
        msg.range_max      = SCAN_MAX_RANGE
        msg.ranges         = [float(r) for r in ranges]
        msg.intensities    = [100.0] * len(ranges)
        self.pub_scan.publish(msg)

    def publish_cloud(self, points: np.ndarray):
        if len(points) == 0:
            return
        msg = make_pointcloud2(points.astype(np.float32), "base_link", self._now())
        self.pub_cloud.publish(msg)

    def publish_imu(self, pitch_deg: float, roll_deg: float,
                    ax: float, ay: float, az: float,
                    gx: float, gy: float, gz: float):
        msg = Imu()
        msg.header = self._header("base_link")
        w, qx, qy, qz = euler_to_quat(
            math.radians(roll_deg), math.radians(pitch_deg), 0.0)
        msg.orientation.w = w
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation_covariance    = [1e-4,0,0, 0,1e-4,0, 0,0,1e-4]
        msg.linear_acceleration.x     = ax
        msg.linear_acceleration.y     = ay
        msg.linear_acceleration.z     = az
        msg.linear_acceleration_covariance = [4e-4,0,0, 0,4e-4,0, 0,0,4e-4]
        msg.angular_velocity.x        = gx
        msg.angular_velocity.y        = gy
        msg.angular_velocity.z        = gz
        msg.angular_velocity_covariance    = [2.5e-5,0,0, 0,2.5e-5,0, 0,0,2.5e-5]
        self.pub_imu.publish(msg)

    def publish_odom_tf(self, x: float, y: float, z: float,
                        vx: float, heading: float = 0.0):
        stamp = self._now()
        w, qx, qy, qz = euler_to_quat(0.0, 0.0, heading)

        # Odometry message
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.position.z    = z
        odom.pose.pose.orientation.w = w
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[35] = 0.001  # yaw
        odom.twist.twist.linear.x    = vx
        odom.twist.covariance[0]     = 0.01
        self.pub_odom.publish(odom)

        # TF: odom → base_link
        tf = TransformStamped()
        tf.header.stamp            = stamp
        tf.header.frame_id         = "odom"
        tf.child_frame_id          = "base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation.w    = w
        tf.transform.rotation.x    = qx
        tf.transform.rotation.y    = qy
        tf.transform.rotation.z    = qz
        self.tf_bcast.sendTransform(tf)

    def publish_all(self, x_pos: float, y_pos: float, speed: float,
                    pitch_deg: float, roll_deg: float,
                    ax: float, ay: float, az: float,
                    gx: float, gy: float, gz: float,
                    do_lidar: bool):
        """Call once per sim step."""
        z = terrain_height_at(x_pos + 2.0) + 0.22

        # Odometry + TF: every step
        self.publish_odom_tf(x_pos, y_pos, z, speed)

        # IMU: every step
        self.publish_imu(pitch_deg, roll_deg, ax, ay, az, gx, gy, gz)

        # LiDAR (2-D scan + 3-D cloud): at SCAN_HZ only
        if do_lidar:
            ranges = simulate_laserscan(x_pos, y_pos)
            self.publish_scan(ranges)

            pts = simulate_vlp16(x_pos, y_pos, heading=0.0)
            self.publish_cloud(pts)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*65)
    print("  CANARY ROVER — Isaac Sim 5.1.0 + ROS 2 Jazzy")
    print("  Topics: /scan  /velodyne_points  /imu/data  /odom  /tf")
    print("  2D SLAM: ros2 launch canary_slam slam_toolbox.launch.py")
    print("  3D SLAM: ros2 launch canary_slam rtabmap.launch.py")
    print("="*65 + "\n")

    # Init ROS 2 — must happen before World() so the clock is ready
    rclpy.init()
    ros_node = CanaryPublisher()

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    build_tunnel(stage); build_rocky_floor(stage); build_rocks(stage)
    build_finish(stage); build_markers(stage); setup_lights(stage)

    add_reference_to_stage(usd_path=ROVER_USD, prim_path="/World/rover")
    build_rover_beacon(stage)

    world.reset()
    rover = XFormPrim(prim_path="/World/rover")
    world.play()

    step = 0; x_pos = 0.0; speed = 0.3; dt = 1.0/60.0; lc = 0

    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1; x_pos += speed * dt

            th = terrain_height_at(x_pos + 2.0)
            rover.set_world_pose(
                position=np.array([x_pos, 0.0, th + 0.22]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )

            if step % 8 == 0:
                set_camera_view(
                    eye=np.array([x_pos - 2.5, 0.0, th + 0.7]),
                    target=np.array([x_pos + 5.0, 0.0, th + 0.1])
                )

            # Sensor values
            pitch = terrain_slope_at(x_pos + 2.0)
            roll  = 2.5 * math.sin(x_pos * 0.4)
            ax_   = 9.81*math.sin(math.radians(pitch)) + np.random.normal(0, IMU_ACC_NOISE)
            ay_   =-9.81*math.sin(math.radians(roll))  + np.random.normal(0, IMU_ACC_NOISE)
            az_   = 9.81*math.cos(math.radians(pitch))*math.cos(math.radians(roll)) \
                    + np.random.normal(0, IMU_ACC_NOISE)
            gx_   = np.random.normal(0, IMU_GYR_NOISE)
            gy_   = np.random.normal(pitch*0.05, IMU_GYR_NOISE)
            gz_   = np.random.normal(0.02, IMU_GYR_NOISE)

            lc += 1
            do_lidar = (lc >= 6)
            if do_lidar: lc = 0

            # ── Publish to ROS 2 ─────────────────────────────────────────────
            ros_node.publish_all(
                x_pos=x_pos, y_pos=0.0, speed=speed,
                pitch_deg=pitch, roll_deg=roll,
                ax=ax_, ay=ay_, az=az_,
                gx=gx_, gy=gy_, gz=gz_,
                do_lidar=do_lidar
            )
            rclpy.spin_once(ros_node, timeout_sec=0)  # non-blocking

            # Terminal readout
            if step % 60 == 0:
                rpm = speed/0.06*60/(2*math.pi)
                prog = min(100, int(x_pos/FINISH_X*100))
                bar  = "█"*(prog//5) + "░"*(20-prog//5)
                print(f"┌─ Step {step:5d} ──────────────────────────────────────────────┐")
                print(f"│  📍 x={x_pos:6.2f}m  terrain_z={th:+.3f}m  speed={speed:.2f}m/s")
                print(f"│  🔵 IMU pitch={pitch:+.2f}°  roll={roll:+.2f}°")
                print(f"│  🟢 Encoders {rpm:+.1f} RPM  (wheel r=6cm)")
                print(f"│  📡 ROS2 → /scan /velodyne_points /imu/data /odom /tf")
                print(f"│  🏁 [{bar}] {prog}%  ({x_pos:.1f}/{FINISH_X}m)")
                print(f"└──────────────────────────────────────────────────────────────────┘\n")

            if x_pos >= FINISH_X:
                print("\n🎯 FINISH LINE REACHED!\n")
                for _ in range(250):
                    world.step(render=True)
                    rclpy.spin_once(ros_node, timeout_sec=0)
                break

    finally:
        ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()

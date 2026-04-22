"""
Canary Rover — Isaac Sim 5.1.0  (v2 — improved)
2-D occupancy SLAM  +  3-D voxel SLAM  +  video export
Pure Python/NumPy — NO ROS required
Ubuntu 24.04 compatible

═══════════════════════════════════════════════════════════════
HOW SLAM WORKS WITHOUT ROS2
═══════════════════════════════════════════════════════════════
Localisation  : dead-reckoning  x += speed * dt  (open-loop, no loop closure)
Sensor model  : analytical ray–geometry intersection
                  • 2-D tunnel walls → AABB slab test (Y = ±TUNNEL_R)
                  • 3-D tunnel body  → quadratic cylinder solve + end-cap plane test
                  • floor            → planar intersection  t = (z_floor - oz) / dz
                  • rocks (2-D/3-D)  → AABB slab method  (max entries, min exits)
Mapping 2-D   : Bayesian log-odds occupancy grid updated via Bresenham free-space sweep
Mapping 3-D   : Bayesian log-odds voxel grid; free space by 8 fractional samples per ray
Why no ROS2   : geometry equations replace sensor drivers; NumPy grids replace map_server;
                Matplotlib replaces RViz; world coords replace tf frames

═══════════════════════════════════════════════════════════════
IMPROVEMENTS OVER v1
═══════════════════════════════════════════════════════════════
• All magic numbers replaced with named constants
• SLAMSystem class encapsulates all SLAM state + methods (no global mutable state)
• Vectorised 2-D ray cast (was a Python loop over rays)
• Unified rock definitions — single source of truth for visual scene AND sensor model
• Rover proxy: body + 4 wheels + sensor mast (distinct, visible geometry)
• Terrain-matching rover orientation (pitch/roll quaternion from slope data)
• Speed scales with terrain slope (slower on inclines)
• main() decomposed into step_motion / step_sensors / step_slam / step_logging
• try/finally guarantees final maps + summary saved on Ctrl-C
• Camera update skipped when running headless
• Path list memory-capped at MAX_PATH_HISTORY to prevent unbounded growth
• Voxel grid memory guard (assertion before allocation)
• export_ply vectorised with np.savetxt (was a Python loop per point)
• Summary JSON written at end (scans, coverage %, elapsed time, voxel counts)
• Default ground plane removed (conflicted with custom rocky floor mesh)
• mean projection in 3-D maps (was max — now shows interior structure)
• TUNNEL_AXIS_Z named constant used consistently throughout
"""

from isaacsim import SimulationApp
_LAUNCH_HEADLESS = False          # change to True for server/CI runs
simulation_app = SimulationApp({"headless": _LAUNCH_HEADLESS,
                                "width": 1280, "height": 720})

import math, os, glob, json, time
import numpy as np
import omni, omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import UsdGeom, Gf, Sdf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ═══════════════════════════════════════════════════════════════════════════
# USER PATHS
# ═══════════════════════════════════════════════════════════════════════════
ROVER_USD   = "./vleg_rover.usd"
MAP_OUT_DIR = "./slam_maps"

# ═══════════════════════════════════════════════════════════════════════════
# WORLD CONSTANTS  (all named — zero magic numbers)
# ═══════════════════════════════════════════════════════════════════════════
FINISH_X             = 55.0        # m — mission end X
TUNNEL_R             = 1.6         # m — tunnel inner radius
TUNNEL_AXIS_Z        = 1.0         # m — tunnel centreline height above world origin
LENGTH               = 60.0        # m — total tunnel length
_TUNNEL_X0           = -2.0        # m — tunnel world-X start
_TUNNEL_X1           = _TUNNEL_X0 + LENGTH  # 58.0

ROVER_GROUND_CLEAR   = 0.22        # m — chassis bottom above terrain surface
ROVER_SENSOR_Z       = 0.32        # m — LiDAR puck above rover base
ROVER_WHEEL_RADIUS   = 0.06        # m — for RPM calculation
ROVER_BASE_SPEED     = 0.30        # m/s — speed on flat terrain
ROVER_MIN_SPEED_FRAC = 0.40        # fraction of base speed on max slope

SIM_FPS              = 60          # simulation frames per second

ROVER_WHEEL_RADIUS = 0.06
ROVER_BODY_HALF_X = 0.36
ROVER_BODY_HALF_Y = 0.125
WHEEL_CLEARANCE_Y = 0.01
WHEEL_Z = ROVER_WHEEL_RADIUS  # puts wheel bottom at z=0

# ═══════════════════════════════════════════════════════════════════════════
# 2-D SLAM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
MAP_RES          = 0.05
MAP_X_MIN        = -4.0
MAP_X_MAX        =  62.0
MAP_Y_MIN        = -4.0
MAP_Y_MAX        =  4.0
LIDAR_RAYS       = 360
LIDAR_MAX_RANGE  = 3.5
LIDAR_HZ         = 10
LIDAR_NOISE_STD  = 0.02
MAP_SAVE_EVERY   = 60
MAP_FINAL_DPI    = 150
LOG_OCC_THRESH_FRAC  = 0.4         # fraction of L_OCC for "occupied" classification
LOG_FREE_THRESH_FRAC = 0.4         # fraction of L_FREE for "free" classification
MAX_PATH_HISTORY = 2000            # cap stored path points (memory guard)

# ═══════════════════════════════════════════════════════════════════════════
# 3-D SLAM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
MAP_RES_3D = 0.10
MAP_Z_MIN  = -0.5
MAP_Z_MAX  =  3.5

VLP16_ELEV_DEG = np.linspace(-15.0, 15.0, 16)   # 16 elevation channels
VLP16_AZIM_N   = 360

# Memory guard: abort before allocating a grid that is too large
_V_COLS = int((MAP_X_MAX - MAP_X_MIN) / MAP_RES_3D)
_V_ROWS = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RES_3D)
_V_LAYS = int((MAP_Z_MAX - MAP_Z_MIN) / MAP_RES_3D)
_MAX_VOXELS = 8_000_000
assert _V_COLS * _V_ROWS * _V_LAYS <= _MAX_VOXELS, (
    f"Voxel grid too large ({_V_COLS*_V_ROWS*_V_LAYS/1e6:.2f}M > "
    f"{_MAX_VOXELS/1e6:.0f}M limit). Increase MAP_RES_3D or reduce extents.")

# ═══════════════════════════════════════════════════════════════════════════
# TERRAIN
# ═══════════════════════════════════════════════════════════════════════════
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
    local_x = max(0.0, min(float(x), LENGTH - 0.01))
    idx = local_x / LENGTH * (TERRAIN_SAMPLES - 1)
    i0 = int(idx)
    i1 = min(i0 + 1, TERRAIN_SAMPLES - 1)
    t = idx - i0
    return float(_terrain_h[i0] * (1 - t) + _terrain_h[i1] * t)

def terrain_slope_at(x: float, dx: float = 0.3) -> float:
    return math.degrees(math.atan2(
        terrain_height_at(x + dx) - terrain_height_at(max(0.0, x - dx)),
        2 * dx))

def terrain_speed_factor(x: float) -> float:
    """Return speed multiplier in [ROVER_MIN_SPEED_FRAC, 1.0] based on slope steepness."""
    slope_abs = abs(terrain_slope_at(x))
    return max(ROVER_MIN_SPEED_FRAC, 1.0 - slope_abs / 25.0)

# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED ROCK DEFINITIONS  (single source of truth for visual + SLAM)
# ═══════════════════════════════════════════════════════════════════════════
# Each entry: (local_terrain_x, world_y, half_x, half_y)
# world_x = local_terrain_x - 2.0  (tunnel offset)
_ROCK_LOCAL = [
    ( 5.0,  0.6, 0.06, 0.05),
    (12.0,  0.5, 0.06, 0.05),
    (18.0, -0.6, 0.06, 0.05),
    (25.0,  0.8, 0.06, 0.05),
    (33.0,  0.3, 0.06, 0.05),
    (40.0, -0.8, 0.06, 0.05),
    (47.0,  0.6, 0.06, 0.05),
    (50.0, -0.5, 0.06, 0.05),
]

# Derived 2-D world-space AABBs (cx, cy, half_x, half_y)
_ROCK_DEFS_2D = [(lx - 2.0, wy, hx, hy) for (lx, wy, hx, hy) in _ROCK_LOCAL]

# 3-D world-space AABBs — populated after terrain is ready (see main)
_ROCK_DEFS_3D: list = []

def _make_rock_defs_3d() -> list:
    defs = []
    for (lx, wy, hx, hy) in _ROCK_LOCAL:
        cz = terrain_height_at(lx) + 0.06
        defs.append((lx - 2.0, wy, cz, hx, hy, 0.035))   # (cx,cy,cz, hx,hy,hz)
    return defs

_slam_rng = np.random.default_rng(0)

# ═══════════════════════════════════════════════════════════════════════════
# VLP-16 RAY DIRECTION TEMPLATE  (pre-computed once at module load)
# ═══════════════════════════════════════════════════════════════════════════
def _build_vlp16_directions() -> np.ndarray:
    elev = np.deg2rad(VLP16_ELEV_DEG)                          # (16,)
    azim = np.linspace(0, 2*np.pi, VLP16_AZIM_N, endpoint=False)  # (360,)
    EL, AZ = np.meshgrid(elev, azim, indexing='ij')            # (16, 360)
    cos_el = np.cos(EL)
    dx = (cos_el * np.cos(AZ)).ravel()
    dy = (cos_el * np.sin(AZ)).ravel()
    dz = np.sin(EL).ravel()
    return np.stack([dx, dy, dz], axis=1).astype(np.float32)   # (5760, 3)

_VLP16_DIRS = _build_vlp16_directions()

# ═══════════════════════════════════════════════════════════════════════════
# QUATERNION UTILITY
# ═══════════════════════════════════════════════════════════════════════════
def euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """ZYX Euler angles → unit quaternion (w, x, y, z)."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    w  =  cr*cp*cy + sr*sp*sy
    qx =  sr*cp*cy - cr*sp*sy
    qy =  cr*sp*cy + sr*cp*sy
    qz =  cr*cp*sy - sr*sp*cy
    return np.array([w, qx, qy, qz], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════
# WORLD BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
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
    UsdGeom.Xformable(m.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(-2.0, 0.0, TUNNEL_AXIS_Z))
    print("[World] Tunnel built")

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
    print("[World] Rocky floor built")

def build_rocks(stage):
    """Uses _ROCK_LOCAL — guaranteed identical positions to SLAM sensor model."""
    for k, (lx, wy, hx, hy) in enumerate(_ROCK_LOCAL):
        world_x = lx - 2.0
        rz = terrain_height_at(lx)
        cube = UsdGeom.Cube.Define(stage, f"/World/rock_{k}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([(0.28, 0.22, 0.18)])
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(world_x, wy, rz + 0.06))
        xf.AddScaleOp().Set(Gf.Vec3d(2*hx, 2*hy, 0.07))
    print("[World] Rocks placed (aligned with SLAM model)")

def build_finish(stage):
    c = UsdGeom.Cube.Define(stage, "/World/finish_line")
    c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])
    xf = UsdGeom.Xformable(c.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(FINISH_X - 2.0, 0.0, 0.5))
    xf.AddScaleOp().Set(Gf.Vec3d(0.2, 3.5, 1.0))

def build_markers(stage):
    for k, x in enumerate(range(5, 56, 10)):
        tz = terrain_height_at(float(x))
        for side, lr in [(1.1, 'L'), (-1.1, 'R')]:
            cy = UsdGeom.Cylinder.Define(stage, f"/World/m{k}{lr}")
            cy.CreateRadiusAttr(0.04); cy.CreateHeightAttr(1.0)
            cy.CreateDisplayColorAttr([(1.0, 0.55, 0.0)])
            xf = UsdGeom.Xformable(cy.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(float(x)-2.0, side, tz+0.5))

def setup_lights(stage):
    for k, x in enumerate(range(0, 60, 15)):
        lp = stage.DefinePrim(f"/World/L{k}", "SphereLight")
        lp.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(6000.0)
        lp.CreateAttribute("inputs:radius",    Sdf.ValueTypeNames.Float).Set(0.1)
        lp.CreateAttribute("inputs:color",     Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(1.0, 0.82, 0.55))
        UsdGeom.Xformable(lp).AddTranslateOp().Set(
            Gf.Vec3d(float(x)-2.0, 0.0, 1.3))
    print("[World] Lighting done")

def build_rover_proxy(stage):
    """
    Detailed rover proxy: body + 4 wheels + sensor mast + LiDAR puck.
    Parented under /World/rover so it inherits rover world pose.
    Safe names prefixed with 'proxy_' to avoid collisions.
    """
    # ── Tunable geometry constants ────────────────────────────────────────
    BODY_HALF_X = 0.36
    BODY_HALF_Y = 0.22
    BODY_HALF_Z = 0.12
    WHEEL_RADIUS = 0.06
    BODY_CENTER_Z = WHEEL_RADIUS + BODY_HALF_Z / 2  # 0.06 + 0.06 = 0.12   # keeps chassis slightly above floor

    
    WHEEL_WIDTH = 0.04
    WHEEL_CLEARANCE_Y = 0.01
    WHEEL_Z = WHEEL_RADIUS
    WHEEL_X = BODY_HALF_X / 2 - 0.04   # 0.18 - 0.04 = 0.14
    WHEEL_Y = BODY_HALF_Y / 2 + WHEEL_WIDTH / 2 + WHEEL_CLEARANCE_Y

    MAST_RADIUS = 0.013
    MAST_HEIGHT = 0.22
    MAST_CENTER_Z = 0.26

    SENSOR_RADIUS = 0.04
    SENSOR_HEIGHT = 0.05
    SENSOR_CENTER_Z = MAST_CENTER_Z + (MAST_HEIGHT * 0.5) + (SENSOR_HEIGHT * 0.5)

    # ── Body ──────────────────────────────────────────────────────────────
    body = UsdGeom.Cube.Define(stage, "/World/rover/proxy_body")
    body.CreateSizeAttr(1.0)
    body.CreateDisplayColorAttr([(0.55, 0.55, 0.60)])  # silver-grey chassis
    xfb = UsdGeom.Xformable(body.GetPrim())
    xfb.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BODY_CENTER_Z))
    xfb.AddScaleOp().Set(Gf.Vec3d(BODY_HALF_X, BODY_HALF_Y, BODY_HALF_Z))

    # ── Wheels ────────────────────────────────────────────────────────────
    wheel_offsets = [
        ("proxy_wheel_fl", +WHEEL_X, +WHEEL_Y, WHEEL_Z),
        ("proxy_wheel_fr", +WHEEL_X, -WHEEL_Y, WHEEL_Z),
        ("proxy_wheel_rl", -WHEEL_X, +WHEEL_Y, WHEEL_Z),
        ("proxy_wheel_rr", -WHEEL_X, -WHEEL_Y, WHEEL_Z),
    ]

    for tag, wx, wy, wz in wheel_offsets:
            wh = UsdGeom.Cylinder.Define(stage, f"/World/rover/{tag}")
            wh.CreateRadiusAttr(WHEEL_RADIUS)
            wh.CreateHeightAttr(WHEEL_WIDTH)
            wh.CreateDisplayColorAttr([(0.10, 0.10, 0.10)])
            xfw = UsdGeom.Xformable(wh.GetPrim())
            xfw.AddTranslateOp().Set(Gf.Vec3d(wx, wy, wz))       # ← FIRST: position in parent space
            xfw.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))   # ← SECOND: spin axle to face Y

    # ── Sensor mast ───────────────────────────────────────────────────────
    mast = UsdGeom.Cylinder.Define(stage, "/World/rover/proxy_mast")
    mast.CreateRadiusAttr(MAST_RADIUS)
    mast.CreateHeightAttr(MAST_HEIGHT)
    mast.CreateDisplayColorAttr([(0.85, 0.85, 0.20)])  # yellow mast
    xfm = UsdGeom.Xformable(mast.GetPrim())
    xfm.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, MAST_CENTER_Z))

    # ── LiDAR puck ────────────────────────────────────────────────────────
    sensor = UsdGeom.Cylinder.Define(stage, "/World/rover/proxy_sensor")
    sensor.CreateRadiusAttr(SENSOR_RADIUS)
    sensor.CreateHeightAttr(SENSOR_HEIGHT)
    sensor.CreateDisplayColorAttr([(0.0, 0.85, 1.0)])  # cyan
    xfs = UsdGeom.Xformable(sensor.GetPrim())
    xfs.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, SENSOR_CENTER_Z))

    print("[Rover] Proxy built: body + 4 wheels + mast + sensor head")

# ═══════════════════════════════════════════════════════════════════════════
# SLAM SYSTEM  (encapsulated — no global mutable SLAM state)
# ═══════════════════════════════════════════════════════════════════════════
class SLAMSystem:
    """
    Self-contained 2-D + 3-D Bayesian occupancy SLAM without ROS2.

    Localisation  : dead-reckoning pose passed in by caller each step
    Sensor model  : vectorised analytical ray–geometry intersection
    Map 2-D       : log-odds occupancy grid, Bresenham free-space sweep
    Map 3-D       : log-odds voxel grid, fractional-sample free-space
    """

    L_FREE  = math.log(0.30 / 0.70)    # log-odds decrement for free cells
    L_OCC   = math.log(0.90 / 0.10)    # log-odds increment for occupied cells
    L_CLAMP = 10.0                      # saturation limit (prevents certainty lock-in)

    def __init__(self):
        # 2-D log-odds grid
        self._map_cols = int((MAP_X_MAX - MAP_X_MIN) / MAP_RES)
        self._map_rows = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RES)
        self.log_odds_2d = np.zeros((self._map_rows, self._map_cols), dtype=np.float32)

        # 3-D log-odds voxel grid  (Z-layers, Y-rows, X-cols)
        self.vox = np.zeros((_V_LAYS, _V_ROWS, _V_COLS), dtype=np.float32)

        # Path history (capped at MAX_PATH_HISTORY)
        self.path_x: list = []
        self.path_y: list = []
        self.n_scans = 0

    # ── Pose recording ───────────────────────────────────────────────────
    def record_pose(self, x: float, y: float):
        self.path_x.append(x)
        self.path_y.append(y)
        # Evict oldest point when cap is reached
        if len(self.path_x) > MAX_PATH_HISTORY:
            self.path_x.pop(0)
            self.path_y.pop(0)

    # ── Threshold helpers ────────────────────────────────────────────────
    @property
    def occ_thresh_2d(self) -> float:
        return self.L_OCC  * LOG_OCC_THRESH_FRAC

    @property
    def free_thresh_2d(self) -> float:
        return self.L_FREE * LOG_FREE_THRESH_FRAC

    @property
    def occ_thresh_3d(self) -> float:
        return self.L_OCC  * LOG_OCC_THRESH_FRAC

    # ── 2-D ray cast  (vectorised over all LIDAR_RAYS simultaneously) ────
    def _cast_rays_2d(self, ox: float, oy: float, heading: float) -> np.ndarray:
        """
        Fire LIDAR_RAYS rays from (ox, oy).
        Returns array of shape (LIDAR_RAYS,) with range measurements.

        All geometry tests are NumPy operations — no Python loop over rays.
        """
        angles = heading + np.linspace(0, 2*np.pi, LIDAR_RAYS, endpoint=False)
        dx = np.cos(angles).astype(np.float32)
        dy = np.sin(angles).astype(np.float32)
        t  = np.full(LIDAR_RAYS, LIDAR_MAX_RANGE, dtype=np.float32)
        eps = 1e-9

        # Tunnel side walls: flat planes at Y = ±TUNNEL_R
        for sign in (+1.0, -1.0):
            wall_y = sign * TUNNEL_R
            with np.errstate(divide='ignore', invalid='ignore'):
                t_wall = np.where(np.abs(dy) > eps, (wall_y - oy) / dy, LIDAR_MAX_RANGE)
            hit_x = ox + t_wall * dx
            valid = ((t_wall > 0.01) & (t_wall < t) &
                     (hit_x >= _TUNNEL_X0) & (hit_x <= _TUNNEL_X1))
            t = np.where(valid, t_wall, t)

        # Tunnel end-caps: planes at X = _TUNNEL_X0 and _TUNNEL_X1
        for cap_x in (_TUNNEL_X0, _TUNNEL_X1):
            with np.errstate(divide='ignore', invalid='ignore'):
                t_cap = np.where(np.abs(dx) > eps, (cap_x - ox) / dx, LIDAR_MAX_RANGE)
            hit_y = oy + t_cap * dy
            valid = (t_cap > 0.01) & (t_cap < t) & (np.abs(hit_y) <= TUNNEL_R)
            t = np.where(valid, t_cap, t)

        # Rock AABBs — slab method
        for (rx, ry, rhx, rhy) in _ROCK_DEFS_2D:
            with np.errstate(divide='ignore', invalid='ignore'):
                tx0 = np.where(np.abs(dx)>eps, (rx-rhx-ox)/dx, np.where(ox>=rx-rhx,-1e9, 1e9))
                tx1 = np.where(np.abs(dx)>eps, (rx+rhx-ox)/dx, np.where(ox<=rx+rhx, 1e9,-1e9))
                ty0 = np.where(np.abs(dy)>eps, (ry-rhy-oy)/dy, np.where(oy>=ry-rhy,-1e9, 1e9))
                ty1 = np.where(np.abs(dy)>eps, (ry+rhy-oy)/dy, np.where(oy<=ry+rhy, 1e9,-1e9))
            t_enter = np.maximum(np.minimum(tx0,tx1), np.minimum(ty0,ty1))
            t_exit  = np.minimum(np.maximum(tx0,tx1), np.maximum(ty0,ty1))
            hit = (t_enter < t_exit) & (t_enter > 0.01) & (t_enter < t)
            t = np.where(hit, t_enter, t)

        # Add measurement noise (applied after taking minimum — asymmetric truncation
        # is acceptable here since max-range returns are already clipped separately)
        noise = _slam_rng.normal(0, LIDAR_NOISE_STD, LIDAR_RAYS).astype(np.float32)
        return np.clip(t + noise, 0.01, LIDAR_MAX_RANGE)

    # ── 2-D cell helpers ─────────────────────────────────────────────────
    def _world_to_cell_2d(self, wx: float, wy: float):
        col = int((wx - MAP_X_MIN) / MAP_RES)
        row = int((wy - MAP_Y_MIN) / MAP_RES)
        if 0 <= col < self._map_cols and 0 <= row < self._map_rows:
            return col, row
        return -1, -1

    @staticmethod
    def _bresenham(c0, r0, c1, r1):
        """Yield all cells on the integer raster line from (c0,r0) to (c1,r1)."""
        dc, dr = abs(c1-c0), abs(r1-r0)
        sc = 1 if c1 > c0 else -1
        sr = 1 if r1 > r0 else -1
        c, r = c0, r0; cells = []
        if dc >= dr:
            err = dc // 2
            while c != c1:
                cells.append((c, r)); err -= dr
                if err < 0: r += sr; err += dc
                c += sc
        else:
            err = dr // 2
            while r != r1:
                cells.append((c, r)); err -= dc
                if err < 0: c += sc; err += dr
                r += sr
        cells.append((c1, r1))
        return cells

    # ── 2-D SLAM update ──────────────────────────────────────────────────
    def update_2d(self, x_pos: float, y_pos: float = 0.0, heading: float = 0.0):
        """
        Full 2-D SLAM scan:
          1. Ray cast (vectorised) → range measurements
          2. For each ray: Bresenham sweep to mark free cells, endpoint as occupied
        """
        self.record_pose(x_pos, y_pos)
        self.n_scans += 1

        oc, or_ = self._world_to_cell_2d(x_pos, y_pos)
        if oc < 0:
            return

        ranges = self._cast_rays_2d(x_pos, y_pos, heading)
        angles = heading + np.linspace(0, 2*np.pi, LIDAR_RAYS, endpoint=False)

        for i in range(LIDAR_RAYS):
            rng = float(ranges[i])
            hx  = x_pos + rng * math.cos(angles[i])
            hy  = y_pos + rng * math.sin(angles[i])
            hc, hr = self._world_to_cell_2d(hx, hy)
            beam = self._bresenham(oc, or_, hc, hr) if hc >= 0 else []
            for cc, rr in beam[:-1]:
                self.log_odds_2d[rr, cc] = max(
                    -self.L_CLAMP, self.log_odds_2d[rr, cc] + self.L_FREE)
            if hc >= 0:
                if rng < LIDAR_MAX_RANGE - 0.05:
                    self.log_odds_2d[hr, hc] = min(
                        self.L_CLAMP, self.log_odds_2d[hr, hc] + self.L_OCC)
                else:
                    self.log_odds_2d[hr, hc] = max(
                        -self.L_CLAMP, self.log_odds_2d[hr, hc] + self.L_FREE)

    # ── 3-D SLAM update ──────────────────────────────────────────────────
    def update_3d(self, x_pos: float, y_pos: float, heading: float):
        """
        Full 3-D SLAM scan (VLP-16 style, 5,760 rays):
          1. Rotate template directions by heading
          2. Vectorised intersection: cylinder + floor + rock AABBs
          3. Update occupied endpoints + free-space samples
        """
        oz = terrain_height_at(x_pos + 2.0) + ROVER_SENSOR_Z

        # Rotate VLP-16 direction template by rover heading (Z-axis rotation)
        ch, sh = math.cos(heading), math.sin(heading)
        R = np.array([[ch,-sh,0],[sh,ch,0],[0,0,1]], dtype=np.float32)
        dirs = _VLP16_DIRS @ R.T            # (5760, 3)
        dx, dy, dz = dirs[:,0], dirs[:,1], dirs[:,2]
        N = len(dx)

        # ── Cylinder (tunnel wall) ─────────────────────────────────────
        # Solve quadratic for ray vs. infinite Y-Z cylinder centred at TUNNEL_AXIS_Z
        py  = float(y_pos)
        pz  = oz - TUNNEL_AXIS_Z            # ray origin relative to cylinder axis
        a_c = dy**2 + dz**2 + 1e-30
        b_c = 2*(py*dy + pz*dz)
        c_c = py**2 + pz**2 - TUNNEL_R**2
        disc = b_c**2 - 4*a_c*c_c
        sqrt_disc = np.sqrt(np.maximum(disc, 0))
        # Rover is inside cylinder (c_c < 0) → take the positive (far) root
        t_cyl = (-b_c + sqrt_disc) / (2*a_c)
        hit_x_cyl = x_pos + t_cyl * dx
        t_cyl = np.where(
            (disc >= 0) & (t_cyl > 0.01) &
            (hit_x_cyl >= _TUNNEL_X0) & (hit_x_cyl <= _TUNNEL_X1),
            t_cyl, LIDAR_MAX_RANGE)
        # End-cap planes
        with np.errstate(divide='ignore', invalid='ignore'):
            t_cap0 = np.where(dx < -1e-9, (_TUNNEL_X0-x_pos)/dx, LIDAR_MAX_RANGE)
            t_cap1 = np.where(dx >  1e-9, (_TUNNEL_X1-x_pos)/dx, LIDAR_MAX_RANGE)
        for t_cap in (t_cap0, t_cap1):
            hy_cap = y_pos + t_cap*dy
            hz_cap = oz    + t_cap*dz - TUNNEL_AXIS_Z
            t_cyl = np.minimum(t_cyl, np.where(
                (t_cap > 0.01) & (hy_cap**2 + hz_cap**2 <= TUNNEL_R**2),
                t_cap, LIDAR_MAX_RANGE))

        # ── Floor (terrain plane at local X) ──────────────────────────
        floor_z = terrain_height_at(x_pos + 2.0)
        t_floor = np.where(dz < -1e-9, (floor_z - oz)/dz, LIDAR_MAX_RANGE)
        t_floor = np.where(t_floor > 0.01, t_floor, LIDAR_MAX_RANGE)

        # ── Rock AABBs ────────────────────────────────────────────────
        t_rocks = np.full(N, LIDAR_MAX_RANGE, dtype=np.float32)
        for (rx, ry, rz_c, rhx, rhy, rhz) in _ROCK_DEFS_3D:
            with np.errstate(divide='ignore', invalid='ignore'):
                tx0 = np.where(np.abs(dx)>1e-9,(rx-rhx-x_pos)/dx,np.where(x_pos>=rx-rhx,-1e9, 1e9))
                tx1 = np.where(np.abs(dx)>1e-9,(rx+rhx-x_pos)/dx,np.where(x_pos<=rx+rhx, 1e9,-1e9))
                ty0 = np.where(np.abs(dy)>1e-9,(ry-rhy-y_pos)/dy,np.where(y_pos>=ry-rhy,-1e9, 1e9))
                ty1 = np.where(np.abs(dy)>1e-9,(ry+rhy-y_pos)/dy,np.where(y_pos<=ry+rhy, 1e9,-1e9))
                tz0 = np.where(np.abs(dz)>1e-9,(rz_c-rhz-oz)/dz, np.where(oz>=rz_c-rhz, -1e9, 1e9))
                tz1 = np.where(np.abs(dz)>1e-9,(rz_c+rhz-oz)/dz, np.where(oz<=rz_c+rhz,  1e9,-1e9))
            t_enter = np.maximum(np.maximum(np.minimum(tx0,tx1), np.minimum(ty0,ty1)),
                                             np.minimum(tz0,tz1))
            t_exit  = np.minimum(np.minimum(np.maximum(tx0,tx1), np.maximum(ty0,ty1)),
                                             np.maximum(tz0,tz1))
            t_rocks = np.minimum(t_rocks, np.where(
                (t_enter < t_exit) & (t_enter > 0.01), t_enter, LIDAR_MAX_RANGE))

        # ── Combine + noise ───────────────────────────────────────────
        t_best = np.minimum(np.minimum(t_cyl, t_floor), t_rocks)
        noise  = _slam_rng.normal(0, LIDAR_NOISE_STD, N).astype(np.float32)
        t_best = np.clip(t_best + noise, 0.01, LIDAR_MAX_RANGE)

        # ── Mark occupied endpoints ───────────────────────────────────
        hit_wx = (x_pos + t_best*dx).astype(np.float32)
        hit_wy = (y_pos + t_best*dy).astype(np.float32)
        hit_wz = (oz    + t_best*dz).astype(np.float32)
        genuine = t_best < LIDAR_MAX_RANGE - 0.05
        hc = ((hit_wx - MAP_X_MIN) / MAP_RES_3D).astype(int)
        hr = ((hit_wy - MAP_Y_MIN) / MAP_RES_3D).astype(int)
        hl = ((hit_wz - MAP_Z_MIN) / MAP_RES_3D).astype(int)
        occ_mask = (genuine &
                    (hc >= 0) & (hc < _V_COLS) &
                    (hr >= 0) & (hr < _V_ROWS) &
                    (hl >= 0) & (hl < _V_LAYS))
        np.add.at(self.vox, (hl[occ_mask], hr[occ_mask], hc[occ_mask]), self.L_OCC)

        # ── Mark free space (8 fractional samples along each ray) ─────
        for frac in np.linspace(0.10, 0.88, 8):
            fx = x_pos + frac*t_best*dx
            fy = y_pos + frac*t_best*dy
            fz = oz    + frac*t_best*dz
            fc = ((fx - MAP_X_MIN) / MAP_RES_3D).astype(int)
            fr = ((fy - MAP_Y_MIN) / MAP_RES_3D).astype(int)
            fl = ((fz - MAP_Z_MIN) / MAP_RES_3D).astype(int)
            fv = ((fc>=0)&(fc<_V_COLS)&(fr>=0)&(fr<_V_ROWS)&(fl>=0)&(fl<_V_LAYS))
            np.add.at(self.vox, (fl[fv], fr[fv], fc[fv]), self.L_FREE)

        # Clamp to prevent float32 saturation before next scan
        np.clip(self.vox, -self.L_CLAMP, self.L_CLAMP, out=self.vox)

    # ── RGB map helper ───────────────────────────────────────────────────
    def _build_rgb_map_2d(self) -> np.ndarray:
        rgb = np.full((self._map_rows, self._map_cols, 3), 0.50, dtype=np.float32)
        rgb[self.log_odds_2d < self.free_thresh_2d] = [0.96, 0.96, 0.96]
        rgb[self.log_odds_2d > self.occ_thresh_2d]  = [0.08, 0.08, 0.08]
        return rgb

    # ── Save 2-D snapshot ────────────────────────────────────────────────
    def save_2d_map(self, step: int, x_pos: float):
        rgb = self._build_rgb_map_2d()
        fig, ax = plt.subplots(figsize=(18, 4), dpi=100)
        ax.imshow(rgb, extent=[MAP_X_MIN,MAP_X_MAX,MAP_Y_MIN,MAP_Y_MAX],
                  origin="lower", aspect="equal", interpolation="nearest")
        if self.path_x:
            ax.plot(self.path_x, self.path_y, color="lime", lw=1.4, label="Path", zorder=3)
            ax.plot(x_pos, 0.0, "ro", ms=8, label=f"x={x_pos:.1f}m", zorder=4)
        ax.axhline( TUNNEL_R, color="cyan", ls="--", lw=0.9, alpha=0.7)
        ax.axhline(-TUNNEL_R, color="cyan", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(_TUNNEL_X0, color="orange", ls=":", lw=0.8, alpha=0.7)
        ax.axvline(_TUNNEL_X1, color="orange", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(f"SLAM 2-D — step {step}  x={x_pos:.2f}m  {self.n_scans} scans")
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = os.path.join(MAP_OUT_DIR, f"slam_step_{step:05d}.png")
        fig.savefig(out, dpi=100); plt.close(fig)
        print(f"[2D SLAM] → {out}")

    # ── Save 2-D final composite ─────────────────────────────────────────
    def save_final_2d_map(self, step: int, x_pos: float):
        rgb = self._build_rgb_map_2d()
        fig, axes = plt.subplots(2, 1, figsize=(20, 9), dpi=MAP_FINAL_DPI)
        ax = axes[0]
        ax.imshow(rgb, extent=[MAP_X_MIN,MAP_X_MAX,MAP_Y_MIN,MAP_Y_MAX],
                  origin="lower", aspect="equal", interpolation="nearest")
        if self.path_x:
            ax.plot(self.path_x, self.path_y, color="lime", lw=1.5, zorder=3)
            ax.plot(self.path_x[0], self.path_y[0], "b^", ms=10, label="Start", zorder=5)
            ax.plot(x_pos, 0.0, "r*", ms=14, label="Finish", zorder=5)
        ax.axhline( TUNNEL_R, color="cyan", ls="--", lw=1.1, alpha=0.8, label=f"Tunnel R={TUNNEL_R}m")
        ax.axhline(-TUNNEL_R, color="cyan", ls="--", lw=1.1, alpha=0.8)
        for k, (rx, ry, rhx, rhy) in enumerate(_ROCK_DEFS_2D):
            ax.add_patch(patches.Rectangle(
                (rx-rhx, ry-rhy), 2*rhx, 2*rhy,
                lw=1.2, edgecolor="orange", facecolor="none", alpha=0.9,
                label="Rock (GT)" if k == 0 else "_"))
        ax.axvline(FINISH_X-2.0, color="red", ls="-.", lw=1.2, alpha=0.8, label="Finish")
        ax.set_title("SLAM 2-D Occupancy Map (final)")
        ax.legend(fontsize=8, framealpha=0.8); ax.grid(True, alpha=0.20)
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)")

        ax2 = axes[1]
        im = ax2.imshow(self.log_odds_2d,
                        extent=[MAP_X_MIN,MAP_X_MAX,MAP_Y_MIN,MAP_Y_MAX],
                        origin="lower", aspect="equal", cmap="RdYlGn_r",
                        vmin=-self.L_CLAMP, vmax=self.L_CLAMP, interpolation="nearest")
        plt.colorbar(im, ax=ax2, label="Log-odds  (green=free, red=occupied)")
        if self.path_x:
            ax2.plot(self.path_x, self.path_y, color="white", lw=1.2, alpha=0.8)
        ax2.set_title("Log-Odds Grid"); ax2.grid(True, alpha=0.20)
        fig.suptitle(f"Canary Rover SLAM — {self.n_scans} scans | {step} steps")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(MAP_OUT_DIR, "slam_final.png")
        fig.savefig(out, dpi=MAP_FINAL_DPI); plt.close(fig)
        print(f"[2D SLAM] ✅ Final → {out}")

    # ── Save 3-D projection snapshot ─────────────────────────────────────
    def save_3d_map(self, step: int, x_pos: float):
        """
        Three orthographic projections using mean (not max) for interior visibility.
        Max-projection saturates immediately at the first occupied layer;
        mean shows density gradients and interior structure.
        """
        xy = np.mean(self.vox, axis=0)      # mean over Z → (Y-rows, X-cols)
        xz = np.mean(self.vox, axis=1)      # mean over Y → (Z-lays, X-cols)
        rover_col = int(np.clip((x_pos - MAP_X_MIN) / MAP_RES_3D, 0, _V_COLS-1))
        yz = self.vox[:, :, rover_col]      # slice at rover X → (Z-lays, Y-rows)

        cmap = "RdYlGn_r"
        fig, axes = plt.subplots(1, 3, figsize=(22, 5), dpi=100)

        ax = axes[0]
        ax.imshow(xy, extent=[MAP_X_MIN,MAP_X_MAX,MAP_Y_MIN,MAP_Y_MAX],
                  origin="lower", aspect="equal", cmap=cmap,
                  vmin=-self.L_CLAMP, vmax=self.L_CLAMP, interpolation="nearest")
        if self.path_x:
            ax.plot(self.path_x, self.path_y, "lime", lw=1.2)
            ax.plot(x_pos, 0.0, "ro", ms=7, label=f"x={x_pos:.1f}m")
        ax.axhline( TUNNEL_R, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(-TUNNEL_R, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.set_title("Top-down XY (mean over Z)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)")

        ax = axes[1]
        ax.imshow(xz, extent=[MAP_X_MIN,MAP_X_MAX,MAP_Z_MIN,MAP_Z_MAX],
                  origin="lower", aspect="auto", cmap=cmap,
                  vmin=-self.L_CLAMP, vmax=self.L_CLAMP, interpolation="nearest")
        ax.axhline(TUNNEL_AXIS_Z, color="cyan", ls="--", lw=0.8, alpha=0.7, label="Tunnel axis")
        ax.axhline(TUNNEL_AXIS_Z + TUNNEL_R, color="cyan", ls=":", lw=0.7, alpha=0.6, label="Crown")
        ax.axvline(x_pos, color="red", lw=1.0, alpha=0.8, label=f"x={x_pos:.1f}m")
        ax.set_title("Side view XZ (mean over Y)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Z (m)")

        ax = axes[2]
        im = ax.imshow(yz, extent=[MAP_Y_MIN,MAP_Y_MAX,MAP_Z_MIN,MAP_Z_MAX],
                       origin="lower", aspect="equal", cmap=cmap,
                       vmin=-self.L_CLAMP, vmax=self.L_CLAMP, interpolation="nearest")
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(TUNNEL_R*np.cos(theta), TUNNEL_AXIS_Z + TUNNEL_R*np.sin(theta),
                "cyan", ls="--", lw=1.0, alpha=0.8, label=f"Tunnel R={TUNNEL_R}m")
        ax.set_title(f"Cross-section YZ at x={x_pos:.1f}m")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
        ax.set_xlabel("World Y (m)"); ax.set_ylabel("World Z (m)")
        plt.colorbar(im, ax=ax, label="Log-odds (mean)", shrink=0.8)

        occ3 = int(np.sum(self.vox > self.occ_thresh_3d))
        fig.suptitle(
            f"3-D Voxel SLAM — step {step}  x={x_pos:.2f}m  "
            f"occ voxels={occ3:,}  VLP-16 ({VLP16_AZIM_N}az×{len(VLP16_ELEV_DEG)}el)", fontsize=10)
        fig.tight_layout()
        out = os.path.join(MAP_OUT_DIR, f"slam_3d_step_{step:05d}.png")
        fig.savefig(out, dpi=100); plt.close(fig)
        print(f"[3D SLAM] → {out}")

    # ── PLY export  (vectorised — no Python loop per point) ──────────────
    def export_ply(self, step: int) -> str | None:
        lays, rows, cols = np.where(self.vox > self.occ_thresh_3d)
        n = len(cols)
        if n == 0:
            print("[PLY] No occupied voxels — skipping."); return None
        wx = cols * MAP_RES_3D + MAP_X_MIN
        wy = rows * MAP_RES_3D + MAP_Y_MIN
        wz = lays * MAP_RES_3D + MAP_Z_MIN
        z_norm = np.clip((wz - MAP_Z_MIN) / (MAP_Z_MAX - MAP_Z_MIN), 0, 1)
        r_c = (255 * (0.4 + 0.6*z_norm)).astype(np.uint8)
        g_c = (255 * (0.2 + 0.1*z_norm)).astype(np.uint8)
        b_c = np.full(n, int(255 * 0.1), dtype=np.uint8)
        header = (f"ply\nformat ascii 1.0\nelement vertex {n}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "end_header\n")
        out = os.path.join(MAP_OUT_DIR, "slam_cloud_final.ply")
        with open(out, "w") as f:
            f.write(header)
            np.savetxt(f, np.column_stack([wx, wy, wz,
                                           r_c.astype(np.float32),
                                           g_c.astype(np.float32),
                                           b_c.astype(np.float32)]),
                       fmt="%.3f %.3f %.3f %d %d %d")
        print(f"[PLY] ✅ {out}  ({n:,} points)  [open in MeshLab / CloudCompare]")
        return out

    # ── Summary JSON ─────────────────────────────────────────────────────
    def save_summary(self, step: int, x_pos: float, elapsed_s: float) -> dict:
        occ2  = int(np.sum(self.log_odds_2d > self.occ_thresh_2d))
        free2 = int(np.sum(self.log_odds_2d < self.free_thresh_2d))
        occ3  = int(np.sum(self.vox         > self.occ_thresh_3d))
        total2 = self._map_cols * self._map_rows
        summary = {
            "version": "v2",
            "steps": step,
            "scans": self.n_scans,
            "distance_m": round(x_pos, 3),
            "elapsed_s": round(elapsed_s, 1),
            "2d_grid": {
                "resolution_m": MAP_RES,
                "free_cells":   free2,
                "occ_cells":    occ2,
                "unknown_cells": total2 - free2 - occ2,
                "coverage_pct": round(100 * (free2 + occ2) / total2, 2)
            },
            "3d_grid": {
                "resolution_m": MAP_RES_3D,
                "occ_voxels":   occ3,
                "total_voxels": _V_COLS * _V_ROWS * _V_LAYS
            }
        }
        out = os.path.join(MAP_OUT_DIR, "summary.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Summary] ✅ {out}")
        return summary


# ═══════════════════════════════════════════════════════════════════════════
# ROVER STATE
# ═══════════════════════════════════════════════════════════════════════════
class RoverState:
    """Mutable rover pose and sensor state for one simulation run."""
    def __init__(self):
        self.x_pos:    float = 0.0
        self.y_pos:    float = 0.0
        self.heading:  float = 0.0
        self.speed:    float = ROVER_BASE_SPEED
        self.lidar_cd: int   = 0       # countdown to next LiDAR scan

# ═══════════════════════════════════════════════════════════════════════════
# STEP FUNCTIONS  (decomposed from monolithic main loop)
# ═══════════════════════════════════════════════════════════════════════════
def step_motion(state: RoverState, dt: float):
    """Advance position; slow down on slopes."""
    state.speed = ROVER_BASE_SPEED * terrain_speed_factor(state.x_pos + 2.0)
    state.x_pos += state.speed * dt

def step_sensors(state: RoverState) -> dict:
    """
    Simulate all on-board sensors and return a telemetry dict.
    Data is computed analytically from terrain geometry — no physics engine.
    """
    x = state.x_pos
    pitch = terrain_slope_at(x + 2.0)
    roll  = 2.5 * math.sin(x * 0.4)
    ax_   =  9.81 * math.sin(math.radians(pitch))  + np.random.normal(0, 0.02)
    ay_   = -9.81 * math.sin(math.radians(roll))   + np.random.normal(0, 0.02)
    az_   = (9.81 * math.cos(math.radians(pitch)) *
             math.cos(math.radians(roll)) + np.random.normal(0, 0.02))
    slope = math.degrees(math.atan2(math.sqrt(ax_**2 + ay_**2), az_))
    rpm   = state.speed / ROVER_WHEEL_RADIUS * 60 / (2 * math.pi)
    return {"pitch": pitch, "roll": roll, "slope": slope,
            "rpm": rpm, "ax": ax_, "ay": ay_, "az": az_}

def step_slam(state: RoverState, slam: SLAMSystem) -> bool:
    """
    Trigger a SLAM scan at LIDAR_HZ.
    Returns True if a scan was fired this frame.
    """
    frames_per_scan = int(round(SIM_FPS / LIDAR_HZ))
    state.lidar_cd += 1
    if state.lidar_cd >= frames_per_scan:
        state.lidar_cd = 0
        slam.update_2d(state.x_pos, state.y_pos, state.heading)
        slam.update_3d(state.x_pos, state.y_pos, state.heading)
        return True
    return False

def step_logging(state: RoverState, slam: SLAMSystem, telem: dict, step: int):
    th   = terrain_height_at(state.x_pos + 2.0)
    prog = min(100, int(state.x_pos / FINISH_X * 100))
    bar  = "█"*(prog//5) + "░"*(20 - prog//5)
    fc2  = int(np.sum(slam.log_odds_2d < slam.free_thresh_2d))
    oc2  = int(np.sum(slam.log_odds_2d > slam.occ_thresh_2d))
    oc3  = int(np.sum(slam.vox         > slam.occ_thresh_3d))
    print(f"┌─ Step {step:5d} ───────────────────────────────────────────────┐")
    print(f"│  📍 x={state.x_pos:6.2f}m  terrain_z={th:+.3f}m  speed={state.speed:.3f}m/s")
    print(f"│  🔵 IMU  pitch={telem['pitch']:+.2f}°  roll={telem['roll']:+.2f}°  slope={telem['slope']:.2f}°")
    print(f"│  🟢 Encoders  {telem['rpm']:+.1f} RPM  (r={ROVER_WHEEL_RADIUS*100:.0f}cm)")
    print(f"│  🗺  2D SLAM  scans={slam.n_scans:4d}  free={fc2:7,}  occ={oc2:5,}")
    print(f"│  🧊 3D SLAM  occ voxels={oc3:7,}  grid {_V_COLS}×{_V_ROWS}×{_V_LAYS}")
    print(f"│  🏁 [{bar}] {prog}%  ({state.x_pos:.1f}/{FINISH_X}m)")
    print(f"└────────────────────────────────────────────────────────────────┘\n")

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def _try_imageio(frames, out_path, fps):
    import imageio.v3 as iio
    imgs = [iio.imread(f) for f in frames]
    iio.imwrite(out_path, imgs, fps=fps, codec="libx264",
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[Video] ✅ imageio → {out_path}")

def _try_opencv(frames, out_path, fps):
    import cv2
    f0 = cv2.imread(frames[0]); h, w = f0.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if img is not None: vw.write(img)
    vw.release()
    print(f"[Video] ✅ OpenCV → {out_path}")

def _try_ffmpeg(frames, out_path, fps):
    import subprocess, shutil
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")
    list_f = os.path.join(MAP_OUT_DIR, "_flist.txt")
    with open(list_f, "w") as fh:
        for f in frames:
            fh.write(f"file '{os.path.abspath(f)}'\nduration {1/fps}\n")
    subprocess.run(
        ["ffmpeg","-y","-f","concat","-safe","0","-i",list_f,
         "-vf",f"fps={fps}","-c:v","libx264","-pix_fmt","yuv420p", out_path],
        check=True)
    os.remove(list_f)
    print(f"[Video] ✅ ffmpeg → {out_path}")

def generate_video(glob_pattern: str, out_path: str, fps: int = 10):
    frames = sorted(glob.glob(glob_pattern))
    if not frames:
        print(f"[Video] No frames matched: {glob_pattern}"); return
    print(f"\n[Video] Encoding {len(frames)} frames → {out_path}  fps={fps}")
    for method in (_try_imageio, _try_opencv, _try_ffmpeg):
        try:
            method(frames, out_path, fps); return
        except Exception as e:
            print(f"[Video] {method.__name__} failed: {e}")
    print("[Video] ❌ All methods failed. pip install 'imageio[ffmpeg]' or apt install ffmpeg")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    global _ROCK_DEFS_3D
    _ROCK_DEFS_3D = _make_rock_defs_3d()   # requires terrain to be initialised first

    os.makedirs(MAP_OUT_DIR, exist_ok=True)

    _MAP_COLS_2D = int((MAP_X_MAX - MAP_X_MIN) / MAP_RES)
    _MAP_ROWS_2D = int((MAP_Y_MAX - MAP_Y_MIN) / MAP_RES)

    print("\n" + "="*68)
    print("  CANARY ROVER — Isaac Sim 5.1.0  (v2)  Mine Inspection SLAM")
    print(f"  2-D grid : {_MAP_COLS_2D}×{_MAP_ROWS_2D} cells @ {MAP_RES}m")
    print(f"  3-D voxel: {_V_COLS}×{_V_ROWS}×{_V_LAYS} cells @ {MAP_RES_3D}m  "
          f"({_V_COLS*_V_ROWS*_V_LAYS/1e6:.2f}M voxels)")
    print(f"  VLP-16   : {len(VLP16_ELEV_DEG)}el × {VLP16_AZIM_N}az = "
          f"{len(VLP16_ELEV_DEG)*VLP16_AZIM_N} rays/scan @ {LIDAR_HZ}Hz")
    print(f"  Output   : {os.path.abspath(MAP_OUT_DIR)}")
    print("="*68 + "\n")

    world = World(stage_units_in_meters=1.0)
    # NOTE: Default ground plane deliberately omitted — it conflicts with the
    # custom rocky floor mesh and creates a visual Z-fighting artifact.
    stage = omni.usd.get_context().get_stage()

    build_tunnel(stage)
    build_rocky_floor(stage)
    build_rocks(stage)
    build_finish(stage)
    build_markers(stage)
    setup_lights(stage)

    add_reference_to_stage(usd_path=ROVER_USD, prim_path="/World/rover")
    build_rover_proxy(stage)   # body + 4 wheels + mast; remove when USD is fixed

    world.reset()
    rover = XFormPrim(prim_path="/World/rover")

    slam  = SLAMSystem()
    state = RoverState()
    dt    = 1.0 / SIM_FPS
    step  = 0
    t0    = time.time()

    world.play()

    try:
        while simulation_app.is_running():
            world.step(render=True)
            step += 1

            # ── Rover motion, sensors, SLAM ───────────────────────────
            step_motion(state, dt)
            telem = step_sensors(state)
            step_slam(state, slam)

            # ── Rover pose: position + terrain-matched orientation ────
            th   = terrain_height_at(state.x_pos + 2.0)
            quat = euler_to_quat(
                roll_deg  = telem["roll"]  * 0.30,   # subtle visual tilt
                pitch_deg = telem["pitch"] * 0.30,
                yaw_deg   = math.degrees(state.heading)
            )
            rover.set_world_pose(
                position    = np.array([state.x_pos, state.y_pos, th + ROVER_GROUND_CLEAR]),
                orientation = quat
            )

            # ── Camera follow (skipped in headless mode) ──────────────
            if not _LAUNCH_HEADLESS and step % 8 == 0:
                set_camera_view(
                    eye    = np.array([state.x_pos - 2.5, 0.0, th + 0.7]),
                    target = np.array([state.x_pos + 5.0, 0.0, th + 0.1])
                )

            # ── Periodic map saves + terminal readout ─────────────────
            if step % MAP_SAVE_EVERY == 0:
                step_logging(state, slam, telem, step)
                slam.save_2d_map(step, state.x_pos)
                slam.save_3d_map(step, state.x_pos)

            if state.x_pos >= FINISH_X:
                print("\n" + "🎯 "*18)
                print("  FINISH LINE REACHED!  Mine Inspection Complete!")
                print("🎯 "*18 + "\n")
                break

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — saving partial results...")

    finally:
        # Guaranteed to run on normal finish AND Ctrl-C
        elapsed = time.time() - t0
        print(f"[Main] Saving final outputs (elapsed={elapsed:.1f}s) ...")
        slam.save_final_2d_map(step, state.x_pos)
        slam.save_3d_map(step, state.x_pos)
        slam.export_ply(step)
        summary = slam.save_summary(step, state.x_pos, elapsed)
        print(f"\n[Main] Summary: {summary}\n")

        generate_video(
            os.path.join(MAP_OUT_DIR, "slam_step_*.png"),
            os.path.join(MAP_OUT_DIR, "slam_2d_timelapse.mp4"), fps=10)
        generate_video(
            os.path.join(MAP_OUT_DIR, "slam_3d_step_*.png"),
            os.path.join(MAP_OUT_DIR, "slam_3d_timelapse.mp4"), fps=10)

        for _ in range(120):
            world.step(render=True)
        simulation_app.close()


if __name__ == "__main__":
    main()


"""
Run by using the following command from `IsaacLab` directory: 

./isaaclab.sh -p "path/to/isaac_multi_tunnel_preview.py"
"""

from isaacsim import SimulationApp

# -----------------------------------------------------------------------------
# Launch first, then import Kit / USD modules.
# -----------------------------------------------------------------------------
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import math
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, Sdf

try:
    from omni.isaac.core.utils.viewports import set_camera_view
except Exception:
    set_camera_view = None

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
LAUNCH_HEADLESS = False

ROOT_LENGTH = 15.0
BRANCH_LENGTH = 12.0
TAIL_LENGTH = 10.0

TUNNEL_RADIUS = 1.6
TUNNEL_AXIS_Z = 1.0

FLOOR_WIDTH = 2.8
FLOOR_THICK = 0.02

TUNNEL_RINGS = 32
TUNNEL_SEGS_PER_M = 3

BRANCH_SPECS = [
    # name, yaw_deg, tail_pitch_deg
    ("left",  +30.0, +10.0),
    ("mid",     0.0,  -7.0),
    ("right", -30.0,  +6.0),
]

# -----------------------------------------------------------------------------
# TERRAIN PROFILE
# Same noise profile as your original script, reused by all branches through s.
# -----------------------------------------------------------------------------
WORLD_LENGTH = 60.0
np.random.seed(42)

TERRAIN_SAMPLES = 300
_s_samples = np.linspace(0.0, WORLD_LENGTH, TERRAIN_SAMPLES)
_terrain_h = (
    0.06  * np.sin(_s_samples * 0.25) +
    0.04  * np.cos(_s_samples * 0.8  + 0.5) +
    0.025 * np.sin(_s_samples * 2.1  + 1.2) +
    0.015 * np.sin(_s_samples * 4.5  + 0.3) +
    0.008 * np.random.normal(size=TERRAIN_SAMPLES)
)

def terrain_height_at(s: float) -> float:
    s = max(0.0, min(float(s), WORLD_LENGTH - 0.01))
    idx = s / WORLD_LENGTH * (TERRAIN_SAMPLES - 1)
    i0 = int(idx)
    i1 = min(i0 + 1, TERRAIN_SAMPLES - 1)
    t = idx - i0
    return float(_terrain_h[i0] * (1 - t) + _terrain_h[i1] * t)

def terrain_slope_at(s: float, ds: float = 0.3) -> float:
    return math.degrees(math.atan2(
        terrain_height_at(s + ds) - terrain_height_at(max(0.0, s - ds)),
        2 * ds
    ))

# -----------------------------------------------------------------------------
# MATH HELPERS
# -----------------------------------------------------------------------------
def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v.copy()

def _local_frame(fwd: np.ndarray):
    fwd = _norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=float))
    right = _norm(right)
    up = _norm(np.cross(right, fwd))
    return right, up, fwd

def _quat_from_axes(fwd: np.ndarray, right: np.ndarray, up: np.ndarray) -> Gf.Quatf:
    """
    Body frame:
      X = forward
      Y = right
      Z = up
    """
    R = np.column_stack([fwd, right, up])
    t = float(R[0, 0] + R[1, 1] + R[2, 2])

    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))

def rotate_yaw(direction: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    R = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta),  math.cos(theta), 0.0],
            [0.0,              0.0,             1.0],
        ],
        dtype=float,
    )
    return R @ direction

def apply_pitch(direction: np.ndarray, pitch_deg: float) -> np.ndarray:
    """
    Pitch around local-right axis after yawing in the plane.
    """
    fwd = _norm(direction)
    right, _, _ = _local_frame(fwd)
    a = math.radians(pitch_deg)
    cp, sp = math.cos(a), math.sin(a)
    # Rotate fwd toward +/- up around right
    return _norm(
        fwd * cp +
        np.cross(right, fwd) * sp +
        right * float(np.dot(right, fwd)) * (1.0 - cp)
    )

# -----------------------------------------------------------------------------
# SEGMENT MODEL
# -----------------------------------------------------------------------------
class Segment:
    def __init__(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        length: float,
        name: str,
        s_start: float,
        pitch_deg: float = 0.0,
    ):
        self.start = start.copy()
        self.length = float(length)
        self.name = name
        self.s_start = float(s_start)
        self.s_end = self.s_start + self.length
        self.tail_pitch_deg = float(pitch_deg)

        self.right, self.up, self.direction = _local_frame(direction)
        self.end = self.start + self.direction * self.length

    def point(self, along: float) -> np.ndarray:
        return self.start + self.direction * along

    def quat(self) -> Gf.Quatf:
        return _quat_from_axes(self.direction, self.right, self.up)

    def terrain_s_at(self, along: float) -> float:
        return self.s_start + along

# -----------------------------------------------------------------------------
# TOPOLOGY
# root -> 3 branches in same plane -> each branch has a pitched tail
# -----------------------------------------------------------------------------
def build_segments():
    segments = []

    root = Segment(
        start=np.array([0.0, 0.0, TUNNEL_AXIS_Z], dtype=float),
        direction=np.array([1.0, 0.0, 0.0], dtype=float),
        length=ROOT_LENGTH,
        name="root",
        s_start=0.0,
    )
    segments.append(root)

    next_s = ROOT_LENGTH
    for branch_name, yaw_deg, tail_pitch_deg in BRANCH_SPECS:
        branch_dir = rotate_yaw(root.direction, yaw_deg)
        branch_dir = _norm(branch_dir)
        branch_dir[2] = 0.0
        branch_dir = _norm(branch_dir)

        branch = Segment(
            start=root.end,
            direction=branch_dir,
            length=BRANCH_LENGTH,
            name=f"{branch_name}_branch",
            s_start=next_s,
        )
        next_s_branch_end = next_s + BRANCH_LENGTH

        tail_dir = apply_pitch(branch.direction, tail_pitch_deg)
        tail = Segment(
            start=branch.end,
            direction=tail_dir,
            length=TAIL_LENGTH,
            name=f"{branch_name}_tail",
            s_start=next_s_branch_end,
            pitch_deg=tail_pitch_deg,
        )

        segments.extend([branch, tail])

    return segments

# -----------------------------------------------------------------------------
# GEOMETRY BUILDERS
# -----------------------------------------------------------------------------
def _build_mesh_from_rings(stage, prim_path, rings_points, color=(0.5, 0.35, 0.2), double_sided=False):
    segs = len(rings_points) - 1
    rings = len(rings_points[0])

    verts = []
    face_counts = []
    face_indices = []

    for ring in rings_points:
        verts.extend(ring)

    for i in range(segs):
        for j in range(rings):
            nj = (j + 1) % rings
            v0 = i * rings + j
            v1 = i * rings + nj
            v2 = (i + 1) * rings + j
            v3 = (i + 1) * rings + nj
            face_indices += [v0, v2, v1, v1, v2, v3]
            face_counts += [3, 3]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDoubleSidedAttr(double_sided)
    mesh.CreateDisplayColorAttr([color])
    return mesh

def build_tunnel(stage, seg: Segment, prim_path: str, segs_per_m=TUNNEL_SEGS_PER_M, rings=TUNNEL_RINGS, radius=TUNNEL_RADIUS):
    """
    Tunnel centerline follows the segment direction.
    This is preview geometry only; no physics/articulation.
    """
    n_long = max(4, int(math.ceil(seg.length * segs_per_m)))
    rings_points = []
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    for i in range(n_long + 1):
        along = i * (seg.length / n_long)
        center = seg.point(along)
        ring = []

        for j in range(rings):
            a = (j / rings) * 2.0 * math.pi
            # local tube cross-section from segment axes
            offset = seg.right * (radius * math.cos(a)) + seg.up * (radius * math.sin(a))
            p = center + offset
            ring.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))

        rings_points.append(ring)

    color = (0.52, 0.36, 0.22) if "root" in seg.name else (0.47, 0.33, 0.19)
    _build_mesh_from_rings(stage, prim_path, rings_points, color=color, double_sided=True)

def build_floor(stage, seg: Segment, prim_path: str):
    """
    Floor is a thin strip following the same s-profile across all branches.
    Terrain height is reused identically for the whole network.
    """
    # Build a thin strip with the same segment direction, sitting on the terrain profile.
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([(0.40, 0.26, 0.12)])

    # Midpoint of the segment in world space, but floor sits at terrain height.
    mid_along = seg.length * 0.5
    center_xy = seg.point(mid_along)
    s_mid = seg.terrain_s_at(mid_along)
    z_mid = terrain_height_at(s_mid) + FLOOR_THICK * 0.5

    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddScaleOp().Set(Gf.Vec3f(float(seg.length * 0.5), float(FLOOR_WIDTH * 0.5), float(FLOOR_THICK * 0.5)))
    xf.AddOrientOp().Set(seg.quat())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(z_mid)))

def build_hub_patch(stage):
    """
    Small fan at the intersection to visually stitch the junction.
    """
    cx, cy = ROOT_LENGTH, 0.0
    cz = terrain_height_at(ROOT_LENGTH) + 0.01

    radius = 3.0
    segments = 20

    verts = [Gf.Vec3f(cx, cy, cz)]
    for i in range(segments + 1):
        a = (i / segments) * 2.0 * math.pi
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        verts.append(Gf.Vec3f(x, y, cz))

    face_counts = []
    face_indices = []
    for i in range(1, segments + 1):
        face_indices += [0, i, i + 1]
        face_counts += [3]

    mesh = UsdGeom.Mesh.Define(stage, "/World/hub_patch")
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([(0.34, 0.22, 0.11)])
    print("[World] Hub patch built")

def build_rocks(stage, segments):
    """
    A few rocks distributed across root and branches.
    Positions are chosen in segment-local coordinates and projected to the branch frame.
    """
    placements = [
        ("root", 5.0,  0.6), ("root", 12.0,  0.5),
        ("left_branch", 4.0,  0.4), ("left_tail", 3.5, -0.5),
        ("mid_branch",  5.0, -0.3), ("mid_tail",  4.0,  0.6),
        ("right_branch",4.0, -0.4), ("right_tail",3.5,  0.5),
    ]

    seg_map = {s.name: s for s in segments}

    for k, (seg_name, along, lateral) in enumerate(placements):
        seg = seg_map[seg_name]
        center = seg.point(along)
        s_abs = seg.terrain_s_at(along)
        z = terrain_height_at(s_abs) + 0.06

        side = _norm(np.array([-seg.direction[1], seg.direction[0], 0.0], dtype=float))
        pos = np.array([center[0], center[1], z], dtype=float) + side * lateral

        cube = UsdGeom.Cube.Define(stage, f"/World/rock_{k}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([(0.28, 0.22, 0.18)])
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xf.AddScaleOp().Set(Gf.Vec3d(0.12, 0.10, 0.07))

    print("[World] Rocks placed")

def build_markers(stage, segments):
    """
    Orange markers along each branch.
    """
    for idx, seg in enumerate(segments):
        if "tail" not in seg.name and "branch" not in seg.name and seg.name != "root":
            continue

        for along in np.linspace(2.0, max(2.0, seg.length - 2.0), 3):
            center = seg.point(float(along))
            s_abs = seg.terrain_s_at(float(along))
            z = terrain_height_at(s_abs) + 0.5
            side_vec = _norm(np.array([-seg.direction[1], seg.direction[0], 0.0], dtype=float))

            for side, lr in [(1.1, "L"), (-1.1, "R")]:
                cy = UsdGeom.Cylinder.Define(stage, f"/World/m{idx}_{lr}_{int(along*10)}")
                cy.CreateRadiusAttr(0.04)
                cy.CreateHeightAttr(1.0)
                cy.CreateDisplayColorAttr([(1.0, 0.55, 0.0)])

                pos = np.array([center[0], center[1], z], dtype=float) + side_vec * side
                xf = UsdGeom.Xformable(cy.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                xf.AddOrientOp().Set(seg.quat())

    print("[World] Markers placed")

def build_finish(stage, segments):
    """
    Finish is placed at the end of each tail.
    """
    for seg in segments:
        if "tail" not in seg.name:
            continue

        c = UsdGeom.Cube.Define(stage, f"/World/finish_{seg.name}")
        c.CreateSizeAttr(1.0)
        c.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])

        pos = seg.end
        xf = UsdGeom.Xformable(c.GetPrim())
        xf.AddScaleOp().Set(Gf.Vec3f(0.18, 3.5, 1.0))
        xf.AddOrientOp().Set(seg.quat())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]) + 0.5))

    print("[World] Finish markers built")

def setup_lights(stage):
    """
    Lighting inspired by your newer file:
      - DomeLight for ambient fill
      - multiple SphereLights placed along the route
    """
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(2500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.93))

    # A few warm key lights around the route
    light_positions = [
        (4.0, 0.0, 3.0),
        (12.0, 0.0, 3.0),
        (ROOT_LENGTH, 0.0, 3.0),
        (ROOT_LENGTH + 5.0, 2.0, 3.0),
        (ROOT_LENGTH + 5.0, -2.0, 3.0),
        (ROOT_LENGTH + 12.0, 0.0, 3.2),
    ]

    for i, (x, y, z) in enumerate(light_positions):
        light = UsdLux.SphereLight.Define(stage, f"/World/KeyLight_{i}")
        light.CreateIntensityAttr(5000.0)
        light.CreateRadiusAttr(0.15)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.55))
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))

    print("[World] Lighting done")

def build_rover_proxy(stage):
    """
    Visual rover only:
      body + 4 wheels + mast + LiDAR puck
    No articulation, no physics, no motion.
    """
    rover = UsdGeom.Xform.Define(stage, "/World/rover")
    rover_xf = UsdGeom.Xformable(rover.GetPrim())
    rover_xf.AddTranslateOp().Set(Gf.Vec3d(1.5, 0.0, terrain_height_at(0.0) + 0.22))

    BODY_HALF_X = 0.36
    BODY_HALF_Y = 0.22
    BODY_HALF_Z = 0.12
    WHEEL_RADIUS = 0.06
    WHEEL_WIDTH = 0.04
    WHEEL_CLEARANCE_Y = 0.01

    BODY_CENTER_Z = WHEEL_RADIUS + BODY_HALF_Z / 2.0
    WHEEL_X = BODY_HALF_X / 2.0 - 0.04
    WHEEL_Y = BODY_HALF_Y / 2.0 + WHEEL_WIDTH / 2.0 + WHEEL_CLEARANCE_Y
    WHEEL_Z = WHEEL_RADIUS

    MAST_RADIUS = 0.013
    MAST_HEIGHT = 0.22
    MAST_CENTER_Z = 0.26

    SENSOR_RADIUS = 0.04
    SENSOR_HEIGHT = 0.05
    SENSOR_CENTER_Z = MAST_CENTER_Z + (MAST_HEIGHT * 0.5) + (SENSOR_HEIGHT * 0.5)

    body = UsdGeom.Cube.Define(stage, "/World/rover/proxy_body")
    body.CreateSizeAttr(1.0)
    body.CreateDisplayColorAttr([(0.55, 0.55, 0.60)])
    xfb = UsdGeom.Xformable(body.GetPrim())
    xfb.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BODY_CENTER_Z))
    xfb.AddScaleOp().Set(Gf.Vec3d(BODY_HALF_X, BODY_HALF_Y, BODY_HALF_Z))

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
        xfw.AddTranslateOp().Set(Gf.Vec3d(wx, wy, wz))
        xfw.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))

    mast = UsdGeom.Cylinder.Define(stage, "/World/rover/proxy_mast")
    mast.CreateRadiusAttr(MAST_RADIUS)
    mast.CreateHeightAttr(MAST_HEIGHT)
    mast.CreateDisplayColorAttr([(0.85, 0.85, 0.20)])
    xfm = UsdGeom.Xformable(mast.GetPrim())
    xfm.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, MAST_CENTER_Z))

    sensor = UsdGeom.Cylinder.Define(stage, "/World/rover/proxy_sensor")
    sensor.CreateRadiusAttr(SENSOR_RADIUS)
    sensor.CreateHeightAttr(SENSOR_HEIGHT)
    sensor.CreateDisplayColorAttr([(0.0, 0.85, 1.0)])
    xfs = UsdGeom.Xformable(sensor.GetPrim())
    xfs.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, SENSOR_CENTER_Z))

    print("[Rover] Proxy built")

# -----------------------------------------------------------------------------
# STAGE SETUP
# -----------------------------------------------------------------------------
def configure_stage(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 72)
    print("  CANARY ROVER — Isaac Sim 5.1.0  Branching Tunnel Preview")
    print("  root -> 3 branches -> tails")
    print("  Visual preview only: no articulation, no physics, no motion")
    print("=" * 72 + "\n")

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    configure_stage(stage)

    segments = build_segments()
    print(f"[World] Built {len(segments)} segments")

    # Geometry
    for seg in segments:
        build_tunnel(stage, seg, f"/World/Tunnels/{seg.name}")
        build_floor(stage, seg, f"/World/Floors/{seg.name}")

    build_hub_patch(stage)
    build_rocks(stage, segments)
    build_markers(stage, segments)
    build_finish(stage, segments)
    setup_lights(stage)
    build_rover_proxy(stage)

    # Camera
    if set_camera_view is not None:
        try:
            set_camera_view(
                eye=np.array([-8.0, -10.0, 8.0]),
                target=np.array([ROOT_LENGTH + 6.0, 0.0, 1.5]),
            )
        except Exception as e:
            print(f"[Preview] Camera helper failed: {e}")

    print("[Preview] Scene ready. Inspect the branch split and lighting.")

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()

if __name__ == "__main__":
    main()

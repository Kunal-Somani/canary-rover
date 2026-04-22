from isaacsim import SimulationApp

# -----------------------------------------------------------------------------
# Launch first, then import Kit / USD modules.
# -----------------------------------------------------------------------------
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import math
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
from pxr import PhysxSchema

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim

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
FLOOR_THICK = 0.2               # increased for better collision

TUNNEL_RINGS = 32
TUNNEL_SEGS_PER_M = 3

# straight tunnel branches only; no inclination
BRANCH_SPECS = [
    ("left",  +30.0),
    ("mid",     0.0),
    ("right", -30.0),
]

# soil configuration: cover the tunnel up to 60% of its height
SOIL_COVER_FRAC = 0.60
SOIL_OPENING_LENGTH = 2.0
SOIL_MAIN_Y_PAD = 1.8
SOIL_CAP_LEN = 1.5

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

def yaw_from_quat_wxyz(q):
    w, x, y, z = map(float, q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

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
    ):
        self.start = start.copy()
        self.length = float(length)
        self.name = name
        self.s_start = float(s_start)
        self.s_end = self.s_start + self.length

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
# root -> 3 branches in same plane -> each branch has a straight tail
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
    for branch_name, yaw_deg in BRANCH_SPECS:
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

        tail = Segment(
            start=branch.end,
            direction=branch.direction.copy(),
            length=TAIL_LENGTH,
            name=f"{branch_name}_tail",
            s_start=next_s_branch_end,
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
    This is preview geometry only; collision is added so the rover can interact with it.
    """
    n_long = max(4, int(math.ceil(seg.length * segs_per_m)))
    rings_points = []

    for i in range(n_long + 1):
        along = i * (seg.length / n_long)
        center = seg.point(along)
        ring = []

        for j in range(rings):
            a = (j / rings) * 2.0 * math.pi
            offset = seg.right * (radius * math.cos(a)) + seg.up * (radius * math.sin(a))
            p = center + offset
            ring.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))

        rings_points.append(ring)

    color = (0.52, 0.36, 0.22) if "root" in seg.name else (0.47, 0.33, 0.19)
    mesh = _build_mesh_from_rings(stage, prim_path, rings_points, color=color, double_sided=True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")

    return mesh

def build_floor(stage, seg: Segment, prim_path: str):
    """
    Creates a continuous mesh floor that follows the terrain height profile
    with small sinusoidal bumps (matching the preview script).
    Collision is added so the rover can drive on it.
    """
    SX = 80          # steps along the segment
    SY = 16          # steps across the width

    verts = []
    indices = []
    counts = []

    for i in range(SX):
        along = (i / (SX - 1)) * seg.length
        center = seg.point(along)

        s_abs = seg.terrain_s_at(along)
        base_z = terrain_height_at(s_abs)

        for j in range(SY):
            frac = j / (SY - 1)
            lateral = (frac - 0.5) * FLOOR_WIDTH

            side = _norm(np.array([-seg.direction[1], seg.direction[0], 0.0]))
            p = center + side * lateral

            # Bumpiness identical to preview script
            z = base_z + 0.01 * math.sin(i * 0.3 + j * 0.8)

            verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(z)))

    # Build quad faces
    for i in range(SX - 1):
        for j in range(SY - 1):
            v0 = i * SY + j
            v1 = i * SY + (j + 1)
            v2 = (i + 1) * SY + j
            v3 = (i + 1) * SY + (j + 1)

            indices += [v0, v1, v3, v2]
            counts.append(4)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)                     # so rover contacts from above
    mesh.CreateDisplayColorAttr([(0.32, 0.20, 0.10)])

    # Add collision properties
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    physx_coll.CreateContactOffsetAttr(0.02)
    physx_coll.CreateRestOffsetAttr(0.001)

    print(f"[Floor] Mesh built for {seg.name}")

# def _add_soil_cube(stage, prim_path, center, orient_q, scale_xyz, color=(0.38, 0.26, 0.14)):
#     cube = UsdGeom.Cube.Define(stage, prim_path)
#     cube.CreateSizeAttr(1.0)
#     cube.CreateDisplayColorAttr([color])
#     xf = UsdGeom.Xformable(cube.GetPrim())
#     xf.AddOrientOp().Set(orient_q)
#     xf.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
#     xf.AddScaleOp().Set(Gf.Vec3d(float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])))
#     return cube

def build_soil_mesh(stage, segments):
    """
    Smooth soil embankments with a slightly rounded top to avoid boxy look.
    """
    stage.DefinePrim("/World/Soil", "Xform")

    radius = TUNNEL_RADIUS
    center_z = TUNNEL_AXIS_Z
    bank_height_target = 0.6 * (2.0 * radius)
    bank_thickness = 1.6 * 5.0          # 8.0 m
    underfloor_depth = TUNNEL_RADIUS    # 1.6 m

    SX = 80          # along segment
    SY_bank = 32     # increased for smoother top
    SY_under = 20

    for seg in segments:
        start_along = 0.0
        end_along = seg.length
        if end_along <= start_along:
            continue

        side_dir = _norm(np.array([-seg.direction[1], seg.direction[0], 0.0], dtype=float))

        # -----------------------------------------------------------------
        # BANKS (left and right)
        # -----------------------------------------------------------------
        for side_sign, side_name in [(1, "L"), (-1, "R")]:
            verts = []
            indices = []
            counts = []

            # Precompute top heights along the segment (without thickness variation yet)
            top_z_along = []
            for i in range(SX):
                along = start_along + (i / (SX - 1)) * (end_along - start_along)
                s_abs = seg.terrain_s_at(along)
                base_z = terrain_height_at(s_abs)
                max_allowed = center_z + radius
                top_z = min(base_z + bank_height_target, max_allowed)
                # Small bump along length only
                top_z += 0.008 * math.sin(i * 0.4)
                top_z_along.append(top_z)

            for i in range(SX):
                along = start_along + (i / (SX - 1)) * (end_along - start_along)
                center = seg.point(along)
                base_z = terrain_height_at(seg.terrain_s_at(along))
                top_z_base = top_z_along[i]

                for j in range(SY_bank):
                    frac = j / (SY_bank - 1)   # 0 = inner edge, 1 = outer edge
                    dist = radius + frac * bank_thickness
                    offset = side_dir * (side_sign * dist)
                    p = center + offset

                    # Smooth curvature: top is slightly higher in the middle (convex)
                    curve = 1.0 - 4.0 * (frac - 0.5) * (frac - 0.5)   # parabola, max at 0.5
                    top_z = top_z_base + 0.03 * curve   # raise middle by 3 cm

                    verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(base_z)))
                    verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(top_z)))

            def idx(i_, j_, layer):
                return (i_ * SY_bank + j_) * 2 + layer

            for i in range(SX - 1):
                for j in range(SY_bank - 1):
                    # Side wall
                    v0 = idx(i, j, 0)
                    v1 = idx(i, j, 1)
                    v2 = idx(i+1, j, 1)
                    v3 = idx(i+1, j, 0)
                    indices += [v0, v1, v2, v3]
                    counts.append(4)
                    # Top surface
                    v0 = idx(i, j, 1)
                    v1 = idx(i, j+1, 1)
                    v2 = idx(i+1, j+1, 1)
                    v3 = idx(i+1, j, 1)
                    indices += [v0, v1, v2, v3]
                    counts.append(4)
                    # Bottom (optional)
                    v0 = idx(i, j, 0)
                    v1 = idx(i+1, j, 0)
                    v2 = idx(i+1, j+1, 0)
                    v3 = idx(i, j+1, 0)
                    indices += [v0, v1, v2, v3]
                    counts.append(4)

            mesh = UsdGeom.Mesh.Define(stage, f"/World/Soil/{seg.name}_bank_{side_name}")
            mesh.CreatePointsAttr(verts)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([(0.45, 0.30, 0.18)])

        # -----------------------------------------------------------------
        # UNDERFLOOR (unchanged, but ensure it connects smoothly)
        # -----------------------------------------------------------------
        verts = []
        indices = []
        counts = []
        under_width = 2.0 * (radius + bank_thickness)

        for i in range(SX):
            along = start_along + (i / (SX - 1)) * (end_along - start_along)
            center = seg.point(along)
            terrain_z = terrain_height_at(seg.terrain_s_at(along))
            bottom_z_fill = terrain_z - underfloor_depth
            bottom_z_fill += 0.005 * math.sin(i * 0.5)   # tiny waviness

            for j in range(SY_under):
                frac = j / (SY_under - 1)
                lateral = (frac - 0.5) * under_width
                side = _norm(np.array([-seg.direction[1], seg.direction[0], 0.0]))
                p = center + side * lateral
                verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(bottom_z_fill)))
                verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(terrain_z)))

        def idx_under(i_, j_, layer):
            return (i_ * SY_under + j_) * 2 + layer

        for i in range(SX - 1):
            for j in range(SY_under - 1):
                v0 = idx_under(i, j, 0)
                v1 = idx_under(i, j, 1)
                v2 = idx_under(i+1, j, 1)
                v3 = idx_under(i+1, j, 0)
                indices += [v0, v1, v2, v3]
                counts.append(4)
                v0 = idx_under(i, j, 1)
                v1 = idx_under(i, j+1, 1)
                v2 = idx_under(i+1, j+1, 1)
                v3 = idx_under(i+1, j, 1)
                indices += [v0, v1, v2, v3]
                counts.append(4)
                v0 = idx_under(i, j, 0)
                v1 = idx_under(i+1, j, 0)
                v2 = idx_under(i+1, j+1, 0)
                v3 = idx_under(i, j+1, 0)
                indices += [v0, v1, v2, v3]
                counts.append(4)

        mesh = UsdGeom.Mesh.Define(stage, f"/World/Soil/{seg.name}_underfill")
        mesh.CreatePointsAttr(verts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([(0.45, 0.30, 0.18)])

    print("[World] Soil mesh built with smooth, non‑boxy embankments")

def _build_circular_cap(stage, seg, prim_path, n_segs=TUNNEL_RINGS):
    """Filled circular disk sealing the far end of a tunnel bore."""
    center = seg.end.copy()

    # centre vertex + ring
    verts = [Gf.Vec3f(float(center[0]), float(center[1]), float(center[2]))]
    for j in range(n_segs):
        a      = (j / n_segs) * 2.0 * math.pi
        offset = seg.right * (TUNNEL_RADIUS * math.cos(a)) + seg.up * (TUNNEL_RADIUS * math.sin(a))
        p      = center + offset
        verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))

    face_counts  = []
    face_indices = []
    for j in range(n_segs):
        nj = (j + 1) % n_segs
        face_indices += [0, j + 1, nj + 1]
        face_counts  += [3]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([(0.47, 0.33, 0.19)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    return mesh

def build_tunnel_end_caps(stage, segments):
    """
    Seals the far (non-entry) end of every child tunnel
    with a filled circular mesh cap + collision.
    """
    stage.DefinePrim("/World/TunnelCaps", "Xform")
    seg_map = {s.name: s for s in segments}

    for name in ["left_tail", "mid_tail", "right_tail"]:
        seg = seg_map[name]
        _build_circular_cap(stage, seg, f"/World/TunnelCaps/{name}_cap")
        print(f"[World] End cap built: {name}")

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
    return mesh

def build_rocks(stage, segments):
    """
    Rocks are visual only: no collision API is applied to them.
    """
    placements = [
        ("root", 5.0,  0.6), ("root", 12.0,  0.5),
        ("left_branch", 4.0,  0.4), ("left_tail", 3.5, -0.5),
        ("mid_branch",  5.0, -0.3), ("mid_tail",  4.0,  0.6),
        ("right_branch",4.0, -0.4), ("right_tail",3.5,  0.5),
    ]

    seg_map = {s.name: s for s in segments}
    rng = np.random.default_rng(7)

    for k, (seg_name, along, lateral) in enumerate(placements):
        seg = seg_map[seg_name]
        center = seg.point(along)
        s_abs = seg.terrain_s_at(along)
        z = terrain_height_at(s_abs) + 0.06

        side = _norm(np.array([-seg.direction[1], seg.direction[0], 0.0], dtype=float))
        pos = np.array([center[0], center[1], z], dtype=float) + side * lateral

        scale_factor = float(rng.uniform(3.0, 4.0))
        sx = 0.12 * scale_factor
        sy = 0.10 * scale_factor
        sz = 0.07 * scale_factor

        max_w = TUNNEL_RADIUS * 0.75
        sx = min(sx, max_w)
        sy = min(sy, max_w)

        cube = UsdGeom.Cube.Define(stage, f"/World/rock_{k}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([(0.28, 0.22, 0.18)])
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

        pos[2] -= sz * 0.25
        xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

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

            for side, lr in [(1.1, 'L'), (-1.1, 'R')]:
                cy = UsdGeom.Cylinder.Define(stage, f"/World/m{idx}{lr}_{int(along*10)}")
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
    Rover visual + physics shell:
      - rigid root on /World/rover
      - hidden collider under the root
      - visual body, wheels, mast, sensor
    Rocks remain visual-only.
    """
    rover = UsdGeom.Xform.Define(stage, "/World/rover")
    rover_prim = rover.GetPrim()

    # NO CollisionAPI on the parent Xform – it has no geometry and would be ignored.
    # UsdPhysics.CollisionAPI.Apply(rover_prim)   <-- removed

    rb = UsdPhysics.RigidBodyAPI.Apply(rover_prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)

    mass_api = UsdPhysics.MassAPI.Apply(rover_prim)
    mass_api.CreateMassAttr(5.0)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(rover_prim)
    physx_rb.CreateLinearDampingAttr(0.1)
    physx_rb.CreateAngularDampingAttr(0.2)
    # Enable CCD to prevent tunnelling through thin floor tiles
    physx_rb.CreateEnableCCDAttr(True)

    rover_xf = UsdGeom.Xformable(rover.GetPrim())
    rover_xf.AddTranslateOp().Set(Gf.Vec3d(1.5, 0.0, terrain_height_at(0.0) + 0.22))
    rover_xf.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

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

    # hidden collider that actually contacts the environment
    collider = UsdGeom.Cube.Define(stage, "/World/rover/collider")
    collider.CreateSizeAttr(1.0)
    collider.CreateDisplayColorAttr([(0.0, 0.0, 0.0)])
    xfc = UsdGeom.Xformable(collider.GetPrim())
    xfc.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BODY_HALF_Z))
    xfc.AddScaleOp().Set(Gf.Vec3d(BODY_HALF_X * 1.05, BODY_HALF_Y * 1.05, BODY_HALF_Z * 1.1))
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    UsdGeom.Imageable(collider.GetPrim()).MakeInvisible()

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
    return rover

# -----------------------------------------------------------------------------
# MOTION CONTROLLER (FIXED: no forced zero vertical velocity)
# -----------------------------------------------------------------------------
class RoverTraversal:
    """
    Deterministic traversal:
      start -> root -> left tunnel -> back to hub
      -> mid tunnel -> back to hub
      -> right tunnel -> back to hub
      -> back to start
    """
    def __init__(self, segments):
        self.seg_map = {s.name: s for s in segments}
        self.actions = []

        self._add_drive("root", False)
        self._add_turn_to("left_branch", False)
        self._add_drive("left_branch", False)
        self._add_drive("left_tail", False)
        self._add_drive("left_tail", True)
        self._add_drive("left_branch", True)
        self._add_turn_to("mid_branch", False)
        self._add_drive("mid_branch", False)
        self._add_drive("mid_tail", False)
        self._add_drive("mid_tail", True)
        self._add_drive("mid_branch", True)
        self._add_turn_to("right_branch", False)
        self._add_drive("right_branch", False)
        self._add_drive("right_tail", False)
        self._add_drive("right_tail", True)
        self._add_drive("right_branch", True)
        self._add_turn_to("root", True)
        self._add_drive("root", True)

        self.idx = 0
        self.local_s = 0.0
        self.drive_speed = 0.32
        self.turn_rate = 0.9
        self.turn_tolerance = 0.03

    def _add_drive(self, seg_name, reverse):
        self.actions.append({"kind": "drive", "segment": seg_name, "reverse": reverse})

    def _add_turn_to(self, seg_name, reverse):
        seg = self.seg_map[seg_name]
        direction = (-seg.direction) if reverse else seg.direction
        target_yaw = math.atan2(float(direction[1]), float(direction[0]))
        self.actions.append({"kind": "turn", "target_yaw": target_yaw})

    def current_action(self):
        return self.actions[self.idx]

    def is_done(self):
        return self.idx >= len(self.actions)

    def step(self, rover: RigidPrim, dt: float):
        if self.is_done():
            rover.set_linear_velocity(np.zeros(3, dtype=np.float32))
            rover.set_angular_velocity(np.zeros(3, dtype=np.float32))
            return True

        action = self.current_action()

        if action["kind"] == "drive":
            seg = self.seg_map[action["segment"]]
            reverse = bool(action["reverse"])
            direction = (-seg.direction) if reverse else seg.direction

            # Get current velocity, keep vertical component unchanged (gravity will act)
            vel = rover.get_linear_velocity()
            # Set only X/Y to the desired direction, preserve Z
            vel[0] = direction[0] * self.drive_speed
            vel[1] = direction[1] * self.drive_speed
            rover.set_linear_velocity(vel)
            rover.set_angular_velocity(np.zeros(3, dtype=np.float32))

            # Update progress along the segment (for internal state, not physics)
            s_eff = seg.length - self.local_s if reverse else self.local_s
            self.local_s += self.drive_speed * dt
            if self.local_s >= seg.length:
                self.idx += 1
                self.local_s = 0.0
                # Stop briefly when transitioning to a turn
                rover.set_linear_velocity(np.zeros(3, dtype=np.float32))
                rover.set_angular_velocity(np.zeros(3, dtype=np.float32))
            return False

        # turn-in-place at intersection
        pos, quat = rover.get_world_pose()
        yaw = yaw_from_quat_wxyz(quat)
        target = float(action["target_yaw"])
        err = math.atan2(math.sin(target - yaw), math.cos(target - yaw))

        if abs(err) <= self.turn_tolerance:
            # Snap exactly once we're aligned, then advance
            rover.set_linear_velocity(np.zeros(3, dtype=np.float32))
            rover.set_angular_velocity(np.zeros(3, dtype=np.float32))
            self.idx += 1
            self.local_s = 0.0
            return False

        rover.set_linear_velocity(np.zeros(3, dtype=np.float32))
        rover.set_angular_velocity(np.array([0.0, 0.0, np.sign(err) * self.turn_rate], dtype=np.float32))
        return False

# -----------------------------------------------------------------------------
# STAGE SETUP
# -----------------------------------------------------------------------------
def configure_stage(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

    # Physics scene
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    # Enable CCD on the scene (optional but helpful)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/physicsScene"))
    physx_scene.CreateEnableCCDAttr(True)

    # NOTE: solver iteration attributes are not available in this USD build;
    # the defaults work fine. Remove the following lines if present:
    # scene.CreateSolverPositionIterCountAttr(8)
    # scene.CreateSolverVelocityIterCountAttr(4)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("\n" + "=" * 72)
    print("  CANARY ROVER — Isaac Sim 5.1.0  Branching Tunnel Physics Demo")
    print("  root -> 3 branches -> tails")
    print("  ground contact enabled; rocks are visual-only")
    print("=" * 72 + "\n")

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    configure_stage(stage)

    # build world; no default ground plane because we have our own tunnel floor
    world = World(stage_units_in_meters=1.0)

    segments = build_segments()
    print(f"[World] Built {len(segments)} segments")

    # Geometry
    for seg in segments:
        build_tunnel(stage, seg, f"/World/Tunnels/{seg.name}")
        build_floor(stage, seg, f"/World/Floors/{seg.name}")

    build_tunnel_end_caps(stage, segments)
    build_soil_mesh(stage, segments)
    build_hub_patch(stage)
    build_rocks(stage, segments)
    build_markers(stage, segments)
    build_finish(stage, segments)
    setup_lights(stage)

    build_rover_proxy(stage)

    # load physics-aware rover wrapper
    world.reset()
    rover = RigidPrim("/World/rover")
    rover.initialize()

    # start pose
    start_pos = np.array([0.5, 0.0, terrain_height_at(0.5) + 0.22], dtype=np.float32)
    start_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rover.set_world_pose(position=start_pos, orientation=start_quat)
    rover.set_linear_velocity(np.zeros(3, dtype=np.float32))
    rover.set_angular_velocity(np.zeros(3, dtype=np.float32))

    controller = RoverTraversal(segments)
    world.play()

    step = 0
    dt = 1.0 / 60.0
    mission_complete = False

    try:
        while simulation_app.is_running():
            # command motion before stepping physics
            done = controller.step(rover, dt)
            step += 1

            # follow camera
            if set_camera_view is not None and step % 8 == 0:
                pos, quat = rover.get_world_pose()
                yaw = yaw_from_quat_wxyz(quat)
                forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
                eye = pos - forward * 3.0 + np.array([0.0, 0.0, 1.5], dtype=np.float32)
                target = pos + forward * 4.0
                set_camera_view(eye=eye, target=target)

            if step % 60 == 0:
                pos, quat = rover.get_world_pose()
                yaw = yaw_from_quat_wxyz(quat)
                print(
                    f"[Step {step:5d}] rover=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:5.2f}) "
                    f"yaw={math.degrees(yaw):+6.1f}° "
                    f"action={controller.idx:02d}/{len(controller.actions)}"
                )

            if done and not mission_complete:
                print("Traversal complete")
                mission_complete = True

            world.step(render=True)

        if mission_complete:
            while simulation_app.is_running():
                world.step(render=True)

    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()

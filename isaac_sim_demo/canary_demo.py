from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import math
import numpy as np
import omni, omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, Sdf, PhysxSchema

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.viewports import set_camera_view

ROVER_USD = "/home/kunal/isaac/canary_demo/robot/vleg_rover/vleg_rover.usd"

# ---------------------------------------------------------
# TUNNEL DIMENSIONS (SQUARE OUTER, TALL OVAL INNER)
# ---------------------------------------------------------
OUTER_WIDTH = 4.0
OUTER_HEIGHT = 4.5
INNER_WIDTH = 3.0      
INNER_HEIGHT = 4.0     
TUNNEL_AXIS_Z = 0.0

# Junction dimensions based on outer block width
J_HALF = OUTER_WIDTH / 2.0  # 2.0 meters

TUNNEL_RINGS = 16
TUNNEL_SEGS_PER_M = 3
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

def terrain_height_at(pt):
    # Base terrain height based on distance from origin
    dist = np.linalg.norm(pt[:2])
    idx = max(0.0, min(dist, WORLD_LENGTH - 0.01)) / WORLD_LENGTH * (TERRAIN_SAMPLES - 1)
    i0, i1 = int(idx), min(int(idx) + 1, TERRAIN_SAMPLES - 1)
    t = idx - i0
    return float(_terrain_h[i0] * (1 - t) + _terrain_h[i1] * t)

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
    R = np.column_stack([fwd, right, up])
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))

def yaw_from_quat_wxyz(q):
    w, x, y, z = map(float, q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

# ---------------------------------------------------------
# EXPLICIT 90-DEGREE SEGMENT LOGIC
# Separates Pathing (for the rover) from Rendering (for the meshes)
# ---------------------------------------------------------
class Segment:
    def __init__(self, name, p_start, p_dir, p_length, r_start, r_length):
        self.name = name
        
        # Physics / Pathing data (Meeting perfectly at the junction center)
        self.p_start = np.array(p_start, dtype=float)
        self.direction = _norm(np.array(p_dir, dtype=float))
        self.p_length = float(p_length)
        self.p_end = self.p_start + self.direction * self.p_length
        self.right, self.up, _ = _local_frame(self.direction)
        
        # Rendering data (Stopping at the edges of the junction to prevent overlap)
        self.r_start = np.array(r_start, dtype=float)
        self.r_length = float(r_length)

    def path_point(self, along: float) -> np.ndarray:
        return self.p_start + self.direction * along

    def render_point(self, along: float) -> np.ndarray:
        return self.r_start + self.direction * along

    def quat(self) -> Gf.Quatf:
        return _quat_from_axes(self.direction, self.right, self.up)

def build_segments():
    segments = []
    JX, JY = 15.0 + J_HALF, 0.0 # Center of junction is at X=17, Y=0

    # 1. Root (Main) Tunnel
    segments.append(Segment("root", 
        p_start=[0, 0, 0], p_dir=[1, 0, 0], p_length=JX,
        r_start=[0, 0, 0], r_length=JX - J_HALF
    ))
    
    # 2. Middle Branch (Straight)
    segments.append(Segment("mid_branch",
        p_start=[JX, JY, 0], p_dir=[1, 0, 0], p_length=15.0,
        r_start=[JX + J_HALF, JY, 0], r_length=15.0 - J_HALF
    ))
    segments.append(Segment("mid_tail",
        p_start=[JX + 15.0, JY, 0], p_dir=[1, 0, 0], p_length=10.0,
        r_start=[JX + 15.0, JY, 0], r_length=10.0
    ))

    # 3. Left Branch (Exactly +90 degrees)
    segments.append(Segment("left_branch",
        p_start=[JX, JY, 0], p_dir=[0, 1, 0], p_length=15.0,
        r_start=[JX, JY + J_HALF, 0], r_length=15.0 - J_HALF
    ))
    segments.append(Segment("left_tail",
        p_start=[JX, JY + 15.0, 0], p_dir=[0, 1, 0], p_length=10.0,
        r_start=[JX, JY + 15.0, 0], r_length=10.0
    ))

    # 4. Right Branch (Exactly -90 degrees)
    segments.append(Segment("right_branch",
        p_start=[JX, JY, 0], p_dir=[0, -1, 0], p_length=15.0,
        r_start=[JX, JY - J_HALF, 0], r_length=15.0 - J_HALF
    ))
    segments.append(Segment("right_tail",
        p_start=[JX, JY - 15.0, 0], p_dir=[0, -1, 0], p_length=10.0,
        r_start=[JX, JY - 15.0, 0], r_length=10.0
    ))
    
    return segments

def build_tunnel_mesh(stage, seg: Segment, prim_path: str):
    """ Builds the Square Outer / Oval Inner shell for a segment """
    if seg.r_length <= 0: return
    
    n_long = max(4, int(math.ceil(seg.r_length * TUNNEL_SEGS_PER_M)))
    verts, f_counts, f_indices = [], [], []

    # Define the 2D Cross Section
    points_2d = []
    points_2d.append((-OUTER_WIDTH/2, 0.0))
    points_2d.append((-OUTER_WIDTH/2, OUTER_HEIGHT))
    points_2d.append((OUTER_WIDTH/2, OUTER_HEIGHT))
    points_2d.append((OUTER_WIDTH/2, 0.0))
    
    for i in range(TUNNEL_RINGS + 1):
        t = math.pi * (1.0 - (i / TUNNEL_RINGS)) # Sweeps from Pi down to 0
        x = (INNER_WIDTH / 2.0) * math.cos(t)
        z = INNER_HEIGHT * math.sin(t)
        points_2d.append((x, z))
        
    num_pts = len(points_2d)

    for i in range(n_long + 1):
        along = i * (seg.r_length / n_long)
        center = seg.render_point(along)
        base_z = terrain_height_at(center)
        
        for px, pz in points_2d:
            dx = seg.right[0] * px
            dy = seg.right[1] * px
            verts.append(Gf.Vec3f(float(center[0]+dx), float(center[1]+dy), float(base_z+pz)))

    for i in range(n_long):
        for j in range(num_pts - 1):
            v0 = i * num_pts + j
            v1 = i * num_pts + (j + 1)
            v2 = (i + 1) * num_pts + j
            v3 = (i + 1) * num_pts + (j + 1)
            f_indices += [v0, v2, v1, v1, v2, v3]
            f_counts += [3, 3]

    color = (0.52, 0.36, 0.22) if "root" in seg.name else (0.47, 0.33, 0.19)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(f_counts)
    mesh.CreateFaceVertexIndicesAttr(f_indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([color])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")

def build_bumpy_floor(stage, seg: Segment, prim_path: str):
    """ Builds the collision-enabled bumpy floor for a segment """
    if seg.r_length <= 0: return
    
    SX, SY = max(10, int(seg.r_length * 4)), 16
    verts, indices, counts = [], [], []

    for i in range(SX):
        along = (i / (SX - 1)) * seg.r_length
        center = seg.render_point(along)
        base_z = terrain_height_at(center)
        
        for j in range(SY):
            lateral = ((j / (SY - 1)) - 0.5) * INNER_WIDTH
            side = seg.right
            p = center + side * lateral
            
            # Complex wave ensuring slopes are < 20 degrees
            bump = 0.06 * math.sin(along * 4.0 + lateral * 2.0) + 0.03 * math.cos(along * 8.0 - lateral * 4.0)
            z = base_z + bump
            verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(z)))

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
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([(0.32, 0.20, 0.10)])

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    physx_coll.CreateContactOffsetAttr(0.01)
    physx_coll.CreateRestOffsetAttr(0.001)

def build_junction_caps(stage):
    """ Creates a perfect flat square ceiling and floor to seal the 4-way intersection """
    JX, JY = 15.0 + J_HALF, 0.0
    base_z = terrain_height_at([JX, JY, 0])

    floor = UsdGeom.Cube.Define(stage, "/World/JunctionFloor")
    floor.CreateSizeAttr(1.0)
    floor.CreateDisplayColorAttr([(0.32, 0.20, 0.10)])
    xf_f = UsdGeom.Xformable(floor.GetPrim())
    xf_f.AddTranslateOp().Set(Gf.Vec3d(JX, JY, base_z - 0.5))
    xf_f.AddScaleOp().Set(Gf.Vec3f(OUTER_WIDTH, OUTER_WIDTH, 1.0))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    roof = UsdGeom.Cube.Define(stage, "/World/JunctionRoof")
    roof.CreateSizeAttr(1.0)
    roof.CreateDisplayColorAttr([(0.47, 0.33, 0.19)])
    xf_r = UsdGeom.Xformable(roof.GetPrim())
    xf_r.AddTranslateOp().Set(Gf.Vec3d(JX, JY, base_z + OUTER_HEIGHT + 0.5))
    xf_r.AddScaleOp().Set(Gf.Vec3f(OUTER_WIDTH, OUTER_WIDTH, 1.0))

def build_solid_soil_blocks(stage):
    """ Places 4 massive rock cubes in the exterior corners to 100% seal the environment """
    stage.DefinePrim("/World/Soil", "Xform")
    JX, JY = 15.0 + J_HALF, 0.0
    color = (0.25, 0.18, 0.12) 
    
    def make_block(name, cx, cy, cz, sx, sy, sz):
        cube = UsdGeom.Cube.Define(stage, f"/World/Soil/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([color])
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    S = 40.0 
    H = 10.0 
    Z = terrain_height_at([JX, JY, 0]) + OUTER_HEIGHT / 2.0

    make_block("q1_front_left",  JX + J_HALF + S/2, JY + J_HALF + S/2, Z, S, S, H)
    make_block("q2_back_left",   JX - J_HALF - S/2, JY + J_HALF + S/2, Z, S, S, H)
    make_block("q3_back_right",  JX - J_HALF - S/2, JY - J_HALF - S/2, Z, S, S, H)
    make_block("q4_front_right", JX + J_HALF + S/2, JY - J_HALF - S/2, Z, S, S, H)
    make_block("entrance_seal",  -S/2, 0.0, Z, S, S, H)

def build_tunnel_end_caps(stage, segments):
    stage.DefinePrim("/World/TunnelCaps", "Xform")
    for seg in segments:
        if "tail" not in seg.name: continue
        
        center = seg.render_point(seg.r_length)
        base_z = terrain_height_at(center)
        
        verts = [
            Gf.Vec3f(float(center[0] - seg.right[0]*OUTER_WIDTH/2), float(center[1] - seg.right[1]*OUTER_WIDTH/2), float(base_z)),
            Gf.Vec3f(float(center[0] - seg.right[0]*OUTER_WIDTH/2), float(center[1] - seg.right[1]*OUTER_WIDTH/2), float(base_z + OUTER_HEIGHT)),
            Gf.Vec3f(float(center[0] + seg.right[0]*OUTER_WIDTH/2), float(center[1] + seg.right[1]*OUTER_WIDTH/2), float(base_z + OUTER_HEIGHT)),
            Gf.Vec3f(float(center[0] + seg.right[0]*OUTER_WIDTH/2), float(center[1] + seg.right[1]*OUTER_WIDTH/2), float(base_z))
        ]

        mesh = UsdGeom.Mesh.Define(stage, f"/World/TunnelCaps/{seg.name}_cap")
        mesh.CreatePointsAttr(verts)
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([(0.47, 0.33, 0.19)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")

def build_markers(stage, segments):
    for idx, seg in enumerate(segments):
        if "tail" not in seg.name and "branch" not in seg.name and seg.name != "root": continue
        for along in np.linspace(4.0, max(4.0, seg.r_length - 2.0), 3):
            center = seg.render_point(along)
            z = terrain_height_at(center) + 0.5
            for side, lr in [(INNER_WIDTH/2 - 0.2, 'L'), (-INNER_WIDTH/2 + 0.2, 'R')]:
                cy = UsdGeom.Cylinder.Define(stage, f"/World/m{idx}{lr}_{int(along*10)}")
                cy.CreateRadiusAttr(0.04); cy.CreateHeightAttr(1.0); cy.CreateDisplayColorAttr([(1.0, 0.55, 0.0)])
                pos = center + seg.right * side
                xf = UsdGeom.Xformable(cy.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(z)))
                xf.AddOrientOp().Set(seg.quat())
                xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

def build_finish(stage, segments):
    for seg in segments:
        if "tail" not in seg.name: continue
        c = UsdGeom.Cube.Define(stage, f"/World/finish_{seg.name}")
        c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])
        xf = UsdGeom.Xformable(c.GetPrim())
        
        end_pt = seg.render_point(seg.r_length)
        z_floor = terrain_height_at(end_pt) + 0.02
        xf.AddTranslateOp().Set(Gf.Vec3d(float(end_pt[0]), float(end_pt[1]), float(z_floor)))
        xf.AddOrientOp().Set(seg.quat())
        xf.AddScaleOp().Set(Gf.Vec3f(0.4, INNER_WIDTH - 0.2, 0.02))

def setup_lights(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(2500.0); dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.93))
    
    JX = 15.0 + J_HALF
    light_positions = [
        (4.0, 0.0, 3.0),
        (12.0, 0.0, 3.0),
        (JX, 0.0, 3.0),         
        (JX, 8.0, 3.0),         
        (JX, -8.0, 3.0),        
        (JX + 8.0, 0.0, 3.2),   
    ]

    for i, (x, y, z) in enumerate(light_positions):
        light = UsdLux.SphereLight.Define(stage, f"/World/KeyLight_{i}")
        light.CreateIntensityAttr(5000.0); light.CreateRadiusAttr(0.15); light.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.55))
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(z)))
        xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

def configure_stage(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

class RoverTraversal:
    def __init__(self, segments):
        self.seg_map = {s.name: s for s in segments}
        self.actions = []
        
        self._add_drive("root", False)
        
        self._add_turn_to("left_branch", False)
        self._add_drive("left_branch", False)
        self._add_drive("left_tail", False)
        self._add_turn_to("left_tail", True) 
        self._add_drive("left_tail", True)
        self._add_drive("left_branch", True)
        
        self._add_turn_to("right_branch", False)
        self._add_drive("right_branch", False)
        self._add_drive("right_tail", False)
        self._add_turn_to("right_tail", True) 
        self._add_drive("right_tail", True)
        self._add_drive("right_branch", True)

        self._add_turn_to("mid_branch", False)
        self._add_drive("mid_branch", False)
        self._add_drive("mid_tail", False)
        self._add_turn_to("mid_tail", True) 
        self._add_drive("mid_tail", True)
        self._add_drive("mid_branch", True)

        self._add_turn_to("root", True) 
        self._add_drive("root", True)
        
        self.idx, self.local_s = 0, 0.0
        
        # REQUESTED SPEEDS
        self.drive_speed = 1.0 
        self.turn_rate = 1.25
        self.turn_tolerance = 0.03

    def _add_drive(self, seg_name, reverse):
        self.actions.append({"kind": "drive", "segment": seg_name, "reverse": reverse})

    def _add_turn_to(self, seg_name, reverse):
        seg = self.seg_map[seg_name]
        self.actions.append({"kind": "turn", "target_yaw": math.atan2(float(((-seg.direction) if reverse else seg.direction)[1]), float(((-seg.direction) if reverse else seg.direction)[0]))})

    def step(self, rover: Articulation, dt: float):
        if self.idx >= len(self.actions): return True
        action, (pos, quat) = self.actions[self.idx], rover.get_world_pose()
        yaw = yaw_from_quat_wxyz(quat)

        if action["kind"] == "drive":
            seg, reverse = self.seg_map[action["segment"]], bool(action["reverse"])
            
            drive_dist = seg.p_length
            if "tail" in seg.name:
                drive_dist = seg.p_length - 4.0

            self.local_s += self.drive_speed * dt
            is_done_segment = False
            if self.local_s >= drive_dist:
                self.local_s, is_done_segment = drive_dist, True

            s_eff = (drive_dist - self.local_s) if reverse else self.local_s
            target_pos = seg.path_point(s_eff)
            target_pos[2] = terrain_height_at(target_pos) + 0.35 

            target_yaw = math.atan2(float(((-seg.direction) if reverse else seg.direction)[1]), float(((-seg.direction) if reverse else seg.direction)[0]))
            target_quat = np.array([math.cos(target_yaw * 0.5), 0.0, 0.0, math.sin(target_yaw * 0.5)], dtype=np.float32)
            rover.set_world_pose(position=target_pos.astype(np.float32), orientation=target_quat)
            
            if is_done_segment:
                self.idx, self.local_s = self.idx + 1, 0.0
            return False

        elif action["kind"] == "turn":
            err = math.atan2(math.sin(float(action["target_yaw"]) - yaw), math.cos(float(action["target_yaw"]) - yaw))
            if abs(err) <= self.turn_tolerance:
                self.idx, self.local_s = self.idx + 1, 0.0
                return False
            
            new_yaw = yaw + np.sign(err) * self.turn_rate * dt
            rover.set_world_pose(position=pos.astype(np.float32), orientation=np.array([math.cos(new_yaw * 0.5), 0.0, 0.0, math.sin(new_yaw * 0.5)], dtype=np.float32))
            return False

def main():
    print("\n" + "=" * 72)
    print("  CANARY ROVER — Isaac Sim 5.1.0  Perfect 90-Deg Junction")
    print("=" * 72 + "\n")

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    configure_stage(stage)

    segments = build_segments()
    for seg in segments:
        build_tunnel_mesh(stage, seg, f"/World/Tunnels/{seg.name}")
        build_bumpy_floor(stage, seg, f"/World/Floors/{seg.name}")

    build_junction_caps(stage)
    build_solid_soil_blocks(stage)
    build_tunnel_end_caps(stage, segments)
    build_markers(stage, segments)
    build_finish(stage, segments)
    setup_lights(stage)

    add_reference_to_stage(usd_path=ROVER_USD, prim_path="/World/rover")
    world.reset()
    
    rover = Articulation(prim_path="/World/rover")
    rover.initialize()
    rover.disable_gravity()

    start_pos = np.array([0.5, 0.0, terrain_height_at([0.5, 0.0, 0.0]) + 0.35], dtype=np.float32)
    start_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rover.set_world_pose(position=start_pos, orientation=start_quat)
    
    controller = RoverTraversal(segments)
    world.play()

    step = 0
    dt = 1.0 / 60.0
    mission_complete = False

    while simulation_app.is_running():
        world.step(render=True)
        
        # PLAY/PAUSE/STOP LOGIC 
        if world.is_stopped():
            controller = RoverTraversal(segments) 
            step = 0
            mission_complete = False
            continue 

        if world.is_playing():
            rover.set_linear_velocity(np.zeros(3, dtype=np.float32))
            rover.set_angular_velocity(np.zeros(3, dtype=np.float32))

            done = controller.step(rover, dt)
            step += 1

            if step % 8 == 0:
                pos, quat = rover.get_world_pose()
                yaw = yaw_from_quat_wxyz(quat)
                forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
                set_camera_view(eye=pos - forward * 3.0 + np.array([0.0, 0.0, 1.5], dtype=np.float32), target=pos + forward * 4.0)

            if step % 60 == 0:
                pos, quat = rover.get_world_pose()
                print(f"[Step {step:5d}] rover=({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:5.2f}) yaw={math.degrees(yaw_from_quat_wxyz(quat)):+6.1f}° action={controller.idx:02d}/{len(controller.actions)}")

            if done and not mission_complete:
                print("\n*** Traversal complete! Press Stop in the UI to reset the rover. ***\n")
                mission_complete = True
                world.pause() 

    simulation_app.close()

if __name__ == "__main__":
    main()
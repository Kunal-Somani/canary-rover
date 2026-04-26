import os
for key in list(os.environ.keys()):
    if 'ROS' in key or 'AMENT' in key or 'RMW' in key:
        del os.environ[key]

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.sensors.physx")
enable_extension("isaacsim.sensors.rtx") 
enable_extension("omni.isaac.sensor")

import math
import numpy as np
import omni, omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, Sdf, PhysxSchema

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.viewports import set_camera_view

ROVER_USD = "/home/kunal/isaac/canary_demo/robot/vleg_rover/vleg_rover.usd"

OUTER_WIDTH  = 4.0
OUTER_HEIGHT = 4.5
INNER_WIDTH  = 3.0
INNER_HEIGHT = 4.0
J_HALF       = OUTER_WIDTH / 2.0

TUNNEL_RINGS    = 16
TUNNEL_SEGS_PER_M = 3
WORLD_LENGTH    = 60.0

np.random.seed(42)
TERRAIN_SAMPLES = 300
_s_samples = np.linspace(0.0, WORLD_LENGTH, TERRAIN_SAMPLES)
_terrain_h = (
    0.060 * np.sin(_s_samples * 0.25) +
    0.040 * np.cos(_s_samples * 0.8  + 0.5) +
    0.025 * np.sin(_s_samples * 2.1  + 1.2) +
    0.015 * np.sin(_s_samples * 4.5  + 0.3) +
    0.008 * np.random.normal(size=TERRAIN_SAMPLES)
)

def terrain_height_at(pt):
    dist = np.linalg.norm(pt[:2])
    idx  = max(0.0, min(dist, WORLD_LENGTH - 0.01)) / WORLD_LENGTH * (TERRAIN_SAMPLES - 1)
    i0, i1 = int(idx), min(int(idx) + 1, TERRAIN_SAMPLES - 1)
    t = idx - i0
    return float(_terrain_h[i0] * (1 - t) + _terrain_h[i1] * t)

def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v.copy()

def _local_frame(fwd):
    fwd = _norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=float))
    right = _norm(right)
    up = _norm(np.cross(right, fwd))
    return right, up, fwd

def _quat_from_axes(fwd, right, up):
    R = np.column_stack([fwd, right, up])
    t = float(R[0,0] + R[1,1] + R[2,2])
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        w,x,y,z = 0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w,x,y,z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w,x,y,z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w,x,y,z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))

def yaw_from_quat_wxyz(q):
    w, x, y, z = map(float, q)
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

class Segment:
    def __init__(self, name, p_start, p_dir, p_length, r_start, r_length):
        self.name      = name
        self.p_start   = np.array(p_start, dtype=float)
        self.direction = _norm(np.array(p_dir, dtype=float))
        self.p_length  = float(p_length)
        self.p_end     = self.p_start + self.direction * self.p_length
        self.right, self.up, _ = _local_frame(self.direction)
        self.r_start   = np.array(r_start, dtype=float)
        self.r_length  = float(r_length)

    def path_point(self, along):
        return self.p_start + self.direction * along

    def render_point(self, along):
        return self.r_start + self.direction * along

    def quat(self):
        return _quat_from_axes(self.direction, self.right, self.up)

def build_segments():
    JX, JY = 15.0 + J_HALF, 0.0
    return [
        Segment("root",         [0,0,0],      [1,0,0],  JX,   [0,0,0],           JX-J_HALF),
        Segment("mid_branch",   [JX,JY,0],    [1,0,0],  15.0, [JX+J_HALF,JY,0],  15.0-J_HALF),
        Segment("mid_tail",     [JX+15,JY,0], [1,0,0],  10.0, [JX+15,JY,0],      10.0),
        Segment("left_branch",  [JX,JY,0],    [0,1,0],  15.0, [JX,JY+J_HALF,0],  15.0-J_HALF),
        Segment("left_tail",    [JX,JY+15,0], [0,1,0],  10.0, [JX,JY+15,0],      10.0),
        Segment("right_branch", [JX,JY,0],    [0,-1,0], 15.0, [JX,JY-J_HALF,0],  15.0-J_HALF),
        Segment("right_tail",   [JX,JY-15,0], [0,-1,0], 10.0, [JX,JY-15,0],      10.0),
    ]

def build_tunnel_mesh(stage, seg, prim_path):
    if seg.r_length <= 0: return
    n_long = max(4, int(math.ceil(seg.r_length * TUNNEL_SEGS_PER_M)))
    verts, f_counts, f_indices = [], [], []
    points_2d = [
        (-OUTER_WIDTH/2, 0.0), (-OUTER_WIDTH/2, OUTER_HEIGHT),
        ( OUTER_WIDTH/2, OUTER_HEIGHT), ( OUTER_WIDTH/2, 0.0)
    ]
    for i in range(TUNNEL_RINGS + 1):
        t = math.pi * (1.0 - i/TUNNEL_RINGS)
        points_2d.append(((INNER_WIDTH/2)*math.cos(t), INNER_HEIGHT*math.sin(t)))
    num_pts = len(points_2d)
    for i in range(n_long + 1):
        along  = i * (seg.r_length / n_long)
        center = seg.render_point(along)
        base_z = terrain_height_at(center)
        for px, pz in points_2d:
            verts.append(Gf.Vec3f(
                float(center[0] + seg.right[0]*px),
                float(center[1] + seg.right[1]*px),
                float(base_z + pz)
            ))
    for i in range(n_long):
        for j in range(num_pts - 1):
            v0=i*num_pts+j; v1=i*num_pts+(j+1); v2=(i+1)*num_pts+j; v3=(i+1)*num_pts+(j+1)
            f_indices += [v0,v2,v1, v1,v2,v3]; f_counts += [3,3]
    color = (0.52,0.36,0.22) if "root" in seg.name else (0.47,0.33,0.19)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(f_counts)
    mesh.CreateFaceVertexIndicesAttr(f_indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([color])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")

def build_bumpy_floor(stage, seg, prim_path):
    if seg.r_length <= 0: return
    SX, SY = max(10, int(seg.r_length * 4)), 16
    verts, indices, counts = [], [], []
    for i in range(SX):
        along  = (i/(SX-1)) * seg.r_length
        center = seg.render_point(along)
        base_z = terrain_height_at(center)
        for j in range(SY):
            lateral = ((j/(SY-1)) - 0.5) * INNER_WIDTH
            p  = center + seg.right * lateral
            bmp = 0.06*math.sin(along*4.0+lateral*2.0) + 0.03*math.cos(along*8.0-lateral*4.0)
            verts.append(Gf.Vec3f(float(p[0]), float(p[1]), float(base_z+bmp)))
    for i in range(SX-1):
        for j in range(SY-1):
            v0=i*SY+j; v1=i*SY+(j+1); v2=(i+1)*SY+j; v3=(i+1)*SY+(j+1)
            indices += [v0,v1,v3,v2]; counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([(0.32,0.20,0.10)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    physx_coll.CreateContactOffsetAttr(0.01)
    physx_coll.CreateRestOffsetAttr(0.001)

def build_junction_caps(stage):
    JX, JY = 15.0+J_HALF, 0.0
    base_z = terrain_height_at([JX,JY,0])
    floor = UsdGeom.Cube.Define(stage, "/World/JunctionFloor")
    floor.CreateSizeAttr(1.0); floor.CreateDisplayColorAttr([(0.32,0.20,0.10)])
    xf = UsdGeom.Xformable(floor.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(JX,JY,base_z-0.5))
    xf.AddScaleOp().Set(Gf.Vec3f(OUTER_WIDTH,OUTER_WIDTH,1.0))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    roof = UsdGeom.Cube.Define(stage, "/World/JunctionRoof")
    roof.CreateSizeAttr(1.0); roof.CreateDisplayColorAttr([(0.47,0.33,0.19)])
    xf = UsdGeom.Xformable(roof.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(JX,JY,base_z+OUTER_HEIGHT+0.5))
    xf.AddScaleOp().Set(Gf.Vec3f(OUTER_WIDTH,OUTER_WIDTH,1.0))

def build_solid_soil_blocks(stage):
    stage.DefinePrim("/World/Soil", "Xform")
    JX, JY = 15.0+J_HALF, 0.0
    color = (0.25,0.18,0.12)
    def make_block(name, cx, cy, cz, sx, sy, sz):
        cube = UsdGeom.Cube.Define(stage, f"/World/Soil/{name}")
        cube.CreateSizeAttr(1.0); cube.CreateDisplayColorAttr([color])
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(cx,cy,cz))
        xf.AddScaleOp().Set(Gf.Vec3f(sx,sy,sz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    
    H = OUTER_HEIGHT * 0.75
    Z = terrain_height_at([JX,JY,0]) + (H / 2.0)
    S = 40.0
    
    make_block("q1_front_left",  JX+J_HALF+S/2,  JY+J_HALF+S/2,  Z, S,S,H)
    make_block("q2_back_left",   JX-J_HALF-S/2,  JY+J_HALF+S/2,  Z, S,S,H)
    make_block("q3_back_right",  JX-J_HALF-S/2,  JY-J_HALF-S/2,  Z, S,S,H)
    make_block("q4_front_right", JX+J_HALF+S/2,  JY-J_HALF-S/2,  Z, S,S,H)

def build_tunnel_end_caps(stage, segments):
    stage.DefinePrim("/World/TunnelCaps", "Xform")
    for seg in segments:
        if "tail" not in seg.name: continue
        center = seg.render_point(seg.r_length)
        base_z = terrain_height_at(center)
        hw = OUTER_WIDTH/2
        verts = [
            Gf.Vec3f(float(center[0]-seg.right[0]*hw), float(center[1]-seg.right[1]*hw), float(base_z)),
            Gf.Vec3f(float(center[0]-seg.right[0]*hw), float(center[1]-seg.right[1]*hw), float(base_z+OUTER_HEIGHT)),
            Gf.Vec3f(float(center[0]+seg.right[0]*hw), float(center[1]+seg.right[1]*hw), float(base_z+OUTER_HEIGHT)),
            Gf.Vec3f(float(center[0]+seg.right[0]*hw), float(center[1]+seg.right[1]*hw), float(base_z)),
        ]
        mesh = UsdGeom.Mesh.Define(stage, f"/World/TunnelCaps/{seg.name}_cap")
        mesh.CreatePointsAttr(verts)
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([(0.47,0.33,0.19)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")

def build_markers(stage, segments):
    for idx, seg in enumerate(segments):
        if "tail" not in seg.name and "branch" not in seg.name and seg.name != "root": continue
        for along in np.linspace(4.0, max(4.0, seg.r_length-2.0), 3):
            center = seg.render_point(along)
            z = terrain_height_at(center) + 0.5
            for side, lr in [(INNER_WIDTH/2-0.2,'L'), (-INNER_WIDTH/2+0.2,'R')]:
                cy = UsdGeom.Cylinder.Define(stage, f"/World/m{idx}{lr}_{int(along*10)}")
                cy.CreateRadiusAttr(0.04); cy.CreateHeightAttr(1.0)
                cy.CreateDisplayColorAttr([(1.0,0.55,0.0)])
                pos = center + seg.right * side
                xf = UsdGeom.Xformable(cy.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]),float(pos[1]),float(z)))
                xf.AddOrientOp().Set(seg.quat())

def build_finish(stage, segments):
    for seg in segments:
        if "tail" not in seg.name: continue
        c = UsdGeom.Cube.Define(stage, f"/World/finish_{seg.name}")
        c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([(1.0,0.0,0.0)])
        xf = UsdGeom.Xformable(c.GetPrim())
        end_pt = seg.render_point(seg.r_length)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(end_pt[0]),float(end_pt[1]),float(terrain_height_at(end_pt)+0.02)))
        xf.AddOrientOp().Set(seg.quat())
        xf.AddScaleOp().Set(Gf.Vec3f(0.4, INNER_WIDTH-0.2, 0.02))

def build_rocks(stage):
    stage.DefinePrim("/World/Rocks", "Xform")
    
    r1_x = 18.0
    r1_y = 12.0
    r1_z = terrain_height_at([r1_x, r1_y, 0]) + 0.4
    
    rock1 = UsdGeom.Sphere.Define(stage, "/World/Rocks/Rock1")
    rock1.CreateRadiusAttr(1.0)
    rock1.CreateDisplayColorAttr([(0.15, 0.15, 0.15)]) 
    xf1 = UsdGeom.Xformable(rock1.GetPrim())
    xf1.AddTranslateOp().Set(Gf.Vec3d(r1_x, r1_y, r1_z))
    xf1.AddScaleOp().Set(Gf.Vec3f(0.5, 0.8, 0.6))
    UsdPhysics.CollisionAPI.Apply(rock1.GetPrim())
    
    r2_x = 28.0
    r2_y = -1.1
    r2_z = terrain_height_at([r2_x, r2_y, 0]) + 0.5
    
    rock2 = UsdGeom.Sphere.Define(stage, "/World/Rocks/Rock2")
    rock2.CreateRadiusAttr(1.0)
    rock2.CreateDisplayColorAttr([(0.18, 0.18, 0.18)])
    xf2 = UsdGeom.Xformable(rock2.GetPrim())
    xf2.AddTranslateOp().Set(Gf.Vec3d(r2_x, r2_y, r2_z))
    xf2.AddScaleOp().Set(Gf.Vec3f(1.0, 0.5, 0.7))
    UsdPhysics.CollisionAPI.Apply(rock2.GetPrim())

def setup_lights(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0,0.98,0.93))
    
    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(4000.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    sun.CreateAngleAttr(0.53) 
    sun_xf = UsdGeom.Xformable(sun.GetPrim())
    sun_xf.AddRotateXOp().Set(-45.0) 
    sun_xf.AddRotateZOp().Set(30.0)  

    JX = 15.0 + J_HALF
    for i, (x,y,z) in enumerate([
        (4.0,0.0,3.0),(12.0,0.0,3.0),(JX,0.0,3.0),
        (JX,8.0,3.0),(JX,-8.0,3.0),(JX+8.0,0.0,3.2)
    ]):
        light = UsdLux.SphereLight.Define(stage, f"/World/KeyLight_{i}")
        light.CreateIntensityAttr(3000.0); light.CreateRadiusAttr(0.15)
        light.CreateColorAttr(Gf.Vec3f(1.0,0.82,0.55))
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(float(x),float(y),float(z)))

def configure_stage(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0,0.0,-1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

class RoverTraversal:
    def __init__(self, segments):
        self.seg_map        = {s.name: s for s in segments}
        self.actions        = []
        self.total_dist     = 0.0
        self.cumulative_dist = 0.0

        self._add_drive("root", False)

        for branch in ("left", "right", "mid"):
            self._add_turn_to(f"{branch}_branch", False)
            self._add_drive(f"{branch}_branch", False)
            self._add_drive(f"{branch}_tail",   False)
            self._add_turn_to(f"{branch}_tail", True)
            self._add_drive(f"{branch}_tail",   True)
            self._add_drive(f"{branch}_branch", True)

        self._add_turn_to("root", True)
        self._add_drive("root", True)

        self.idx         = 0
        self.local_s     = 0.0
        self.drive_speed = 3.0
        self.turn_rate   = 3.0
        self.turn_tol    = 0.08

    def current_action(self):
        return self.actions[self.idx] if self.idx < len(self.actions) else None

    def _add_drive(self, seg_name, reverse):
        seg = self.seg_map[seg_name]
        dist = seg.p_length - (4.0 if "tail" in seg.name else 0.0)
        self.total_dist += dist
        self.actions.append({"kind":"drive","segment":seg_name,"reverse":reverse,"dist":dist})

    def _add_turn_to(self, seg_name, reverse):
        seg = self.seg_map[seg_name]
        d   = (-seg.direction) if reverse else seg.direction
        self.actions.append({"kind":"turn","target_yaw":math.atan2(float(d[1]),float(d[0]))})

    def step(self, rover, dt):
        if self.idx >= len(self.actions): return True
        action   = self.actions[self.idx]
        pos, quat = rover.get_world_pose()
        yaw      = yaw_from_quat_wxyz(quat)

        if action["kind"] == "drive":
            seg       = self.seg_map[action["segment"]]
            reverse   = bool(action["reverse"])
            drive_dist = action["dist"]

            self.local_s        += self.drive_speed * dt
            self.cumulative_dist += self.drive_speed * dt
            done_seg = False
            if self.local_s >= drive_dist:
                self.cumulative_dist -= (self.local_s - drive_dist)
                self.local_s, done_seg = drive_dist, True

            s_eff = (drive_dist - self.local_s) if reverse else self.local_s
            tgt   = seg.path_point(s_eff)
            tgt[2] = terrain_height_at(tgt) + 0.35
            d     = (-seg.direction) if reverse else seg.direction
            tyaw  = math.atan2(float(d[1]), float(d[0]))
            tquat = np.array([math.cos(tyaw*0.5),0,0,math.sin(tyaw*0.5)], dtype=np.float32)
            rover.set_world_pose(position=tgt.astype(np.float32), orientation=tquat)
            if done_seg:
                self.idx, self.local_s = self.idx+1, 0.0
            return False

        elif action["kind"] == "turn":
            err = math.atan2(
                math.sin(float(action["target_yaw"])-yaw),
                math.cos(float(action["target_yaw"])-yaw)
            )
            if abs(err) <= self.turn_tol:
                self.idx, self.local_s = self.idx+1, 0.0
                return False
            new_yaw = yaw + np.sign(err)*self.turn_rate*dt
            rover.set_world_pose(
                position=pos.astype(np.float32),
                orientation=np.array([math.cos(new_yaw*0.5),0,0,math.sin(new_yaw*0.5)], dtype=np.float32)
            )
            return False

def main():
    print("\n" + "="*72)
    print("  CANARY ROVER — Isaac Sim 5.1.0  Professional Mine Inspection")
    print("  [Graphical Recording Mode - High Speed Traversal]")
    print("="*72 + "\n")

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
    build_rocks(stage)
    setup_lights(stage)

    add_reference_to_stage(usd_path=ROVER_USD, prim_path="/World/base_link")
    world.reset()

    rover = Articulation(prim_path="/World/base_link")
    rover.initialize()
    rover.disable_gravity()

    start_pos = np.array([0.5, 0.0, terrain_height_at([0.5,0.0,0.0])+0.35], dtype=np.float32)
    rover.set_world_pose(position=start_pos, orientation=np.array([1,0,0,0], dtype=np.float32))

    controller       = RoverTraversal(segments)

    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped():
            controller         = RoverTraversal(segments)
            continue

        if world.is_playing():
            rover.set_linear_velocity(np.zeros(3, dtype=np.float32))
            rover.set_angular_velocity(np.zeros(3, dtype=np.float32))

            done     = controller.step(rover, dt=1.0/60.0)
            pos, quat = rover.get_world_pose()
            yaw      = yaw_from_quat_wxyz(quat)

            if world.current_time_step_index % 8 == 0:
                fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
                set_camera_view(
                    eye    = pos - fwd*3.0 + np.array([0,0,1.5], dtype=np.float32),
                    target = pos + fwd*4.0
                )

            # RESTORED TELEMETRY BLOCK
            if world.current_time_step_index % 60 == 0:
                act          = controller.current_action()
                speed        = controller.drive_speed if not done and act and act["kind"]=="drive" else 0.0
                
                # IMU Data Mocking
                pitch        = math.degrees(math.atan2(
                    terrain_height_at([pos[0]+0.1,pos[1],0]) - terrain_height_at([pos[0]-0.1,pos[1],0]), 0.2
                ))
                roll         = 2.5*math.sin(world.current_time_step_index*0.1) if speed > 0 else 0.0
                ax = 9.81*math.sin(math.radians(pitch))   + np.random.normal(0,0.02)
                ay = -9.81*math.sin(math.radians(roll))   + np.random.normal(0,0.02)
                az = 9.81*math.cos(math.radians(pitch))*math.cos(math.radians(roll)) + np.random.normal(0,0.02)
                gx = np.random.normal(0, 0.005)
                gy = np.random.normal(pitch*0.05, 0.005) if speed > 0 else np.random.normal(0,0.005)
                gz = np.random.normal(0.02, 0.005)        if speed > 0 else np.random.normal(0,0.005)
                slope  = math.degrees(math.atan2(math.sqrt(ax**2+ay**2), az))
                
                # Encoder / Lidar Mocking
                rpm    = (speed/0.06)*(60/(2*math.pi))
                lidar_f = 12.0
                lidar_l = round((INNER_WIDTH/2)+np.random.normal(0,0.015), 2)
                lidar_r = round((INNER_WIDTH/2)+np.random.normal(0,0.015), 2)
                
                # Progress Mocking
                pct     = min(100.0,(controller.cumulative_dist/controller.total_dist)*100) if controller.total_dist>0 else 0.0
                bar     = "█"*int(pct//5) + "░"*(20-int(pct//5))

                print(f"┌─ Step {world.current_time_step_index:5d} ────────────────────────────────────────────────┐")
                print(f"│  [POS] x={pos[0]:6.2f}m  y={pos[1]:6.2f}m  z={pos[2]:+.3f}m  speed={speed:.2f}m/s")
                print(f"│  [IMU] pitch={pitch:+.2f}deg  roll={roll:+.2f}deg  slope={slope:.2f}deg")
                print(f"│  [ACC] ax={ax:+.2f}  ay={ay:+.2f}  az={az:+.2f} m/s^2")
                print(f"│  [GYR] wx={gx:+.3f}  wy={gy:+.3f}  wz={gz:+.3f} rad/s")
                print(f"│  [LDR] F={lidar_f:.1f}m  L={lidar_l:.2f}m  R={lidar_r:.2f}m")
                print(f"│  [ENC] FL/RL/FR/RR: {rpm:+.1f} RPM  (wheel r=6cm)")
                print(f"│  [PRG] [{bar}] {pct:5.1f}%  ({controller.cumulative_dist:.1f}/{controller.total_dist:.1f}m)")
                print(f"│  [SYS] Telemetry Active | High-Speed Graphical Traversal")
                print(f"└───────────────────────────────────────────────────────────────┘\n")

            if done:
                print("\n*** Traversal complete! ***\n")
                world.pause()

    simulation_app.close()

if __name__ == "__main__":
    main()
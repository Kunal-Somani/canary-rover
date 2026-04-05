from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import math
import numpy as np
import omni, omni.usd
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

ROVER_USD = "/home/kunal/isaac/canary_demo/robot/vleg_rover/vleg_rover.usd"
FINISH_X  = 55.0
TUNNEL_R  = 1.6
LENGTH    = 60.0

np.random.seed(42)
TERRAIN_SAMPLES = 300
_rx = np.linspace(0, LENGTH, TERRAIN_SAMPLES)
_terrain_h = (
    0.06  * np.sin(_rx * 0.25) +
    0.04  * np.cos(_rx * 0.8  + 0.5) +
    0.025 * np.sin(_rx * 2.1  + 1.2) +
    0.015 * np.sin(_rx * 4.5  + 0.3) +
    0.008 * np.random.normal(size=TERRAIN_SAMPLES)
)

def terrain_height_at(x):
    x = max(0.0, min(x, LENGTH - 0.01))
    idx = x / LENGTH * (TERRAIN_SAMPLES - 1)
    i0, i1 = int(idx), min(int(idx)+1, TERRAIN_SAMPLES-1)
    t = idx - i0
    return float(_terrain_h[i0]*(1-t) + _terrain_h[i1]*t)

def terrain_slope_at(x, dx=0.3):
    h1 = terrain_height_at(x + dx)
    h0 = terrain_height_at(max(0, x - dx))
    return math.degrees(math.atan2(h1 - h0, 2*dx))

def build_tunnel(stage):
    segs, rings = 160, 32
    verts, fcounts, idx = [], [], []
    for i in range(segs+1):
        x = i*(LENGTH/segs)
        for j in range(rings):
            a = (j/rings)*2*math.pi
            verts.append(Gf.Vec3f(x, TUNNEL_R*math.cos(a), TUNNEL_R*math.sin(a)))
    for i in range(segs):
        for j in range(rings):
            nj=(j+1)%rings
            v0=i*rings+j; v1=i*rings+nj; v2=(i+1)*rings+j; v3=(i+1)*rings+nj
            idx+=[v0,v2,v1,v1,v2,v3]; fcounts+=[3,3]
    m=UsdGeom.Mesh.Define(stage,"/World/tunnel")
    m.CreatePointsAttr(verts); m.CreateFaceVertexCountsAttr(fcounts)
    m.CreateFaceVertexIndicesAttr(idx); m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr([(0.5,0.35,0.2)])
    UsdGeom.Xformable(m.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-2.0,0.0,1.0))
    print("[World] Tunnel built")

def build_rocky_floor(stage):
    SX=TERRAIN_SAMPLES; SY=16; W=2.8
    verts,fcounts,indices=[],[],[]
    for i in range(SX):
        for j in range(SY):
            x=_rx[i]-2.0; y=(j/(SY-1))*W-W/2
            z=_terrain_h[i]+0.01*math.sin(j*0.8+i*0.3)
            verts.append(Gf.Vec3f(x,y,z))
    for i in range(SX-1):
        for j in range(SY-1):
            v0=i*SY+j; v1=i*SY+(j+1); v2=(i+1)*SY+j; v3=(i+1)*SY+(j+1)
            indices+=[v0,v1,v3,v2]; fcounts.append(4)
    mesh=UsdGeom.Mesh.Define(stage,"/World/rocky_floor")
    mesh.CreatePointsAttr(verts); mesh.CreateFaceVertexCountsAttr(fcounts)
    mesh.CreateFaceVertexIndicesAttr(indices); mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([(0.32,0.20,0.10)])
    print("[World] Rocky floor built")

def build_rocks(stage):
    positions=[(5,0.6),(12,0.5),(18,-0.6),(25,0.8),(33,0.3),(40,-0.8),(47,0.6),(50,-0.5)]
    for k,(rx,ry) in enumerate(positions):
        rz=terrain_height_at(rx)
        cube=UsdGeom.Cube.Define(stage,f"/World/rock_{k}")
        cube.CreateSizeAttr(1.0); cube.CreateDisplayColorAttr([(0.28,0.22,0.18)])
        xf=UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(rx-2.0,ry,rz+0.06))
        xf.AddScaleOp().Set(Gf.Vec3d(0.12,0.10,0.07))
    print("[World] Rocks placed")

def build_finish(stage):
    c=UsdGeom.Cube.Define(stage,"/World/finish_line")
    c.CreateSizeAttr(1.0); c.CreateDisplayColorAttr([(1.0,0.0,0.0)])
    xf=UsdGeom.Xformable(c.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(FINISH_X-2.0,0.0,0.5))
    xf.AddScaleOp().Set(Gf.Vec3d(0.2,3.5,1.0))
    print(f"[World] Finish line at x={FINISH_X}m")

def build_markers(stage):
    for k,x in enumerate(range(5,56,10)):
        tz=terrain_height_at(float(x))
        for side,lr in [(1.1,'L'),(-1.1,'R')]:
            cy=UsdGeom.Cylinder.Define(stage,f"/World/m{k}{lr}")
            cy.CreateRadiusAttr(0.04); cy.CreateHeightAttr(1.0)
            cy.CreateDisplayColorAttr([(1.0,0.55,0.0)])
            xf=UsdGeom.Xformable(cy.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(float(x)-2.0,side,tz+0.5))
    print("[World] Markers placed")

def setup_lights(stage):
    for k,x in enumerate(range(0,60,15)):
        lp=stage.DefinePrim(f"/World/L{k}","SphereLight")
        lp.CreateAttribute("inputs:intensity",Sdf.ValueTypeNames.Float).Set(6000.0)
        lp.CreateAttribute("inputs:radius",Sdf.ValueTypeNames.Float).Set(0.1)
        lp.CreateAttribute("inputs:color",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0,0.82,0.55))
        UsdGeom.Xformable(lp).AddTranslateOp().Set(Gf.Vec3d(float(x)-2.0,0.0,1.3))
    print("[World] Lighting done")

def main():
    print("\n"+"="*62)
    print("  CANARY ROVER — Isaac Sim 5.1.0  Mine Inspection Demo")
    print("  IMU (50Hz) | RPLiDAR A1M8 (10Hz) | 4x BLDC Encoders (50Hz)")
    print("="*62+"\n")

    world=World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage=omni.usd.get_context().get_stage()

    build_tunnel(stage); build_rocky_floor(stage); build_rocks(stage)
    build_finish(stage); build_markers(stage); setup_lights(stage)

    add_reference_to_stage(usd_path=ROVER_USD, prim_path="/World/rover")
    world.reset()
    rover=Articulation(prim_path="/World/rover")
    rover.initialize()
    print(f"[Rover] DOF: {rover.dof_names}\n")

    world.play()

    step=0; x_pos=0.0; speed=0.3; dt=1.0/60.0; lc=0; ll=None

    while simulation_app.is_running():
        world.step(render=True)
        step+=1; x_pos+=speed*dt

        th=terrain_height_at(x_pos)
        rover.set_world_pose(
            position=np.array([x_pos,0.0,th+0.22]),
            orientation=np.array([1.0,0.0,0.0,0.0])
        )

        if step%8==0:
            set_camera_view(
                eye=np.array([x_pos-2.5,0.0,th+0.7]),
                target=np.array([x_pos+5.0,0.0,th+0.1])
            )

        pitch=terrain_slope_at(x_pos)
        roll=2.5*math.sin(x_pos*0.4)
        ax=9.81*math.sin(math.radians(pitch))+np.random.normal(0,0.02)
        ay=-9.81*math.sin(math.radians(roll))+np.random.normal(0,0.02)
        az=9.81*math.cos(math.radians(pitch))*math.cos(math.radians(roll))+np.random.normal(0,0.02)
        gx=np.random.normal(0,0.005)
        gy=np.random.normal(pitch*0.05,0.005)
        gz=np.random.normal(0.02,0.005)
        slope=math.degrees(math.atan2(math.sqrt(ax**2+ay**2),az))
        rpm=speed/0.06*60/(2*math.pi)

        lc+=1
        if lc>=6:
            ll={"front":12.0,"left":round(TUNNEL_R+np.random.normal(0,0.015),2),
                "right":round(TUNNEL_R+np.random.normal(0,0.015),2),"min":round(TUNNEL_R-0.05,2)}
            lc=0

        if step%60==0:
            prog=min(100,int(x_pos/FINISH_X*100))
            bar="█"*(prog//5)+"░"*(20-prog//5)
            print(f"┌─ Step {step:5d} ────────────────────────────────────────────┐")
            print(f"│  📍 x={x_pos:6.2f}m  terrain_z={th:+.3f}m  speed={speed:.2f}m/s")
            print(f"│  🔵 IMU pitch={pitch:+.2f}°  roll={roll:+.2f}°  slope={slope:.2f}°")
            print(f"│  🔵 Accel ax={ax:+.2f}  ay={ay:+.2f}  az={az:+.2f} m/s²")
            print(f"│  🔵 Gyro  wx={gx:+.3f}  wy={gy:+.3f}  wz={gz:+.3f} rad/s")
            if ll: print(f"│  🟡 LiDAR F={ll['front']:.1f}m L={ll['left']:.2f}m R={ll['right']:.2f}m")
            print(f"│  🟢 Encoders FL/RL/FR/RR: {rpm:+.1f} RPM  (wheel r=6cm)")
            print(f"│  🏁 [{bar}] {prog}%  ({x_pos:.1f}/{FINISH_X}m)")
            print(f"└───────────────────────────────────────────────────────────────┘\n")

        if x_pos>=FINISH_X:
            print("\n"+"🎯 "*20)
            print("  FINISH LINE REACHED!  Mine Inspection Complete!")
            print("🎯 "*20+"\n")
            for _ in range(250): world.step(render=True)
            break

    simulation_app.close()

if __name__=="__main__":
    main()

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
TUNNEL_R  = 1.6
LENGTH    = 60.0

np.random.seed(42)
N = 300
_rx = np.linspace(0, LENGTH, N)
_h  = (0.06*np.sin(_rx*0.25) + 0.04*np.cos(_rx*0.8+0.5) +
       0.025*np.sin(_rx*2.1+1.2) + 0.015*np.sin(_rx*4.5+0.3))

def th(x):
    x = max(0.0, min(x, LENGTH-0.01))
    i = x/LENGTH*(N-1); i0=int(i); i1=min(i0+1,N-1); t=i-i0
    return float(_h[i0]*(1-t)+_h[i1]*t)

def build_tunnel(stage):
    segs,rings=160,32; verts,fcounts,idx=[],[],[]
    for i in range(segs+1):
        x=i*(LENGTH/segs)
        for j in range(rings):
            a=(j/rings)*2*math.pi
            verts.append(Gf.Vec3f(x,TUNNEL_R*math.cos(a),TUNNEL_R*math.sin(a)))
    for i in range(segs):
        for j in range(rings):
            nj=(j+1)%rings
            v0=i*rings+j;v1=i*rings+nj;v2=(i+1)*rings+j;v3=(i+1)*rings+nj
            idx+=[v0,v2,v1,v1,v2,v3];fcounts+=[3,3]
    m=UsdGeom.Mesh.Define(stage,"/World/tunnel")
    m.CreatePointsAttr(verts);m.CreateFaceVertexCountsAttr(fcounts)
    m.CreateFaceVertexIndicesAttr(idx);m.CreateDoubleSidedAttr(True)
    m.CreateDisplayColorAttr([(0.5,0.35,0.2)])
    UsdGeom.Xformable(m.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-2.0,0.0,1.0))

def build_floor(stage):
    SX=N;SY=16;W=2.8;verts,fcounts,indices=[],[],[]
    for i in range(SX):
        for j in range(SY):
            x=_rx[i]-2.0;y=(j/(SY-1))*W-W/2
            z=_h[i]+0.01*math.sin(j*0.8+i*0.3)
            verts.append(Gf.Vec3f(x,y,z))
    for i in range(SX-1):
        for j in range(SY-1):
            v0=i*SY+j;v1=i*SY+(j+1);v2=(i+1)*SY+j;v3=(i+1)*SY+(j+1)
            indices+=[v0,v1,v3,v2];fcounts.append(4)
    mesh=UsdGeom.Mesh.Define(stage,"/World/floor")
    mesh.CreatePointsAttr(verts);mesh.CreateFaceVertexCountsAttr(fcounts)
    mesh.CreateFaceVertexIndicesAttr(indices);mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([(0.32,0.20,0.10)])

def build_markers(stage):
    for k,x in enumerate(range(5,56,10)):
        tz=th(float(x))
        for side,lr in [(1.1,'L'),(-1.1,'R')]:
            cy=UsdGeom.Cylinder.Define(stage,f"/World/m{k}{lr}")
            cy.CreateRadiusAttr(0.04);cy.CreateHeightAttr(1.0)
            cy.CreateDisplayColorAttr([(1.0,0.55,0.0)])
            xf=UsdGeom.Xformable(cy.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(float(x)-2.0,side,tz+0.5))

def setup_lights(stage):
    for k,x in enumerate(range(0,60,15)):
        lp=stage.DefinePrim(f"/World/L{k}","SphereLight")
        lp.CreateAttribute("inputs:intensity",Sdf.ValueTypeNames.Float).Set(6000.0)
        lp.CreateAttribute("inputs:radius",Sdf.ValueTypeNames.Float).Set(0.1)
        lp.CreateAttribute("inputs:color",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0,0.82,0.55))
        UsdGeom.Xformable(lp).AddTranslateOp().Set(Gf.Vec3d(float(x)-2.0,0.0,1.3))

# ── CHANGE THIS to move rover position ──────────────────────────
ROVER_X = 10.0     # x position along tunnel (0 to 55)
# ────────────────────────────────────────────────────────────────

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

build_tunnel(stage); build_floor(stage)
build_markers(stage); setup_lights(stage)

add_reference_to_stage(usd_path=ROVER_USD, prim_path="/World/rover")
world.reset()
rover = Articulation(prim_path="/World/rover")
rover.initialize()

# Place rover at exact terrain height — never moves after this
rover.set_world_pose(
    position=np.array([ROVER_X, 0.0, th(ROVER_X) + 0.22]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])
)

print(f"\n Rover placed at x={ROVER_X}m  terrain_z={th(ROVER_X):+.3f}m")
print(" Simulation is STATIC — rover will not move")
print(" Use Alt+drag in viewport to orbit camera freely")
print(" Press Ctrl+F10 to save screenshot")
print(" Change ROVER_X in script for different positions\n")

# Run forever — static, rover never moves
while simulation_app.is_running():
    world.step(render=True)

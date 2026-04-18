import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

# ============================================================
# PHYSICS & DIMENSION CONFIG
# ============================================================
ROAD_THICKNESS  = 0.3
ROAD_Z_TOP      = 2.0
ROAD_Z_CENTER   = ROAD_Z_TOP - ROAD_THICKNESS / 2

SECTION_LEN     = 25.0
NUM_SECTIONS    = 5
FINISH_X        = SECTION_LEN * NUM_SECTIONS

ROVER_START     = [2.0, 0.0, 2.7]

WALL_W_S3 = 2.0
WALL_W_S4 = 1.6
WALL_W_S5 = 1.0

WALL_HEIGHT = 1.0
WALL_THICK  = 0.15

# REWARD SCALES
FORWARD_SCALE   = 15.0
LATERAL_SCALE   = 1.0
STABILITY_SCALE = 0.2
YAW_RATE_SCALE  = 0.2
ACTION_SMOOTH   = 0.05
SURVIVAL_BONUS  = 0.1
STALL_PENALTY   = 1.0

WALL_HIT_ONCE   = 20.0
WALL_HIT_ACCUM  = 5.0
WALL_HARD_FAIL  = 80.0

FALL_PENALTY    = 20.0
FINISH_REWARD   = 300.0

MAX_STEPS       = 800
TORQUE_SCALE    = 40.0
TORQUE_RAMP     = 0.25

# ============================================================
# CURRICULUM THRESHOLDS
#   Synchronized with train.py logic
# ============================================================
CURRICULUM_THRESHOLDS = {
    0: dict(mean=200.0, std=200.0, stable_episodes=50),
    1: dict(mean=350.0, std=250.0, stable_episodes=50),
    2: dict(mean=400.0, std=300.0, stable_episodes=0), # Final level
}

class RoverEnv(gym.Env):

    def __init__(self, render=False, random_level=0):
        super().__init__()

        self._render     = render
        self.random_level = random_level  # 0=fixed, 1=mild, 2=full

        if render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Observation: [x, y, z, roll, pitch, vx, vy, yaw_rate]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        # Action: [left_torque, right_torque]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.robot           = None
        self.joint_map       = {}
        self.wall_ids        = []

        self.prev_x             = 0.0
        self.prev_action        = np.zeros(2, dtype=np.float32)
        self.wall_contact_steps = 0
        self.current_step       = 0
        self.prev_checkpoint    = 0

    # ── Curriculum API ──────────────────────────────────────────

    def set_random_level(self, level: int):
        """Called by CurriculumCallback to increase difficulty."""
        assert level in (0, 1, 2), f"Level {level} out of bounds (0-2)"
        self.random_level = level

    # ── Reset & Build ───────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(fixedTimeStep=1/480, numSolverIterations=300)

        self._build_road()
        self._build_bumps()
        self._build_walls()
        self._build_finish_line()

        self.robot = p.loadURDF(
            "vleg_rover.urdf", ROVER_START,
            flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        self.joint_map = self._get_joint_map()

        # Randomize wheel friction slightly for robustness
        friction = np.random.uniform(0.8, 1.2)
        for j in self._all_wheels():
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)
            p.changeDynamics(self.robot, j, lateralFriction=friction)

        # Let physics settle
        for _ in range(120):
            p.stepSimulation()

        self.prev_x             = float(p.getBasePositionAndOrientation(self.robot)[0][0])
        self.prev_action        = np.zeros(2, dtype=np.float32)
        self.wall_contact_steps = 0
        self.current_step       = 0
        self.prev_checkpoint    = 0

        return self._get_obs(), {}

    def _build_road(self):
        slabs = [
            (0,   25,  1.0),
            (25,  50,  1.0),
            (50,  75,  WALL_W_S3),
            (75,  100, WALL_W_S4),
            (100, 125, WALL_W_S5),
        ]
        for xs, xe, hw in slabs:
            half_len = (xe - xs) / 2.0
            cx = xs + half_len
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[half_len, hw, ROAD_THICKNESS / 2])
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[half_len, hw, ROAD_THICKNESS / 2],
                rgbaColor=[0.55, 0.42, 0.20, 1.0])
            p.createMultiBody(0, col, vis, basePosition=[cx, 0.0, ROAD_Z_CENTER])

    def _bump_params(self):
        """Return (rng, fx_range, fy_range, amp_range) for current level."""
        if self.random_level == 0:
            return np.random.default_rng(42), (2.5, 2.5), (2.5, 2.5), (0.5, 0.5)
        elif self.random_level == 1:
            return np.random.default_rng(), (2.0, 3.0), (2.0, 3.0), (0.4, 0.6)
        else:
            return np.random.default_rng(), (1.5, 4.0), (1.5, 4.0), (0.3, 0.7)

    def _build_bumps(self):
        MAX_BUMP_H = 0.035
        RES        = 64
        zones = [
            (25,  50,  1.0,       1.0),
            (75,  100, WALL_W_S4, 1.4),
            (100, 125, WALL_W_S5, 1.8),
        ]

        rng, fxr, fyr, ampr = self._bump_params()

        for xs, xe, hw, amp_scale in zones:
            lx, ly = float(xe - xs), float(hw * 2)
            cx = xs + lx / 2.0
            XX, YY = np.meshgrid(np.linspace(0, 1, RES), np.linspace(0, 1, RES), indexing='ij')

            h = np.zeros((RES, RES))
            for _ in range(4):
                fx, fy = rng.uniform(*fxr), rng.uniform(*fyr)
                amp = rng.uniform(*ampr) * amp_scale
                h += amp * np.sin(fx*2*np.pi*XX + rng.uniform(0, 2*np.pi)) * \
                           np.sin(fy*2*np.pi*YY + rng.uniform(0, 2*np.pi))

            h -= h.min()
            if h.max() > 1e-6:
                h = h / h.max() * MAX_BUMP_H

            col = p.createCollisionShape(
                p.GEOM_HEIGHTFIELD, meshScale=[lx/(RES-1), ly/(RES-1), 1.0],
                heightfieldData=h.flatten().tolist(), numHeightfieldRows=RES, numHeightfieldColumns=RES)
            body = p.createMultiBody(0, col, basePosition=[cx, 0.0, ROAD_Z_TOP])
            p.changeVisualShape(body, -1, rgbaColor=[0.4, 0.28, 0.14, 1.0])

    def _build_walls(self):
        specs = [(50, 75, WALL_W_S3), (75, 100, WALL_W_S4), (100, 125, WALL_W_S5)]
        self.wall_ids = []
        for xs, xe, ihw in specs:
            half_len, cx = (xe - xs) / 2.0, xs + (xe - xs) / 2.0
            cz = ROAD_Z_TOP + WALL_HEIGHT
            for sign in (-1, +1):
                cy = sign * (ihw + WALL_THICK)
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_len, WALL_THICK, WALL_HEIGHT])
                wid = p.createMultiBody(0, col, basePosition=[cx, cy, cz])
                p.changeVisualShape(wid, -1, rgbaColor=[0.7, 0.1, 0.1, 1.0]) 
                self.wall_ids.append(wid)

    def _build_finish_line(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, WALL_W_S5, 0.4])
        p.createMultiBody(0, col, basePosition=[FINISH_X, 0.0, ROAD_Z_TOP + 0.4])



    def _get_joint_map(self):
        return {p.getJointInfo(self.robot, i)[1].decode(): i for i in range(p.getNumJoints(self.robot))}

    def _all_wheels(self):
        return [self.joint_map[f"wheel_{side}_{pos}_joint"] 
                for side in ["left", "right"] for pos in ["front", "rear"]]

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        return np.array([pos[0], pos[1], pos[2], roll, pitch, lin_vel[0], lin_vel[1], ang_vel[2]], dtype=np.float32)



    def step(self, action):

        action = np.clip(action, -1.0, 1.0)
        delta  = np.clip(action - self.prev_action, -TORQUE_RAMP, TORQUE_RAMP)
        action = np.clip(self.prev_action + delta, -1.0, 1.0)

        left, right = float(action[0]), float(action[1])
        wheels = self._all_wheels()
        for i, j in enumerate(wheels):
            torque = left if i < 2 else right
            p.setJointMotorControl2(self.robot, j, p.TORQUE_CONTROL, force=torque * TORQUE_SCALE)

        for _ in range(8):
            p.stepSimulation()

        obs = self._get_obs()
        x, y, z, roll, pitch, vx, vy, yaw_rate = obs

        forward = x - self.prev_x
        self.prev_x = x
        action_delta = float(np.sum(np.abs(action - self.prev_action)))
        self.prev_action = action.copy()


        contacts = p.getContactPoints(bodyA=self.robot)
        wall_contacts = [c for c in contacts if c[2] in self.wall_ids]
        n_wall = len(wall_contacts)

        if n_wall > 0:
            self.wall_contact_steps += 1
        else:
            self.wall_contact_steps = 0

        wall_pen = -min(50.0, (WALL_HIT_ONCE * n_wall + WALL_HIT_ACCUM * self.wall_contact_steps))

        done = truncated = False
        if n_wall > 3: 
            wall_pen -= WALL_HARD_FAIL
            done = True


        reward = (
            FORWARD_SCALE   * forward -
            LATERAL_SCALE   * abs(y) -
            STABILITY_SCALE * (abs(roll) + abs(pitch)) -
            YAW_RATE_SCALE  * abs(yaw_rate) -
            ACTION_SMOOTH   * action_delta +
            SURVIVAL_BONUS +
            wall_pen
        )

        if abs(forward) < 0.002: reward -= STALL_PENALTY
        if vx < 0.05: reward -= 0.3 * max(0.0, 0.05 - vx)


        ckpt = int(max(x, 0) // 10)
        if ckpt > self.prev_checkpoint:
            reward += 10.0
            self.prev_checkpoint = ckpt


        reward = float(np.clip(reward, -50.0, 100.0))

        if x >= FINISH_X:
            reward += FINISH_REWARD
            done = True
        elif z < 1.2 or abs(roll) > 1.3 or abs(pitch) > 1.3:
            reward -= FALL_PENALTY
            done = True

        self.current_step += 1
        if self.current_step >= MAX_STEPS:
            truncated = True

        return obs, reward, done, truncated, {}

    def close(self):
        p.disconnect()

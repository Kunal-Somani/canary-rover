import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class RoverEnv(gym.Env):

    def __init__(self, render=True):
        super().__init__()

        self.render = render

        if render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.robot = None
        self.bootstrap_steps = 0

        self.length = 60
        self.finish_x = 55

        self.prev_x = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240,
            numSolverIterations=300
        )

        self._create_floor()
        self._create_tunnel()
        self._create_finish_line()  # 🔴 ADD THIS

        self.robot = p.loadURDF("vleg_rover.urdf", [-2, 0, 0.7])

        self.joint_map = self._get_joint_map()

        for j in self._all_wheels():
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, force=0)
            p.changeDynamics(self.robot, j, lateralFriction=1.0)

        for _ in range(120):
            p.stepSimulation()

        self.bootstrap_steps = 80
        self.prev_x = 0

        return self._get_obs(), {}


    # ================= FLOOR =================
    def _create_floor(self):

        size = 512
        length = self.length

        heightfield = np.zeros(size * size)

        for i in range(size):
            for j in range(size):

                x_ratio = i / size

                if x_ratio < 0.3:
                    slope = 0
                elif x_ratio < 0.6:
                    slope = 0.002 * (i - size * 0.3)
                else:
                    slope = 0.006 * (i - size * 0.6)

                h1 = 0.015 * np.sin(i * 0.15)
                h2 = 0.015 * np.cos(j * 0.15)

                heightfield[i * size + j] = slope + h1 + h2

        terrain = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[length/size, 5.0/size, 1.0],
            heightfieldData=heightfield,
            numHeightfieldRows=size,
            numHeightfieldColumns=size
        )

        body = p.createMultiBody(0, terrain)
        p.changeVisualShape(body, -1, rgbaColor=[0.45, 0.30, 0.15, 1])


    # ================= TUNNEL =================
    def _create_tunnel(self):

        length = self.length
        segments = 160
        rings = 32
        radius = 1.6

        vertices = []
        indices = []

        for i in range(segments + 1):
            x = i * (length / segments)

            for j in range(rings):
                angle = (j / rings) * 2 * np.pi
                y = radius * np.cos(angle)
                z = radius * np.sin(angle)

                vertices.append([x, y, z])

        for i in range(segments):
            for j in range(rings):
                nj = (j + 1) % rings

                v0 = i * rings + j
                v1 = i * rings + nj
                v2 = (i + 1) * rings + j
                v3 = (i + 1) * rings + nj

                indices += [v0, v2, v1]
                indices += [v1, v2, v3]

        mesh = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices
        )

        visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=[0.5, 0.35, 0.2, 1]
        )

        p.createMultiBody(0, mesh, visual, basePosition=[0, 0, 1.0])


    # ================= 🔴 FINISH LINE =================
    def _create_finish_line(self):

        thickness = 0.1
        width = 4.0
        height = 0.05

        visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[thickness, width/2, height],
            rgbaColor=[1, 0, 0, 1]  # 🔴 RED
        )

        collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[thickness, width/2, height]
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[self.finish_x, 0, 0.05]
        )


    def _get_joint_map(self):
        return {
            p.getJointInfo(self.robot, i)[1].decode(): i
            for i in range(p.getNumJoints(self.robot))
        }


    def _all_wheels(self):
        return [
            self.joint_map["wheel_left_front_joint"],
            self.joint_map["wheel_left_rear_joint"],
            self.joint_map["wheel_right_front_joint"],
            self.joint_map["wheel_right_rear_joint"],
        ]


    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        return np.array([
            pos[0], pos[1], pos[2],
            roll, pitch,
            lin_vel[0], lin_vel[1],
            ang_vel[2]
        ], dtype=np.float32)


    def step(self, action):

        action = np.clip(action, -1, 1)
        left, right = action

        if self.bootstrap_steps > 0:
            left = right = 0.3
            self.bootstrap_steps -= 1

        torque_scale = 120

        for i, j in enumerate(self._all_wheels()):
            torque = left if i < 2 else right
            p.setJointMotorControl2(
                self.robot,
                j,
                p.TORQUE_CONTROL,
                force=torque * torque_scale
            )

        for _ in range(4):
            p.stepSimulation()

        # follow camera
        if self.render:
            pos, _ = p.getBasePositionAndOrientation(self.robot)
            p.resetDebugVisualizerCamera(
                cameraDistance=5,
                cameraYaw=0,
                cameraPitch=-20,
                cameraTargetPosition=pos
            )

        obs = self._get_obs()

        x = obs[0]
        roll, pitch = obs[3], obs[4]

        # 🔥 DIRECTIONAL REWARD
        forward_reward = (x - self.prev_x)
        self.prev_x = x
        
        velocity_rew = 0.05*obs[5]

        reward = forward_reward * 3 + velocity_rew
        reward -= 0.25 * (abs(roll) + abs(pitch))
        reward -= 0.01

        done = False
        truncated = False

        # 🎯 GOAL REWARD (STRONG BUT SAFE)
        if x > self.finish_x:
            reward += 100
            done = True

        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            done = True

        return obs, reward, done, truncated, {}


    def close(self):
        p.disconnect()

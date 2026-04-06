from stable_baselines3 import PPO
from rover_env import RoverEnv

env = RoverEnv(render=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000)

model.save("rover_model")

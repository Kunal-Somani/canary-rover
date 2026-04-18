from stable_baselines3 import PPO
from rover_env import RoverEnv

env = RoverEnv(render=True)

model = PPO.load("rover_refined_final_v2.zip") 

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)

    if done or truncated:
        obs, _ = env.reset()

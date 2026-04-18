
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from rover_env import RoverEnv
import numpy as np

# ============================================================
# 📊 CUSTOM STD LOGGER
# ============================================================

class RewardStdCallback(BaseCallback):
    def __init__(self, window_size=50, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.rewards = []

    def _on_step(self) -> bool:
        # infos contain episode info when episode ends
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.rewards.append(ep_reward)

                # keep rolling window
                if len(self.rewards) > self.window_size:
                    self.rewards.pop(0)

        # log only when we have enough data
        if len(self.rewards) > 1:
            reward_std = np.std(self.rewards)
            reward_mean = np.mean(self.rewards)

            self.logger.record("custom/reward_std", reward_std)
            self.logger.record("custom/reward_mean", reward_mean)

        return True


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "rover_stabilized_final.zip"
TOTAL_CYCLES = 8
STEPS_PER_CYCLE = 20000

# ============================================================
# ENVIRONMENTS (LEVEL 2 ONLY)
# ============================================================

train_env = RoverEnv(render=False, random_level=2)
eval_env  = RoverEnv(render=False, random_level=2)

# ============================================================
# LOAD MODEL
# ============================================================

model = PPO.load(CHECKPOINT_PATH, env=train_env)

print("\nLoaded model:", CHECKPOINT_PATH)

# ============================================================
# 🚀 REFINEMENT MODE
# ============================================================

model.ent_coef = 0.003
model.learning_rate = lambda _: 5e-5
model.clip_range    = lambda _: 0.15

# ============================================================
# CALLBACKS
# ============================================================

checkpoint_cb = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="rover_refined_v2"
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

std_cb = RewardStdCallback(window_size=50)

# ============================================================
# 🔁 TRAIN IN CONTROLLED BURSTS
# ============================================================

for i in range(TOTAL_CYCLES):
    print(f"\n==============================")
    print(f"   Refinement Cycle {i+1}")
    print(f"==============================\n")

    model.learn(
        total_timesteps=STEPS_PER_CYCLE,
        reset_num_timesteps=False,
        callback=[checkpoint_cb, eval_cb, std_cb]
    )

    save_path = f"refined_cycle_{i+1}"
    model.save(save_path)

    print(f"\nSaved: {save_path}\n")

print("\n✅ Refinement complete.")

model.save("rover_refined_final_v2")
print("Final model saved as rover_refined_final_v2")


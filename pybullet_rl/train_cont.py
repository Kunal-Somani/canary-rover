from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from rover_env import RoverEnv
from train import CurriculumCallback   # reuse your existing callback

# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "checkpoints/rover_refined_110000_steps.zip"  # <-- CHANGE if needed
TOTAL_CYCLES = 10
STEPS_PER_CYCLE = 20000

# ============================================================
# ENVIRONMENTS (FREEZE CURRICULUM)
# ============================================================

train_env = RoverEnv(render=False, random_level=0)
eval_env  = RoverEnv(render=False, random_level=0)

# ============================================================
# LOAD MODEL
# ============================================================

model = PPO.load(CHECKPOINT_PATH, env=train_env)

print("\nLoaded checkpoint:", CHECKPOINT_PATH)

# ============================================================
# 🔒 STABILIZATION MODE
# ============================================================

# Smaller updates → prevents collapse
model.learning_rate = lambda _: 1e-4
model.clip_range    = lambda _: 0.15

# Reduce exploration slightly
model.ent_coef = 0.001

# (vf_coef already baked into model from previous training)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint_cb = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="rover_stable"
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)

# OPTIONAL: keep curriculum callback BUT frozen at level 0
curriculum_cb = CurriculumCallback(
    env=train_env,
    eval_env=eval_env,
    verbose=1
)

# Force level 0 (safety)
train_env.set_random_level(0)
eval_env.set_random_level(0)

# ============================================================
# 🔁 TRAIN IN CONTROLLED BURSTS
# ============================================================

for i in range(TOTAL_CYCLES):
    print(f"\n==============================")
    print(f"   Stabilization Cycle {i+1}")
    print(f"==============================\n")

    model.learn(
        total_timesteps=STEPS_PER_CYCLE,
        reset_num_timesteps=False,
        callback=[checkpoint_cb, eval_cb, curriculum_cb]
    )

    # Save after each cycle (VERY IMPORTANT)
    save_path = f"stabilized_cycle_{i+1}"
    model.save(save_path)

    print(f"\nSaved: {save_path}\n")

print("\n✅ Stabilization complete.")

model.save("rover_stabilized_final")
print("Final model saved as rover_stabilized_final")


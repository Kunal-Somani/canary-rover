from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
import numpy as np
from rover_env import RoverEnv, CURRICULUM_THRESHOLDS


# ============================================================
# ADAPTIVE CURRICULUM CALLBACK
# ============================================================

class CurriculumCallback(BaseCallback):

    def __init__(self, env: RoverEnv, eval_env: RoverEnv, verbose=1):
        super().__init__(verbose)

        self.env      = env
        self.eval_env = eval_env

        self._ep_rewards = []
        self._current_ep_reward = 0.0

        self._level = 0
        self._stable_counter = 0

        self._entropy_by_level = {
            0: 0.01,
            1: 0.005,
            2: 0.002
        }

    def _set_level(self, level: int):
        self._level = level
        self.env.set_random_level(level)
        self.eval_env.set_random_level(level)

        new_ent_coef = self._entropy_by_level.get(level, 0.002)
        if hasattr(self.model, "ent_coef"):
            self.model.ent_coef = 0.8 * self.model.ent_coef + 0.2 * new_ent_coef

        if self.verbose:
            print(f"\n[Curriculum] ▶ Level → {level} "
                  f"(step {self.num_timesteps:,}) | Entropy: {new_ent_coef}\n")

    def _check_advance(self) -> bool:
        if self._level >= 2:
            return False

        thresh = CURRICULUM_THRESHOLDS[self._level]

        if len(self._ep_rewards) < 20:
            return False

        window = self._ep_rewards[-100:]
        mean = float(np.mean(window))
        std = float(np.std(window))

        # 🔥 RELAXED STD CONDITION
        std_threshold = thresh["std"] * 1.5

        conditions_met = (
            mean > thresh["mean"] and
            std < std_threshold
        )

        if conditions_met:
            self._stable_counter += 1

            target_episodes = thresh.get(
                "stable_episodes",
                max(1, thresh.get("stable_steps", 50000) // 1000)
            )

            if self._stable_counter >= target_episodes:
                return True
        else:
            self._stable_counter = 0

        return False

    def _check_collapse(self) -> bool:
        if self._level == 0 or len(self._ep_rewards) < 20:
            return False

        thresh = CURRICULUM_THRESHOLDS[self._level - 1]
        window = self._ep_rewards[-50:]
        mean = float(np.mean(window))

        collapse_threshold = thresh["mean"] * 0.65
        return mean < collapse_threshold

    def _on_step(self) -> bool:
        self._current_ep_reward += float(self.locals["rewards"][0])

        done = bool(self.locals["dones"][0])
        truncated = self.locals.get("infos", [{}])[0].get(
            "TimeLimit.truncated", False)

        if done or truncated:
            self._ep_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0.0

            if len(self._ep_rewards) > 500:
                self._ep_rewards = self._ep_rewards[-500:]

            if self._check_advance():
                self._set_level(self._level + 1)
                self._stable_counter = 0

            elif self._check_collapse():
                new_level = max(0, self._level - 1)
                if new_level != self._level:
                    print(f"\n[Curriculum] ⚠ Collapse → level {new_level}\n")
                    self._set_level(new_level)
                    self._stable_counter = 0

        if self.num_timesteps > 0 and self.num_timesteps % 5000 == 0:
            if len(self._ep_rewards) >= 10:
                w = self._ep_rewards[-50:]
                mean = float(np.mean(w))
                std = float(np.std(w))

                thresh = CURRICULUM_THRESHOLDS.get(self._level, {})
                target_eps = thresh.get(
                    "stable_episodes",
                    max(1, thresh.get("stable_steps", 50000) // 1000)
                )

                print(f"[Curriculum] step={self.num_timesteps:>7,} | "
                      f"level={self._level} | "
                      f"mean={mean:>8.1f} | "
                      f"std={std:>7.1f} | "
                      f"stable_eps={self._stable_counter}/{target_eps}")

        return True


# ============================================================
# TRAINING ENTRY POINT
# ============================================================

def main():
    train_env = RoverEnv(render=False, random_level=0)
    eval_env  = RoverEnv(render=False, random_level=0)

    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="rover"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    curriculum_cb = CurriculumCallback(
        env=train_env,
        eval_env=eval_env,
        verbose=1
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=2e-4,   # ✅ stabilized
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.3,          # ✅ CRITICAL FIX (prevents value explosion)
        device="cpu",
        tensorboard_log="./logs/"
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_cb, eval_cb, curriculum_cb]
    )

    model.save("rover_model_final")
    print("\nTraining complete. Model saved.")


if __name__ == "__main__":
    main()

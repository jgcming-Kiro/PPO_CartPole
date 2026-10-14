import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


SEED = 42
ENV_ID = "CartPole-v1"
TOTAL_TIMESTEPS = 30_000
EVAL_EPISODES = 30
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.rows.append(
                    {
                        "timesteps": int(self.num_timesteps),
                        "episode_reward": float(episode["r"]),
                        "episode_length": int(episode["l"]),
                    }
                )
        return True


def make_env(seed: int):
    def _init():
        env = gym.make(ENV_ID)
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def evaluate_random_policy():
    returns = []
    lengths = []
    for episode in range(EVAL_EPISODES):
        env = gym.make(ENV_ID)
        _, _ = env.reset(seed=SEED + episode)
        env.action_space.seed(SEED + episode)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        env.close()
        returns.append(total_reward)
        lengths.append(steps)
    return np.array(returns), np.array(lengths)


def evaluate_trained_model(model):
    returns = []
    lengths = []
    for episode in range(EVAL_EPISODES):
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=SEED + 1000 + episode)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        env.close()
        returns.append(total_reward)
        lengths.append(steps)
    return np.array(returns), np.array(lengths)


def main():
    np.random.seed(SEED)

    baseline_returns, baseline_lengths = evaluate_random_policy()
    pd.DataFrame(
        {
            "episode": np.arange(1, EVAL_EPISODES + 1),
            "return": baseline_returns,
            "length": baseline_lengths,
        }
    ).to_csv(RESULTS_DIR / "baseline_random.csv", index=False)

    train_env = DummyVecEnv([make_env(SEED)])
    model = PPO(
        "MlpPolicy",
        train_env,
        seed=SEED,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        device="cpu",
        verbose=0,
    )

    callback = RewardLogger()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    model.save(RESULTS_DIR / "ppo_cartpole_model")

    learning_df = pd.DataFrame(callback.rows)
    learning_df["rolling_mean_20"] = learning_df["episode_reward"].rolling(20, min_periods=1).mean()
    learning_df.to_csv(RESULTS_DIR / "ppo_learning_curve.csv", index=False)

    ppo_returns, ppo_lengths = evaluate_trained_model(model)
    pd.DataFrame(
        {
            "episode": np.arange(1, EVAL_EPISODES + 1),
            "return": ppo_returns,
            "length": ppo_lengths,
        }
    ).to_csv(RESULTS_DIR / "ppo_evaluation.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "agent": "Baseline aleatorio",
                "mean_return": baseline_returns.mean(),
                "std_return": baseline_returns.std(ddof=1),
                "min_return": baseline_returns.min(),
                "max_return": baseline_returns.max(),
                "mean_length": baseline_lengths.mean(),
            },
            {
                "agent": "PPO entrenado",
                "mean_return": ppo_returns.mean(),
                "std_return": ppo_returns.std(ddof=1),
                "min_return": ppo_returns.min(),
                "max_return": ppo_returns.max(),
                "mean_length": ppo_lengths.mean(),
            },
        ]
    )
    summary_df.to_csv(RESULTS_DIR / "baseline_vs_ppo_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(
        summary_df["agent"],
        summary_df["mean_return"],
        yerr=summary_df["std_return"],
        capsize=6,
        color=["#777777", "#1f77b4"],
    )
    plt.ylabel("Retorno promedio en 30 episodios")
    plt.title("Comparacion de desempeno")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_vs_ppo.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(learning_df["timesteps"], learning_df["episode_reward"], alpha=0.25, label="Retorno por episodio")
    plt.plot(learning_df["timesteps"], learning_df["rolling_mean_20"], linewidth=2.3, label="Promedio movil 20 episodios")
    plt.axhline(475, color="tab:green", linestyle="--", linewidth=1.5, label="Referencia: 475")
    plt.xlabel("Timesteps")
    plt.ylabel("Retorno episodico")
    plt.title("Curva de aprendizaje de PPO en CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ppo_learning_curve.png", dpi=180)
    plt.close()

    print(summary_df.to_string(index=False))
    print(f"Resultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

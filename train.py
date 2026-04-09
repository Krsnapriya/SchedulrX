#!/usr/bin/env python3
"""
SchedulrX PPO Training Script
================================
Train a PyTorch PPO agent on the SchedulrX Gymnasium environment.

Usage:
    # Train
    python train.py --steps 50000 --task easy --lr 3e-4

    # Evaluate a saved model
    python train.py --eval-only --model checkpoint.pt --task hard

    # Quick smoke test
    python train.py --steps 1000 --task easy

Outputs:
    - checkpoint.pt         — saved model weights
    - training_curve.png    — reward/score learning curves
    - eval results (JSON)   — when using --eval-only
"""

import argparse
import json
import random
import sys
import os

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_env import SchedulrXGymEnv, OBS_DIM, ACTION_DIM


def set_all_seeds(seed: int):
    """Determinism lock — reproducible benchmarking."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def train(args):
    """PPO training loop with GAE and action masking."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from schedulrx.agent import SchedulrXActorCritic, RolloutBuffer

    set_all_seeds(args.seed)

    env = SchedulrXGymEnv(task_name=args.task)
    model = SchedulrXActorCritic(OBS_DIM, ACTION_DIM, hidden=args.hidden)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Hyperparameters
    rollout_len = args.rollout_len
    n_epochs = args.ppo_epochs
    batch_size = args.batch_size
    clip_eps = args.clip_eps
    ent_coef = args.ent_coef
    vf_coef = args.vf_coef
    max_grad_norm = 0.5
    gamma = args.gamma
    gae_lam = args.gae_lam

    buffer = RolloutBuffer(rollout_len, OBS_DIM, ACTION_DIM)

    # Tracking
    episode_rewards = []
    episode_scores = []
    episode_steps = []
    current_ep_reward = 0.0
    global_step = 0
    best_score = -float("inf")

    obs, info = env.reset(seed=args.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32)
    mask = env.get_action_mask()
    mask_t = torch.tensor(mask, dtype=torch.bool)

    print(f"[TRAIN] task={args.task} steps={args.steps} lr={args.lr} seed={args.seed}")
    print(f"[TRAIN] obs_dim={OBS_DIM} action_dim={ACTION_DIM} hidden={args.hidden}")

    while global_step < args.steps:
        # ── Collect rollout ──────────────────────────────────────────────
        buffer.reset()
        model.eval()

        for _ in range(rollout_len):
            with torch.no_grad():
                action, log_prob, entropy, value = model.get_action_and_value(
                    obs_t.unsqueeze(0), mask_t.unsqueeze(0)
                )

            action_int = action.item()
            next_obs, reward, done, truncated, step_info = env.step(action_int)
            terminated = done or truncated

            buffer.store(obs, action_int, log_prob.item(), reward, terminated, value.item(), mask)

            current_ep_reward += reward
            global_step += 1

            if terminated:
                score = env.get_grader_score().get("score", 0.0)
                episode_rewards.append(current_ep_reward)
                episode_scores.append(score)
                episode_steps.append(env.core_env.step_count)
                current_ep_reward = 0.0

                obs, info = env.reset(seed=args.seed + global_step)
                mask = env.get_action_mask()
            else:
                obs = next_obs
                mask = env.get_action_mask()

            obs_t = torch.tensor(obs, dtype=torch.float32)
            mask_t = torch.tensor(mask, dtype=torch.bool)

            if global_step >= args.steps:
                break

        # Compute GAE
        with torch.no_grad():
            _, _, _, last_val = model.get_action_and_value(
                obs_t.unsqueeze(0), mask_t.unsqueeze(0)
            )
        buffer.compute_gae(last_val.item(), gamma=gamma, lam=gae_lam)

        # ── PPO Update ───────────────────────────────────────────────────
        model.train()
        n_samples = min(buffer.ptr, buffer.buffer_size)

        for epoch in range(n_epochs):
            for batch_idx in buffer.get_batches(batch_size):
                b_obs = torch.tensor(buffer.obs[batch_idx], dtype=torch.float32)
                b_actions = torch.tensor(buffer.actions[batch_idx], dtype=torch.long)
                b_log_probs = torch.tensor(buffer.log_probs[batch_idx], dtype=torch.float32)
                b_advantages = torch.tensor(buffer.advantages[batch_idx], dtype=torch.float32)
                b_returns = torch.tensor(buffer.returns[batch_idx], dtype=torch.float32)
                b_masks = torch.tensor(buffer.masks[batch_idx], dtype=torch.bool)

                # Normalize advantages
                if len(b_advantages) > 1:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                _, new_log_prob, entropy, new_value = model.get_action_and_value(
                    b_obs, b_masks, action=b_actions
                )

                # Policy loss (clipped surrogate)
                ratio = (new_log_prob - b_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((new_value - b_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────
        if episode_rewards and len(episode_rewards) % 5 == 0:
            recent_r = episode_rewards[-10:]
            recent_s = episode_scores[-10:]
            avg_r = sum(recent_r) / len(recent_r)
            avg_s = sum(recent_s) / len(recent_s)
            print(
                f"[TRAIN] step={global_step:>6d}/{args.steps} "
                f"episodes={len(episode_rewards)} "
                f"avg_reward={avg_r:.3f} avg_score={avg_s:.3f}"
            )

            if avg_s > best_score:
                best_score = avg_s
                torch.save(model.state_dict(), args.save_model)
                print(f"[TRAIN] ★ New best score: {avg_s:.3f} → saved to {args.save_model}")

    # Final save
    import torch as _torch
    _torch.save(model.state_dict(), args.save_model)
    print(f"\n[TRAIN] Training complete. Model saved to {args.save_model}")

    # ── Generate learning curve ──────────────────────────────────────────
    if episode_rewards:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"SchedulrX PPO Training — {args.task} task", fontsize=14, fontweight="bold")

            # Smooth with rolling average
            window = min(20, len(episode_rewards) // 3 + 1)

            def smooth(data, w):
                if len(data) < w:
                    return data
                return [sum(data[max(0, i-w):i+1]) / min(i+1, w) for i in range(len(data))]

            ax1.plot(episode_rewards, alpha=0.3, color="#64748b", linewidth=0.5)
            ax1.plot(smooth(episode_rewards, window), color="#3b82f6", linewidth=2)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Cumulative Reward")
            ax1.set_title("Reward Trajectory")
            ax1.grid(True, alpha=0.3)

            ax2.plot(episode_scores, alpha=0.3, color="#64748b", linewidth=0.5)
            ax2.plot(smooth(episode_scores, window), color="#10b981", linewidth=2)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Grader Score")
            ax2.set_title("Grader Score Trajectory")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig("training_curve.png", dpi=150, bbox_inches="tight")
            print(f"[TRAIN] Learning curve saved to training_curve.png")
        except ImportError:
            print("[TRAIN] matplotlib not available — skipping curve generation")

    # Final evaluation stats
    if episode_scores:
        print(f"\n[RESULTS] task={args.task}")
        print(f"  episodes:    {len(episode_scores)}")
        print(f"  avg_reward:  {sum(episode_rewards)/len(episode_rewards):.3f}")
        print(f"  avg_score:   {sum(episode_scores)/len(episode_scores):.3f}")
        print(f"  best_score:  {max(episode_scores):.3f}")
        print(f"  avg_steps:   {sum(episode_steps)/len(episode_steps):.1f}")


def evaluate(args):
    """Evaluate a trained model (--eval-only mode)."""
    import torch
    from schedulrx.agent import SchedulrXActorCritic

    set_all_seeds(args.seed)

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    model = SchedulrXActorCritic(OBS_DIM, ACTION_DIM, hidden=args.hidden)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    tasks = [args.task] if args.task != "all" else ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        env = SchedulrXGymEnv(task_name=task)
        scores = []
        rewards = []
        steps_list = []
        soft_fails = []

        for ep in range(args.eval_episodes):
            obs, info = env.reset(seed=args.seed + ep)
            mask = env.get_action_mask()
            ep_reward = 0.0
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(
                        obs_t, mask_t, deterministic=True
                    )

                obs, reward, done, truncated, step_info = env.step(action.item())
                done = done or truncated
                ep_reward += reward
                mask = env.get_action_mask()

            grade = env.get_grader_score()
            scores.append(grade.get("score", 0.0))
            rewards.append(ep_reward)
            steps_list.append(env.core_env.step_count)

            # Check soft constraint failures
            profiles_read = set(env.core_env.profiles_read.keys())
            all_pids = set(env.core_env.participants.keys())
            unread = all_pids - profiles_read
            if unread:
                soft_fails.append(1)
            else:
                soft_fails.append(0)

        results[task] = {
            "avg_reward": round(float(np.mean(rewards)), 3),
            "avg_score": round(float(np.mean(scores)), 3),
            "soft_constraint_fail_rate": round(float(np.mean(soft_fails)), 3),
            "avg_steps": round(float(np.mean(steps_list)), 1),
            "episodes": args.eval_episodes,
        }

    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="SchedulrX PPO Training & Evaluation")
    parser.add_argument("--steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--task", type=str, default="easy", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--rollout-len", type=int, default=256, help="Rollout buffer size")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--save-model", type=str, default="checkpoint.pt", help="Model save path")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved model")
    parser.add_argument("--model", type=str, default="checkpoint.pt", help="Model path for eval")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes for evaluation")

    args = parser.parse_args()

    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

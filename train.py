# =============================================================================
# 模块 8: 主训练循环 (Main Training Loop) —— v3
#
# 修改记录 (v2 → v3):
#   [Fix-1] checkpoint_dir 改为根据 run_name 自动创建子文件夹。
#           路径格式：checkpoints/ugv_ppo/{run_name}/
#           若未指定 run_name，则回退到 checkpoints/ugv_ppo/default/
#           这样多次实验的模型天然隔离，不会互相覆盖。
#
#   [Fix-2] no_graphics 改为从命令行参数读取，支持 exe 训练时传 True。
#           连接 Editor 时仍默认 False。
# =============================================================================

import random
import numpy as np
import torch
import argparse
import os
from typing import Optional

from config import PPOConfig
from env_wrapper import UGVSearchEnv
from ppo_algorithm import PPOAgent, RolloutBuffer
from logger import TrainingLogger
from checkpointer import Checkpointer
from evaluator import evaluate


def set_global_seed(seed: int):
    """设置全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Train] 随机种子已设置: {seed}")


def train(config: PPOConfig,
          resume: bool = False,
          checkpoint_path: Optional[str] = None,
          run_name: Optional[str] = None,
          no_graphics: bool = False):
    """
    主训练函数。

    Args:
        config          : 训练配置
        resume          : 是否从检查点续训
        checkpoint_path : 指定续训的检查点路径（None = 自动使用 latest.pth）
        run_name        : 实验名称（用于 TensorBoard 和检查点子文件夹命名）
        no_graphics     : 是否无渲染（连接 exe 时可设 True，连接 Editor 时必须 False）
    """
    set_global_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] 使用设备: {device}")

    # ------------------------------------------------------------------
    # [Fix-1] 根据 run_name 动态设置检查点子目录
    #
    # 逻辑：
    #   run_name 由命令行传入（如 "exp3_shaping_fix_seed42"）
    #   最终路径为 checkpoints/ugv_ppo/exp3_shaping_fix_seed42/
    #   若未指定 run_name，回退到 checkpoints/ugv_ppo/default/
    #   避免多次实验的模型互相覆盖。
    # ------------------------------------------------------------------
    effective_run_name = run_name if run_name else "default"
    # [M-2 修复] 用 copy 避免修改原始 config
    import copy
    config = copy.copy(config)
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, effective_run_name)
    print(f"[Train] 检查点目录: {config.checkpoint_dir}")

    env = None
    try:
        env = UGVSearchEnv(
            config,
            time_scale=20.0,
            # [Fix-2] no_graphics 从外部传入，不再硬编码
            no_graphics=no_graphics
        )

        actual_state_dim  = env.state_dim
        actual_action_dim = env.action_dim

        agent        = PPOAgent(config, state_dim=actual_state_dim)
        buffer       = RolloutBuffer(config, actual_state_dim, actual_action_dim)
        logger       = TrainingLogger(config.log_dir, run_name=effective_run_name)
        checkpointer = Checkpointer(config)

        start_rollout = 0
        if resume:
            _, start_rollout = checkpointer.load(agent, checkpoint_path)
            print(f"[Train] 从 Rollout #{start_rollout} 继续训练，global_step={agent.global_step}")
        else:
            print(f"[Train] 从头开始训练")

        total_rollouts = config.total_timesteps // config.rollout_steps

        print(f"\n[Train] 🚀 开始训练!")
        print(f"[Train] 总 Rollout 数: {total_rollouts}")
        print(f"[Train] 每 Rollout 步数: {config.rollout_steps}")
        print(f"[Train] 状态维度: {actual_state_dim},  动作维度: {actual_action_dim}\n")

        obs = env.reset()
        current_episode_reward = 0.0
        current_episode_length = 0

        for rollout_idx in range(start_rollout, total_rollouts):

            buffer.reset()

            for step in range(config.rollout_steps):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)

                current_episode_reward += reward
                current_episode_length += 1
                logger.step(reward, done)

                # [S-2 修复] 区分超时截断(truncation)和真正终止(termination)
                # 超时时 done=True 但 interrupted=True，此时 Agent 仍有未来收益，
                # GAE 应该用 Critic bootstrap 而不是强制 next_value=0。
                # 因此存入 buffer 的 done 标志：真正终止=True，超时截断=False
                terminated = done and not info.get("interrupted", False)
                buffer.add(obs, action, reward, terminated, log_prob, value)

                agent.global_step += 1
                obs = next_obs

                if done:
                    logger.log_episode(
                        agent.global_step,
                        current_episode_reward,
                        current_episode_length
                    )
                    agent.episode_count      += 1
                    current_episode_reward    = 0.0
                    current_episode_length    = 0
                    obs = env.reset()

            with torch.no_grad():
                obs_t      = torch.from_numpy(obs).unsqueeze(0).to(device)
                last_value = agent.critic(obs_t).squeeze().cpu().item()
            buffer.compute_gae(last_value=last_value)

            actor_loss, critic_loss, entropy = agent.update(buffer)

            # [O-1 新增] 学习率线性衰减（PPO 标准 trick）
            # 从 config.lr 线性下降到 0，防止训练后期策略震荡
            progress = agent.global_step / config.total_timesteps
            new_lr = config.lr * (1.0 - progress)
            new_lr = max(new_lr, 1e-7)  # 最低不为 0
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_lr


            current_lr = agent.optimizer.param_groups[0]["lr"]
            logger.log_training(
                global_step=agent.global_step,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                lr=current_lr
            )

            latest_eval_reward = None
            if (rollout_idx + 1) % config.eval_interval == 0:
                print(f"\n[Train] 开始评估 (Rollout #{rollout_idx+1})...")
                mean_r, std_r, success_rate = evaluate(
                    agent=agent,
                    env=env,
                    n_episodes=config.eval_episodes
                )
                logger.log_eval(agent.global_step, mean_r, std_r, success_rate)
                latest_eval_reward = mean_r

                obs = env.reset()
                current_episode_reward = 0.0
                current_episode_length = 0

            if (rollout_idx + 1) % config.save_interval == 0:
                checkpointer.save(
                    agent, rollout_idx + 1,
                    eval_reward=latest_eval_reward
                )

        print("\n[Train] 🎉 训练完成!")
        checkpointer.save(agent, total_rollouts)
        logger.close()

    except KeyboardInterrupt:
        print("\n[Train] ⚠️  用户中断训练（Ctrl+C）")
        if 'agent' in locals() and 'checkpointer' in locals():
            print("[Train] 正在保存中断时的检查点...")
            checkpointer.save(agent, rollout_idx if 'rollout_idx' in locals() else 0)

    finally:
        if env is not None:
            env.close()


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UGV PPO 训练脚本 v3")

    parser.add_argument("--env_path",        type=str,            default=None)
    parser.add_argument("--worker_id",       type=int,            default=0)
    parser.add_argument("--run_name",        type=str,            default=None)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--checkpoint_path", type=str,            default=None)
    parser.add_argument("--total_timesteps", type=int,            default=None)
    parser.add_argument("--lr",              type=float,          default=None)
    parser.add_argument("--seed",            type=int,            default=None)
    parser.add_argument("--device",          type=str,            default=None)
    # [Fix-2] 新增 --no_graphics 参数，连接 exe 时可加此参数
    parser.add_argument("--no_graphics",     action="store_true",
                        help="无渲染模式（连接打包 exe 时使用，连接 Editor 时不要加）")

    args = parser.parse_args()

    config = PPOConfig()
    if args.env_path        is not None: config.env_path        = args.env_path
    if args.worker_id       is not None: config.worker_id       = args.worker_id
    if args.total_timesteps is not None: config.total_timesteps = args.total_timesteps
    if args.lr              is not None: config.lr              = args.lr
    if args.seed            is not None: config.seed            = args.seed
    if args.device          is not None: config.device          = args.device

    train(
        config=config,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        run_name=args.run_name,
        no_graphics=args.no_graphics   # [Fix-2]
    )
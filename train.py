# =============================================================================
# 模块 8: 主训练循环 (Main Training Loop)
#
# 训练流程（每次 rollout）：
#   1. [采集]  与环境交互 rollout_steps 步，填充 RolloutBuffer
#   2. [计算]  调用 buffer.compute_gae() 计算 GAE 优势函数
#   3. [更新]  调用 agent.update(buffer) 执行 ppo_epochs 次策略更新
#   4. [记录]  写 TensorBoard 日志
#   5. [评估]  每 eval_interval 次 rollout 评估一次
#   6. [保存]  每 save_interval 次 rollout 保存一次检查点
#   7. [清空]  buffer.reset() 准备下一次 rollout
# =============================================================================

# =============================================================================
# 模块 8: 主训练循环 (Main Training Loop) —— v2 修复版
#
# 修复记录：
#   [Fix-1] no_graphics 默认改为 False（连接 Editor 时必须 False）
#   [Fix-2] evaluate() 调用改为传入 env 实例（配合 evaluator.py v2 修复）
#   [Fix-3] 评估后新增 obs = env.reset()，确保评估打断后训练能无缝接续
#   [Fix-4] 增加 env.close() 的 try/finally 保护，避免异常时 Unity 进程僵死
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
          run_name: Optional[str] = None):
    """
    主训练函数。

    Args:
        config          : 训练配置
        resume          : 是否从检查点续训
        checkpoint_path : 指定续训的检查点路径（None = 自动使用 latest.pth）
        run_name        : TensorBoard 运行名称
    """
    set_global_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] 使用设备: {device}")

    # ------------------------------------------------------------------
    # 初始化：env 在 try 块外声明，确保 finally 中能关闭
    # ------------------------------------------------------------------
    env = None
    try:
        # ----------------------------------------------------------------
        # 步骤 1: 启动训练环境
        #
        # 关于 no_graphics：
        #   - 连接 Unity Editor：必须 False（Editor 本身管理渲染）
        #   - 连接打包的 .exe：可设 True 提升训练速度约 20%
        # ----------------------------------------------------------------
        env = UGVSearchEnv(
            config,
            time_scale=20.0,
            no_graphics=False   # [Fix-1] 连接 Editor 时必须 False
        )

        # 用环境真实检测到的维度初始化后续组件
        actual_state_dim  = env.state_dim
        actual_action_dim = env.action_dim

        # ----------------------------------------------------------------
        # 步骤 2: 初始化核心组件
        # ----------------------------------------------------------------
        agent        = PPOAgent(config, state_dim=actual_state_dim)
        buffer       = RolloutBuffer(config, actual_state_dim, actual_action_dim)
        logger       = TrainingLogger(config.log_dir, run_name=run_name)
        checkpointer = Checkpointer(config)

        # ----------------------------------------------------------------
        # 步骤 3: 续训加载
        # ----------------------------------------------------------------
        start_rollout = 0
        if resume:
            _, start_rollout = checkpointer.load(agent, checkpoint_path)
            print(f"[Train] 从 Rollout #{start_rollout} 继续训练，global_step={agent.global_step}")
        else:
            print(f"[Train] 从头开始训练")

        # ----------------------------------------------------------------
        # 步骤 4: 主训练循环
        # ----------------------------------------------------------------
        total_rollouts = config.total_timesteps // config.rollout_steps

        print(f"\n[Train] 🚀 开始训练!")
        print(f"[Train] 总 Rollout 数: {total_rollouts}")
        print(f"[Train] 每 Rollout 步数: {config.rollout_steps}")
        print(f"[Train] 状态维度: {actual_state_dim},  动作维度: {actual_action_dim}")
        print(f"[Train] 提示：若状态维度异常大（如 22778），请检查 Unity 场景中是否挂载了摄像头传感器\n")

        obs = env.reset()
        current_episode_reward = 0.0
        current_episode_length = 0

        for rollout_idx in range(start_rollout, total_rollouts):

            # ============================================================
            # 阶段 A: 数据采集
            # ============================================================
            buffer.reset()

            for step in range(config.rollout_steps):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)

                current_episode_reward += reward
                current_episode_length += 1
                logger.step(reward, done)

                buffer.add(obs, action, reward, done, log_prob, value)
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

            # ============================================================
            # 阶段 B: 计算 GAE（Bootstrap 最后一步的 V(s)）
            # ============================================================
            with torch.no_grad():
                obs_t      = torch.from_numpy(obs).unsqueeze(0).to(device)
                last_value = agent.critic(obs_t).squeeze().cpu().item()
            buffer.compute_gae(last_value=last_value)

            # ============================================================
            # 阶段 C: 策略更新
            # ============================================================
            actor_loss, critic_loss, entropy = agent.update(buffer)

            # ============================================================
            # 阶段 D: 记录日志
            # ============================================================
            current_lr = agent.optimizer.param_groups[0]["lr"]
            logger.log_training(
                global_step=agent.global_step,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                lr=current_lr
            )

            # ============================================================
            # 阶段 E: 周期性评估
            # [Fix-2] 传入 env 实例，不在 evaluate() 内部新建环境
            # ============================================================
            latest_eval_reward = None
            if (rollout_idx + 1) % config.eval_interval == 0:
                print(f"\n[Train] 开始评估 (Rollout #{rollout_idx+1})...")
                mean_r, std_r, success_rate = evaluate(
                    agent=agent,
                    env=env,               # [Fix-2] 传入现有环境，不新建
                    n_episodes=config.eval_episodes
                )
                logger.log_eval(agent.global_step, mean_r, std_r, success_rate)
                latest_eval_reward = mean_r

                # [Fix-3] 评估会调用 env.reset()，结束后必须重新 reset 以衔接训练
                # 否则下一个 rollout 的第一步 obs 是评估最后一个 episode 的末尾状态
                obs = env.reset()
                current_episode_reward = 0.0
                current_episode_length = 0

            # ============================================================
            # 阶段 F: 周期性保存检查点
            # ============================================================
            if (rollout_idx + 1) % config.save_interval == 0:
                checkpointer.save(
                    agent, rollout_idx + 1,
                    eval_reward=latest_eval_reward
                )

        # 训练结束：保存最终模型
        print("\n[Train] 🎉 训练完成!")
        checkpointer.save(agent, total_rollouts)
        logger.close()

    except KeyboardInterrupt:
        print("\n[Train] ⚠️  用户中断训练（Ctrl+C）")
        if 'agent' in locals() and 'checkpointer' in locals():
            print("[Train] 正在保存中断时的检查点...")
            checkpointer.save(agent, rollout_idx if 'rollout_idx' in locals() else 0)

    finally:
        # [Fix-4] 无论是正常结束、异常、还是 Ctrl+C，都确保关闭 Unity
        if env is not None:
            env.close()


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UGV PPO 训练脚本 v2")

    parser.add_argument("--env_path",        type=str,   default=None)
    parser.add_argument("--worker_id",       type=int,   default=0)
    parser.add_argument("--run_name",        type=str,   default=None)
    parser.add_argument("--resume",          action="store_true")
    parser.add_argument("--checkpoint_path", type=str,   default=None)
    parser.add_argument("--total_timesteps", type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--seed",            type=int,   default=None)
    parser.add_argument("--device",          type=str,   default=None)

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
        run_name=args.run_name
    )
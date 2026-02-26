# =============================================================================
# 模块 6: 断点保存与续训 (Checkpointing & Resume)
#
# 保存内容（缺一不可，否则无法完整恢复训练状态）：
#   1. actor 和 critic 网络权重（state_dict）
#   2. 优化器状态（optimizer state_dict）
#      → 包含 Adam 的 m/v 动量，丢失则续训初期不稳定
#   3. 训练进度（global_step, episode_count）
#      → 保证 TensorBoard 横轴连续，不从 0 重新开始
#   4. 配置快照（config）
#      → 方便未来复查当时的超参数设置
# =============================================================================

import os
import torch
import glob
from typing import Optional, Tuple

from config import PPOConfig
from ppo_algorithm import PPOAgent


class Checkpointer:
    """
    模型检查点管理器。

    支持：
      - 定期保存（按 rollout 次数）
      - 保存最优模型（按评估奖励）
      - 从最新或指定检查点恢复训练
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_eval_reward = float("-inf")
        print(f"[Checkpointer] 检查点保存目录: {self.checkpoint_dir}")

    def save(self, agent: PPOAgent, rollout_idx: int,
             eval_reward: Optional[float] = None):
        """
        保存检查点。

        Args:
            agent      : PPOAgent 实例
            rollout_idx: 当前 rollout 序号（用于文件命名）
            eval_reward: 若提供，则与历史最优比较，更优时额外保存 best.pth
        """
        checkpoint = {
            "global_step"   : agent.global_step,
            "episode_count" : agent.episode_count,
            "rollout_idx"   : rollout_idx,
            "actor_state_dict"    : agent.actor.state_dict(),
            "critic_state_dict"   : agent.critic.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "config"        : self.config,  # 保存配置快照
        }

        # 定期保存（带序号）
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{rollout_idx:06d}.pth")
        torch.save(checkpoint, path)
        print(f"[Checkpointer] 💾 已保存: {path}  (step={agent.global_step})")

        # 保存最新（方便快速续训）
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

        # 保存最优（按评估奖励）
        if eval_reward is not None and eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"[Checkpointer] 🏆 新最优模型! EvalReward={eval_reward:.2f}  → {best_path}")

    def load(self, agent: PPOAgent,
             checkpoint_path: Optional[str] = None) -> Tuple[int, int]:
        """
        加载检查点，恢复训练状态。

        Args:
            agent          : PPOAgent 实例（将被就地修改）
            checkpoint_path: 指定路径；None 时自动加载最新的 latest.pth

        Returns:
            (global_step, rollout_idx)  训练恢复的起始位置
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pth")

        if not os.path.exists(checkpoint_path):
            print(f"[Checkpointer] 未找到检查点 {checkpoint_path}，将从头开始训练。")
            return 0, 0

        print(f"[Checkpointer] ♻️  加载检查点: {checkpoint_path}")
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # map_location 确保可以在不同设备间迁移（如训练用 GPU，加载用 CPU）
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 恢复网络权重
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])

        # 恢复优化器状态（Adam 动量等）
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 恢复训练进度
        agent.global_step   = checkpoint["global_step"]
        agent.episode_count = checkpoint["episode_count"]
        rollout_idx         = checkpoint["rollout_idx"]

        print(f"[Checkpointer] ✅ 恢复成功! global_step={agent.global_step}, "
              f"rollout_idx={rollout_idx}")

        return agent.global_step, rollout_idx
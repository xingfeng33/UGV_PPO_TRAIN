# =============================================================================
# 模块 7: 评估与测试 (Evaluation)
#
# 设计原则（重要！）：
#   评估与训练逻辑严格分离：
#   - 训练时：动作带随机性（从 Normal 分布采样），促进探索
#   - 评估时：动作取均值 μ（确定性策略），衡量最优性能
#
#   为何不能在训练中评估：
#   1. 训练时带噪声，评估结果不代表真实策略能力
#   2. 评估时不更新参数，若混在训练中会破坏时序一致性
# =============================================================================

# =============================================================================
# 模块 7: 评估与测试 (Evaluation) —— v2 修复版
#
# 修复记录：
#   [Fix-核心] evaluate() 不再在内部新建 UGVSearchEnv。
#
# 原始设计的致命缺陷：
#   原始 evaluator.py 在函数内部调用 UGVSearchEnv(config, worker_id=0)，
#   但训练环境已经占用了 worker_id=0 对应的端口（默认 5004），
#   导致评估时产生 UnityWorkerInUseException 端口冲突错误。
#
# 修复方案（架构层面）：
#   将已初始化的 env 实例作为参数传入 evaluate()，训练和评估共享同一个
#   Unity 环境连接，彻底消除端口冲突问题。
#   这也是 Stable Baselines3 / CleanRL 等主流框架的标准做法。
#
# 注意：
#   共享环境意味着评估会打断训练的连续性（env.reset() 会重置 Unity 场景）。
#   这在单智能体、单 Unity 实例的场景下是完全可以接受的标准做法。
#   评估结束后主循环会重新 reset 环境，继续采集数据。
# =============================================================================

import numpy as np
import torch
from typing import Tuple

from config import PPOConfig
from env_wrapper import UGVSearchEnv
from ppo_algorithm import PPOAgent


@torch.no_grad()
def evaluate(agent: PPOAgent,
             env: UGVSearchEnv,
             n_episodes: int = 5) -> Tuple[float, float, float]:
    """
    使用确定性策略（取动作均值 μ）评估当前智能体性能。

    Args:
        agent     : 待评估的 PPOAgent
        env       : 【v2修复】复用已有的环境实例，不再新建，避免端口冲突
        n_episodes: 评估的 episode 总数

    Returns:
        mean_reward : float  n_episodes 的平均累积奖励
        std_reward  : float  奖励标准差（论文中用 mean±std 表示稳定性）
        success_rate: float  成功率（reward >= SUCCESS_THRESHOLD 的比例）
    """
    device = torch.device(agent.config.device if torch.cuda.is_available() else "cpu")

    # 切换到评估模式（关闭 Dropout / BatchNorm 的训练行为）
    # 对于我们的 LayerNorm + ELU 网络，eval() 主要影响行为是确定性输出
    agent.actor.eval()
    agent.critic.eval()

    episode_rewards = []
    success_count   = 0

    # 成功阈值参考 UGVAgent.cs：SuccessReward=10，FailureBackwardReward=-10
    # reward > 5 说明至少完成了一次正面碰撞目标（成功导航）
    SUCCESS_THRESHOLD = 5.0

    print(f"[Eval] 开始评估 {n_episodes} 个 Episode（确定性策略）...")

    for ep in range(n_episodes):
        # 【关键】使用传入的环境实例，调用 reset() 重置场景
        obs  = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        while not done:
            # ----------------------------------------------------------
            # 确定性动作选择：只取高斯分布的均值 μ，不采样
            # 训练时：action = dist.sample()  ← 带随机噪声，促进探索
            # 评估时：action = mu              ← 确定性，测试最优性能
            # ----------------------------------------------------------
            obs_t  = torch.from_numpy(obs).unsqueeze(0).to(device)
            # forward() 返回 (mu, std)，评估时只用 mu，忽略 std
            mu, _  = agent.actor(obs_t)           # mu: shape=(1, action_dim)
            action = mu.squeeze(0).cpu().numpy()  # 去掉 batch 维: (action_dim,)

            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            step      += 1

        episode_rewards.append(ep_reward)
        is_success = ep_reward >= SUCCESS_THRESHOLD
        if is_success:
            success_count += 1

        print(f"  [Eval] Episode {ep+1:>2d}/{n_episodes}: "
              f"Reward={ep_reward:>7.2f}, Steps={step:>4d}, "
              f"{'✅ 成功' if is_success else '❌ 失败'}")

    # 恢复训练模式（评估完成后必须恢复，否则后续训练的 LayerNorm 行为会异常）
    agent.actor.train()
    agent.critic.train()

    mean_reward  = float(np.mean(episode_rewards))
    std_reward   = float(np.std(episode_rewards))
    success_rate = success_count / n_episodes

    return mean_reward, std_reward, success_rate
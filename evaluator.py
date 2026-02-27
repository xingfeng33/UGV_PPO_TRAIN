# =============================================================================
# 模块 7: 评估与测试 (Evaluation) —— v3 成功判定修复版
#
# 修改记录 (v2 → v3):
#   [Fix-1] 修复假阳性成功判定：不再单独依赖 ep_reward >= 5.0。
#
#   v2 的缺陷：
#     旧奖励函数（绝对值 shaping）让 Agent 原地耗满 1000 步也能得 100+ 分，
#     导致 ep_reward >= 5.0 这一条件完全失去区分力，出现大量虚假"成功"。
#
#   v3 的修复方案（双重条件缺一不可）：
#     条件 1：ep_reward >= SUCCESS_THRESHOLD（奖励达标，排除撞墙失败局）
#     条件 2：done_reason == "target"（必须是因为碰到目标而结束，非超时）
#
#   实现方式：
#     在 env.step() 返回的 info 字典中追踪终止原因。
#     env_wrapper.py 的 step() 方法在 terminal_steps.interrupted 为 False
#     时说明是正常终止（非超时截断）。
#     我们进一步通过最后一步的 reward 数值来区分"撞目标"和"撞墙"：
#       - 碰到目标（成功）：最后一步 reward == SuccessReward(10) 或其附近
#       - 碰到墙（失败）  ：最后一步 reward == FailureReward(-1) 或 -10
#       - 超时（失败）    ：done=True 且 interrupted=True（ML-Agents 超时标志）
#
#   注意：ML-Agents 在 MaxStep 到达时会将 interrupted=True 写入 TerminalSteps，
#         这是区分"正常终止"和"超时截断"的官方标志位，完全可靠。
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
        env       : 复用已有的环境实例，不再新建，避免端口冲突
        n_episodes: 评估的 episode 总数

    Returns:
        mean_reward : float  n_episodes 的平均累积奖励
        std_reward  : float  奖励标准差（论文中用 mean±std 表示稳定性）
        success_rate: float  真实成功率（必须碰到目标且非超时）
    """
    device = torch.device(agent.config.device if torch.cuda.is_available() else "cpu")

    agent.actor.eval()
    agent.critic.eval()

    episode_rewards = []
    success_count   = 0

    # ------------------------------------------------------------------
    # [Fix-1] 成功判定：奖励阈值（区分撞目标 vs 撞墙）
    # 参考 UGVAgent.cs:
    #   SuccessReward = 10，FailureBackwardReward = -10，FailureReward = -1
    # 最终 reward > 5 且非超时 → 确定是碰到目标成功
    # ------------------------------------------------------------------
    SUCCESS_THRESHOLD = 5.0

    print(f"[Eval] 开始评估 {n_episodes} 个 Episode（确定性策略）...")

    for ep in range(n_episodes):
        obs   = env.reset()
        ep_reward    = 0.0
        done         = False
        step         = 0
        # [Fix-1] 新增：追踪本局是否因超时而结束
        timed_out    = False
        # [Fix-1] 新增：追踪本局最后一步的即时 reward（用于区分终止原因）
        last_reward  = 0.0

        while not done:
            obs_t  = torch.from_numpy(obs).unsqueeze(0).to(device)
            mu, _  = agent.actor(obs_t)
            action = mu.squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)
            ep_reward   += reward
            last_reward  = reward   # [Fix-1] 记录每步 reward，done 时即为终止奖励
            step        += 1

            # [Fix-1] 从 info 中读取 ML-Agents 的超时标志
            # env_wrapper.py 的 step() 在 terminal_steps.interrupted=True 时
            # 会在 info 里写入 {"interrupted": True}
            if done and info.get("interrupted", False):
                timed_out = True

        episode_rewards.append(ep_reward)

        # ------------------------------------------------------------------
        # [Fix-1] 双重条件成功判定（核心修复）：
        #
        # 旧判定（有漏洞）：
        #   is_success = ep_reward >= SUCCESS_THRESHOLD
        #   → 在绝对值 shaping 奖励下，原地耗满 1000 步也满足此条件
        #
        # 新判定（可靠）：
        #   条件 1：timed_out == False（不是超时，说明在 MaxStep 前就终止了）
        #   条件 2：last_reward >= SUCCESS_THRESHOLD（终止原因是碰到目标，
        #           因为碰目标时 SetReward(10)，碰墙时 SetReward(-1 或 -10)）
        #
        # 数学依据：
        #   碰目标：last_reward = SetReward(10) = 10 ≥ 5  ✅
        #   碰墙  ：last_reward = SetReward(-1 或 -10) < 5  ❌
        #   超时  ：timed_out = True  ❌（无论 ep_reward 多高都判失败）
        # ------------------------------------------------------------------
        is_success = (not timed_out) and (last_reward >= SUCCESS_THRESHOLD)

        if is_success:
            success_count += 1

        # [Fix-1] 在日志中额外显示超时状态，方便诊断
        timeout_tag = " [超时]" if timed_out else ""
        print(f"  [Eval] Episode {ep+1:>2d}/{n_episodes}: "
              f"Reward={ep_reward:>7.2f}, Steps={step:>4d}, "
              f"LastR={last_reward:>6.2f}{timeout_tag}  "
              f"{'✅ 成功' if is_success else '❌ 失败'}")

    agent.actor.train()
    agent.critic.train()

    mean_reward  = float(np.mean(episode_rewards))
    std_reward   = float(np.std(episode_rewards))
    success_rate = success_count / n_episodes

    return mean_reward, std_reward, success_rate
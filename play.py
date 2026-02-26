# =============================================================================
# play.py —— 可视化推理脚本 v3（最终稳定版）
#
# 设计原则（Professional Standard）：
#   1. 从 ppo_algorithm.py 导入 Actor（不内嵌），保持代码单一来源
#   2. 从 .pth 文件中读取保存时的 config 来重建网络结构
#      → 这意味着：无论 networks 怎么改，只要 .pth 和 ppo_algorithm.py 配套
#        就能正确加载。这是学术界标准做法。
#   3. 修复键名：使用 'actor_state_dict'（与 checkpointer.py 实际保存一致）
#
# 【正确使用步骤，顺序不能颠倒！】
#   Step 1: 打开 Unity Editor，加载 UGVSearch 场景，暂时【不要】点 Play
#   Step 2: 在终端运行: python play.py
#   Step 3: 等待终端出现 "⏳ 请现在点击 Unity Play 按钮..." 提示
#   Step 4: 点击 Unity 的 ▶ Play 按钮
#   Step 5: 观察画面中小车的行为
#
# 或者使用打包的 exe（自动弹出窗口）：
#   python play.py --env_path "./builds/UGVSearch/UGVSearch.exe"
# =============================================================================

import argparse
import time
import numpy as np
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# ------------------------------------------------------------------
# 【核心设计】从训练模块导入，不内嵌
#
# 当你未来在 ppo_algorithm.py 中修改了 Actor（比如加了视觉模块），
# 这里的导入会自动使用新结构。
# 此时你只需要用新结构训练出的 .pth 来运行 play.py，完全不用改这个文件。
# ------------------------------------------------------------------
from ppo_algorithm import Actor
from config import PPOConfig


# =============================================================================
# 辅助函数：从 Unity 环境中提取并拼接所有传感器的观测
# =============================================================================

def extract_obs(steps) -> np.ndarray:
    """
    将 Unity 返回的多个传感器观测展平后拼接成一个一维向量。
    与 env_wrapper.py 中的 _extract_obs 逻辑完全一致。

    steps: DecisionSteps 或 TerminalSteps 对象
    返回: shape=(total_state_dim,) 的 float32 数组
    """
    return np.concatenate(
        [obs[0].flatten() for obs in steps.obs],
        axis=0
    ).astype(np.float32)


# =============================================================================
# 主推理函数
# =============================================================================

@torch.no_grad()  # 推理阶段关闭梯度，节省显存并加速
def play(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  🎮 UGV PPO 可视化推理脚本 v3")
    print(f"{'='*60}")
    print(f"  模型路径  : {args.checkpoint}")
    print(f"  测试局数  : {args.n_episodes}")
    print(f"  时间倍率  : {args.time_scale}x")
    print(f"  计算设备  : {device}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 步骤 1：先加载检查点（在连接 Unity 之前，尽早发现格式问题）
    #
    # 为什么先加载模型再连接环境？
    # 防止模型加载失败时，Unity 已经进入了 Play 状态需要手动退出的麻烦。
    # ------------------------------------------------------------------
    print(f"[Play] 正在加载检查点: {args.checkpoint}")

    try:
        # weights_only=False 允许加载 PPOConfig 这样的自定义 Python 对象
        checkpoint = torch.load(args.checkpoint, map_location=device,
                                weights_only=False)
    except FileNotFoundError:
        print(f"[Play] ❌ 找不到文件: {args.checkpoint}")
        print(f"[Play] 请检查路径是否正确，或者 checkpoints/ugv_ppo/ 目录下有哪些 .pth 文件")
        return

    # 打印检查点元信息
    print(f"[Play] 检查点信息:")
    print(f"  训练全局步数  : {checkpoint.get('global_step', '未知'):,}")
    print(f"  训练 Episode  : {checkpoint.get('episode_count', '未知')}")
    print(f"  Rollout 索引  : {checkpoint.get('rollout_idx', '未知')}")

    # 验证必要的键是否存在
    if 'actor_state_dict' not in checkpoint:
        print(f"\n[Play] ❌ 检查点中没有 'actor_state_dict' 键！")
        print(f"[Play] 实际存在的键: {list(checkpoint.keys())}")
        print(f"[Play] 这说明 checkpointer.py 的保存格式与 play.py 不匹配，请告知���。")
        return

    # ------------------------------------------------------------------
    # 从检查点中读取保存时的 config（网络"设计图"）
    # 这是专业做法的核心：用保存时的结构，而不是当前 config.py 的值，
    # 避免 config.py 改动后导致网络结构不匹配的问题
    # ------------------------------------------------------------------
    if 'config' in checkpoint and checkpoint['config'] is not None:
        net_config = checkpoint['config']
        print(f"  保存时网络结构: hidden={net_config.mlp_hidden_dims}, feature_dim={net_config.feature_dim}")
    else:
        print(f"  ⚠️  检查点中无 config，使用当前 config.py（若改动过可能不匹配）")
        net_config = PPOConfig()

    print(f"[Play] ✅ 检查点读取成功！\n")

    # ------------------------------------------------------------------
    # 步骤 2：连接 Unity 环境
    # 模型加载成功后再连接 Unity，避免出错后 Unity 卡在 Play 状态
    # ------------------------------------------------------------------
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(
        time_scale=args.time_scale,
        target_frame_rate=60,
        quality_level=3
    )

    print(f"[Play] 正在连接 Unity 环境... (worker_id={args.worker_id})")
    if args.env_path is None:
        # 连接 Editor 模式
        print(f"[Play] ⏳ 请现在点击 Unity Editor 中的 ▶ Play 按钮...")
        print(f"[Play]    （本脚本会等待，直到 Unity 进入 Play Mode）\n")

    env = UnityEnvironment(
        file_name=args.env_path,    # None = 连接 Editor
        worker_id=args.worker_id,
        side_channels=[engine_channel],
        no_graphics=False           # 必须开画面！
    )

    # ------------------------------------------------------------------
    # 步骤 3：初始化环境，自动检测传感器维度
    # ------------------------------------------------------------------
    env.reset()
    detected_names = list(env.behavior_specs.keys())
    behavior_name  = detected_names[0]
    behavior_spec  = env.behavior_specs[behavior_name]

    # 自动计算总状态维度，并逐一打印传感器信息
    state_dim = 0
    print(f"[Play] ✅ 环境连接成功！Behavior: '{behavior_name}'")
    print(f"\n[Play] === 传感器信息 ===")
    for i, obs_spec in enumerate(behavior_spec.observation_specs):
        dim = int(np.prod(obs_spec.shape))
        state_dim += dim
        if len(obs_spec.shape) == 3:
            # 三维 shape = 图像传感器 (C, H, W)
            sensor_type = f"📷 图像传感器 {obs_spec.shape[1]}×{obs_spec.shape[2]}×{obs_spec.shape[0]}"
        else:
            # 一维 shape = 向量或射线传感器
            sensor_type = f"📡 向量/射线传感器"
        print(f"  传感器[{i}]: shape={obs_spec.shape}, 维度={dim}  ← {sensor_type}")

    action_dim = behavior_spec.action_spec.continuous_size
    print(f"  ─────────────────────────────")
    print(f"  总状态维度: {state_dim}")
    print(f"  动作维度  : {action_dim}  (Steer + Motor)")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 步骤 4：构建 Actor 网络并加载权重
    #
    # Actor(config, state_dim) 这里用了两个来源：
    #   - config：来自 .pth 中保存的结构参数（保证结构正确）
    #   - state_dim：来自运行时实际检测的值（保证维度正确）
    # ------------------------------------------------------------------
    actor = Actor(config=net_config, state_dim=state_dim).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()  # 关键：切换到评估模式

    total_params = sum(p.numel() for p in actor.parameters())
    print(f"[Play] ✅ Actor 加载成功！网络参数量: {total_params:,}")
    print(f"[Play]    确定性推理模式（取动作均值 μ，无随机噪声）\n")
    print(f"{'─'*60}")

    # ------------------------------------------------------------------
    # 步骤 5：推理主循环
    # ------------------------------------------------------------------
    all_rewards = []
    all_steps   = []

    for ep in range(args.n_episodes):
        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)
        obs  = extract_obs(decision_steps)

        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        print(f"\n[Episode {ep+1:>2d}/{args.n_episodes}] 开始 ▶")

        while not done:
            # ──────────────────────────────────────────────────────────
            # 确定性动作选择（推理的核心）
            #
            # 训练时：action ~ Normal(μ, σ)  → 带随机噪声，促进探索
            # 推理时：action = μ              → 确定性，展示最优策略
            # ──────────────────────────────────────────────────────────
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            # obs_t.shape: (1, state_dim)，unsqueeze 添加 batch 维度

            mu, _ = actor(obs_t)
            # mu.shape: (1, action_dim)，推理只用均值 μ，丢弃标准差 σ

            action = mu.squeeze(0).cpu().numpy()
            # action.shape: (action_dim,)，squeeze 去掉 batch 维度
            # action[0] = Steer  ∈ [-1, 1]  → 负值左转，正值右转
            # action[1] = Motor  ∈ [-1, 1]  → 负值倒车，正值前进

            # 发送动作
            env.set_actions(
                behavior_name,
                ActionTuple(continuous=action[np.newaxis, :].astype(np.float32))
                # np.newaxis 将 (action_dim,) 扩展为 (1, action_dim)，Unity 需要 batch 格式
            )
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                # Episode 结束
                reward = float(terminal_steps.reward[0])
                obs    = extract_obs(terminal_steps)
                done   = True
            else:
                reward = float(decision_steps.reward[0])
                obs    = extract_obs(decision_steps)

            ep_reward += reward
            ep_steps  += 1

            # 每 100 步打印一次进度（防止刷屏）
            if ep_steps % 100 == 0:
                print(f"  Step {ep_steps:>4d} | 累计奖励: {ep_reward:>7.3f} | "
                      f"Steer={action[0]:>+5.2f}  Motor={action[1]:>+5.2f}")

            if args.step_delay > 0:
                time.sleep(args.step_delay)

        # Episode 结束统计
        is_success = ep_reward >= 5.0  # SuccessReward=10，拿到5分以上算成功
        status     = "✅ 成功（找到目标！）" if is_success else "❌ 失败"

        all_rewards.append(ep_reward)
        all_steps.append(ep_steps)

        print(f"[Episode {ep+1:>2d}] "
              f"Steps={ep_steps:>4d} | "
              f"TotalReward={ep_reward:>7.3f} | "
              f"{status}")
        print(f"{'─'*60}")

    # ------------------------------------------------------------------
    # 步骤 6：测试总结
    # ------------------------------------------------------------------
    success_n = sum(r >= 5.0 for r in all_rewards)

    print(f"\n{'='*60}")
    print(f"  📊 测试总结（{args.n_episodes} 个 Episode）")
    print(f"{'='*60}")
    print(f"  平均奖励  : {np.mean(all_rewards):>7.3f} ± {np.std(all_rewards):.3f}")
    print(f"  平均步数  : {np.mean(all_steps):>7.1f}")
    print(f"  成功率    : {success_n}/{args.n_episodes}  "
          f"({success_n / args.n_episodes * 100:.0f}%)")
    print(f"\n  各 Episode 明细:")
    for i, (r, s) in enumerate(zip(all_rewards, all_steps)):
        flag = "✅" if r >= 5.0 else "❌"
        print(f"    Episode {i+1:>2d}: "
              f"Reward={r:>7.3f},  Steps={s:>4d}  {flag}")
    print(f"{'='*60}\n")

    env.close()
    print("[Play] 环境已关闭，推理完成。")


# =============================================================================
# 命令行参数
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UGV PPO 可视化推理脚本 v3")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ugv_ppo/best.pth",
        help="权重文件路径（默认: checkpoints/ugv_ppo/best.pth）"
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default=None,
        help="Unity exe 路径，不填则连接 Editor（默认）"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="测试局数（默认: 5）"
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help="时间倍率，1.0=正常速度（默认），3.0=加速"
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="gRPC 端口偏移（默认: 0）"
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        help="每步延迟秒数（默认 0.0，设 0.05 可放慢动作方便观察）"
    )

    args = parser.parse_args()
    play(args)
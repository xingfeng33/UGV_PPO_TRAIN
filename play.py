# =============================================================================
# play.py —— 可视化推理脚本（最终版，基于源码直接对齐）
#
# 所有接口均从 GitHub 仓库 xingfeng33/UGV_PPO_TRAIN 源码直接确认：
#
# ① MLPFeatureExtractor(state_dim, hidden_dims, feature_dim)
#    → networks.py L82: def __init__(self, state_dim: int, hidden_dims: List[int], feature_dim: int)
#
# ② Actor(feature_extractor, action_dim)
#    → networks.py L190: def __init__(self, feature_extractor: BaseFeatureExtractor, action_dim: int)
#    → networks.py L227: forward 返回 mu = torch.tanh(self.mu_head(features))，已在[-1,1]
#
# ③ checkpoint 键名：'actor_state_dict'，'config'
#    → checkpointer.py L54,57
#
# ④ 观测提取：np.concatenate([obs[0].flatten() for obs in steps.obs], axis=0)
#    → env_wrapper.py L227-231（_extract_obs 方法，与本文件完全一致）
#
# 【正确使用顺序（不可颠倒！）】
#   Step 1: 打开 Unity Editor，加载 UGVSearch 场景，【不要】点 Play
#   Step 2: 在终端运行: python play.py
#   Step 3: 看到 "⏳ 请现在点击 Unity Play 按钮" 后，再点 Unity 的 ▶
#   Step 4: 观察 Unity 画面中小车的行为
#
# 使用打包的 exe（会自动弹出画面窗口）:
#   python play.py --env_path "./builds/UGVSearch/UGVSearch.exe"
# =============================================================================

import argparse
import time
import numpy as np
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# ──────────────────────────────────────────────────────────────────────────────
# 从训练代码导入，与训练时共享同一份类定义
# 接口来源（已从 GitHub 源码确认）：
#   networks.py L190: Actor(feature_extractor, action_dim)
#   networks.py L82 : MLPFeatureExtractor(state_dim, hidden_dims, feature_dim)
# ──────────────────────────────────────────────────────────────────────────────
from networks import Actor, MLPFeatureExtractor
from config import PPOConfig


# =============================================================================
# 主推理函数
# =============================================================================

@torch.no_grad()    # 推理阶段关闭梯度，节省显存
def play(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*62}")
    print(f"  🎮 UGV PPO 可视化推理脚本（最终版）")
    print(f"{'='*62}")
    print(f"  模型路径  : {args.checkpoint}")
    print(f"  测试局数  : {args.n_episodes}")
    print(f"  时间倍率  : {args.time_scale}x")
    print(f"  计算设备  : {device}")
    print(f"{'='*62}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 1：加载检查点（优先于连接 Unity）
    #
    # 原则：先验证模型文件没问题，再去点 Unity Play，
    #       避免模型有问题时 Unity 已经进入 Play 状态还要手动退出。
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[Play] 正在加载检查点: {args.checkpoint}")
    try:
        # weights_only=False：checkpoint 中含有 PPOConfig 自定义对象，必须设 False
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[Play] ❌ 找不到文件: {args.checkpoint}")
        print(f"[Play]    请确认路径正确，或检查 checkpoints/ugv_ppo/ 下有哪些 .pth 文件")
        return

    # 打印元信息，方便确认加载的是哪次训练的结果
    print(f"[Play] 检查点元信息:")
    print(f"  global_step    : {checkpoint.get('global_step', '未知'):,}")
    print(f"  episode_count  : {checkpoint.get('episode_count', '未知')}")
    print(f"  rollout_idx    : {checkpoint.get('rollout_idx', '未知')}")

    # 验证必要的键（键名来自 checkpointer.py L54）
    if 'actor_state_dict' not in checkpoint:
        print(f"\n[Play] ❌ 检查点中缺少 'actor_state_dict' 键！")
        print(f"[Play]    实际存在的键: {list(checkpoint.keys())}")
        return

    # 从检查点恢复保存时的 config（网络结构"设计图"）
    # 来源：checkpointer.py L57: "config": self.config
    if 'config' in checkpoint and checkpoint['config'] is not None:
        cfg = checkpoint['config']
        print(f"  保存时 state_dim   : {cfg.state_dim}")
        print(f"  保存时 action_dim  : {cfg.action_dim}")
        print(f"  保存时 hidden_dims : {cfg.mlp_hidden_dims}")
        print(f"  保存时 feature_dim : {cfg.feature_dim}")
    else:
        print(f"  ⚠️  检查点中无 config，回退使用当前 config.py（若结构已改动可能不匹配）")
        cfg = PPOConfig()

    print(f"[Play] ✅ 检查点读取完毕\n")

    # ─────────────────────────────────────────────────────────────────────────
    # ��骤 2：连接 Unity 环境
    # ─────────────────────────────────────────────────────────────────────────
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(
        time_scale=args.time_scale,     # 1.0 = 正常速度，方便肉眼观察
        target_frame_rate=60,
        quality_level=3                 # 提高画质，视觉调试更清晰
    )

    print(f"[Play] 正在连接 Unity 环境... (worker_id={args.worker_id})")
    if args.env_path is None:
        print(f"[Play] ⏳ 请现在点击 Unity Editor 中的 ▶ Play 按钮...")
        print(f"[Play]    脚本会持续等待，直到 Unity 进入 Play Mode\n")

    env = UnityEnvironment(
        file_name=args.env_path,        # None = 连接 Editor；有路径 = 启动 exe
        worker_id=args.worker_id,
        side_channels=[engine_channel],
        no_graphics=False               # 必须 False！这是可视化调试的目的
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 3：初始化，自动检测传感器维度
    #
    # 与 env_wrapper.py 的初始化逻辑保持一致（同样先 reset 再读 specs）
    # ─────────────────────────────────────────────────────────────────────────
    env.reset()     # 必须先 reset，behavior_specs 才有内容（env_wrapper.py L101）

    detected_names = list(env.behavior_specs.keys())
    behavior_name  = detected_names[0]  # 自动取第一个，对应 'UGV?team=0'
    behavior_spec  = env.behavior_specs[behavior_name]

    # 自动计算总状态维度（展平所有传感器后的总长度）
    # 逻辑与 env_wrapper.py L138-149 完全一致
    state_dim = 0
    print(f"[Play] ✅ 环境连接成功！Behavior: '{behavior_name}'")
    print(f"\n[Play] === 传感器信息 ===")
    for i, obs_spec in enumerate(behavior_spec.observation_specs):
        dim = int(np.prod(obs_spec.shape))
        state_dim += dim
        if len(obs_spec.shape) == 3:
            # 三维 = 图像传感器 (Channel × Height × Width)
            c, h, w = obs_spec.shape
            sensor_type = f"📷 图像传感器  {h}×{w}×{c}  共{dim}维"
        else:
            # 一维 = 向量传感器或射线感知传感器
            sensor_type = f"📡 向量/射线传感器  共{dim}维"
        print(f"  传感器[{i}]: shape={obs_spec.shape}  ← {sensor_type}")

    action_dim = behavior_spec.action_spec.continuous_size
    print(f"  {'─'*44}")
    print(f"  总状态维度 (state_dim) : {state_dim}")
    print(f"  动作维度   (action_dim): {action_dim}  → [Steer, Motor]")
    print(f"{'='*62}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 4：重建网络并加载权重
    #
    # 实例化顺序与 ppo_algorithm.py（PPOAgent.__init__）中完全一致：
    #   第一步：建 MLPFeatureExtractor
    #           → networks.py L82: __init__(self, state_dim, hidden_dims, feature_dim)
    #             注意参数名是 state_dim，不是 input_dim！
    #   第二步：把 feature_extractor 传给 Actor
    #           → networks.py L190: __init__(self, feature_extractor, action_dim)
    #
    # state_dim 用运行时检测值（最可靠），hidden_dims/feature_dim 用 checkpoint 中的 cfg
    # ─────────────────────────────────────────────────────────────────────────

    # 第一步：建特征提取器
    # 参数名 state_dim 来自 networks.py L82（已从 GitHub 源码确认）
    feature_extractor = MLPFeatureExtractor(
        state_dim=state_dim,                # 运行时实际检测到的维度
        hidden_dims=cfg.mlp_hidden_dims,    # 来自 checkpoint 保存的 config
        feature_dim=cfg.feature_dim         # 来自 checkpoint 保存的 config
    ).to(device)

    # 第二步：把 feature_extractor 传给 Actor（可插拔设计的核心）
    # 未来换视觉模块：只需把第一步换成 VisionFeatureExtractor，其余不变
    actor = Actor(
        feature_extractor=feature_extractor,    # 已构建好的特征提取器
        action_dim=action_dim                   # 运行时实际检测到的动作维度
    ).to(device)

    # 加载权重（键名 'actor_state_dict' 来自 checkpointer.py L54）
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()    # 切换评估模式：关闭 Dropout、LayerNorm 的训练行为

    total_params = sum(p.numel() for p in actor.parameters())
    print(f"[Play] ✅ Actor 加载成功！参数量: {total_params:,}")
    print(f"[Play]    推理模式：确定性策略（取动作均值 μ，无随机噪声）\n")
    print(f"{'─'*62}")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 5：推理主循环
    # ─────────────────────────────────────────────────────────────────────────
    all_rewards = []
    all_steps   = []

    for ep in range(args.n_episodes):

        # 重置环境，获取初始观测
        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)

        # 观测提取：与 env_wrapper.py 的 _extract_obs 方法（L227-231）完全一致
        obs = np.concatenate(
            [o[0].flatten() for o in decision_steps.obs], axis=0
        ).astype(np.float32)

        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        print(f"\n[Episode {ep+1:>2d}/{args.n_episodes}] ▶ 开始")

        while not done:
            # ──────────────────────────────────────────────────────────────
            # 确定性推理核心
            #
            # 训练时 (ppo_algorithm.py select_action):
            #     action ~ Normal(μ, σ)   带随机噪声 → 鼓励探索
            #
            # 推理时（此处）:
            #     action = μ              无噪声 → 展示最自信的确定性策略
            #
            # Actor.forward 返回 (mu, std)：
            #   mu 已经过 tanh 映射到 [-1, 1]（networks.py L227）
            #   推理时直接用 mu，不需要再 clamp
            # ──────────────────────────────────────────────────────────────

            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            # obs_t.shape: (state_dim,) → (1, state_dim)
            # unsqueeze(0) 添加 batch 维，神经网络要求 batch 格式输入

            mu, _ = actor(obs_t)
            # mu.shape: (1, action_dim)
            # _ 是 std，推理时丢弃（只用均值，不采样）

            action = mu.squeeze(0).cpu().numpy()
            # action.shape: (action_dim,) = (2,)
            # squeeze(0) 去掉 batch 维
            # action[0] = Steer ∈ [-1,1]：负值→左转，正值→右转
            # action[1] = Motor ∈ [-1,1]：负值→倒车，正值→前进

            # 发送动作到 Unity
            # np.newaxis 将 (2,) 扩展为 (1, 2)，Unity 需要 batch 格式
            env.set_actions(
                behavior_name,
                ActionTuple(continuous=action[np.newaxis, :].astype(np.float32))
            )
            env.step()

            # 获取执行结果
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                # Episode 结束（成功 / 碰撞 / 超过 MaxStep）
                reward = float(terminal_steps.reward[0])
                # 终止观测提取（与 env_wrapper.py _extract_obs_from_terminal L235-241 一致）
                obs = np.concatenate(
                    [o[0].flatten() for o in terminal_steps.obs], axis=0
                ).astype(np.float32)
                done = True
            else:
                # Episode 继续
                reward = float(decision_steps.reward[0])
                obs = np.concatenate(
                    [o[0].flatten() for o in decision_steps.obs], axis=0
                ).astype(np.float32)

            ep_reward += reward
            ep_steps  += 1

            # 每 100 步打印一次进度（防止刷屏）
            if ep_steps % 100 == 0:
                print(f"  Step {ep_steps:>4d} | 累计奖励: {ep_reward:>7.3f} | "
                      f"Steer={action[0]:>+5.2f}  Motor={action[1]:>+5.2f}")

            if args.step_delay > 0:
                time.sleep(args.step_delay)

        # Episode 结束汇总
        # 参考 UGVAgent.cs L29: SuccessReward = 10f
        # 拿到 ≥5 分即视为找到了目标（排除时间惩罚和零星噪声）
        is_success = ep_reward >= 5.0
        status     = "✅ 成功（找到目标！）" if is_success else "❌ 失败"

        all_rewards.append(ep_reward)
        all_steps.append(ep_steps)

        print(f"[Episode {ep+1:>2d}] "
              f"Steps={ep_steps:>4d} | "
              f"TotalReward={ep_reward:>7.3f} | "
              f"{status}")
        print(f"{'─'*62}")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 6：总结报告
    # ─────────────────────────────────────────────────────────────────────────
    success_n = sum(r >= 5.0 for r in all_rewards)

    print(f"\n{'='*62}")
    print(f"  📊 测试总结（共 {args.n_episodes} 个 Episode）")
    print(f"{'='*62}")
    print(f"  平均奖励  : {np.mean(all_rewards):>7.3f}  ±  {np.std(all_rewards):.3f}")
    print(f"  平均步数  : {np.mean(all_steps):>7.1f}")
    print(f"  成功率    : {success_n}/{args.n_episodes}  "
          f"({success_n / args.n_episodes * 100:.0f}%)")
    print(f"\n  各 Episode 明细:")
    for i, (r, s) in enumerate(zip(all_rewards, all_steps)):
        flag = "✅" if r >= 5.0 else "❌"
        print(f"    Episode {i+1:>2d}:  Reward={r:>7.3f},  Steps={s:>4d}  {flag}")
    print(f"{'='*62}\n")

    env.close()
    print("[Play] Unity 环境已关闭，推理完成。")


# =============================================================================
# 命令行参数入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UGV PPO 可视化推理脚本（最终版）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ugv_ppo/best.pth",
        help="权重文件路径\n"
             "默认: checkpoints/ugv_ppo/best.pth\n"
             "其他示例: checkpoints/ugv_ppo/latest.pth\n"
             "         checkpoints/ugv_ppo/checkpoint_000180.pth"
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default=None,
        help="Unity exe 路径\n"
             "不填 = 连接 Unity Editor（默认，需先运行此脚本再点 Editor 的 ▶）\n"
             "示例: --env_path ./builds/UGVSearch/UGVSearch.exe"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="测试 Episode 数量（默认: 5）"
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help="Unity 时间倍率\n1.0=正常速度（默认，方便肉眼观察）\n3.0=加速观察"
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="gRPC 端口偏移（默认: 0）\n若 train.py 同时运行，需改为其他值避免冲突"
    )
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.0,
        help="每步延迟秒数（默认: 0.0）\n设为 0.05 可放慢动作，便于慢速观察"
    )

    args = parser.parse_args()
    play(args)
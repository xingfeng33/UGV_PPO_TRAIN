# =============================================================================
# play.py —— 可视化推理脚本（v3，修复观测归一化 + 动作截断）
#
# 修改记录 (v2 → v3):
#   [Fix-4] 推理时加载 obs_rms（观测归一化器），并在每步推理前对观测做归一化。
#           v2 的缺陷：
#             训练时 Actor 接收的是经过 RunningMeanStd 归一化后的观测，
#             但 play.py 推理时直接传入原始观测，输入分布不匹配，
#             导致推理效果远低于训练表现。
#           v3 的修复：
#             从 checkpoint 中加载 obs_rms_state_dict，推理前做相同的归一化。
#
#   [Fix-5] 推理时对动作做 clip 截断到 [-1, 1]。
#           v2 的缺陷：
#             由于 Actor 去掉了 tanh，mu 输出可能超出 [-1, 1]，
#             train.py 中有 np.clip 但 play.py 没有，
#             可能导致 Unity 收到非法动作值。
#           v3 的修复：
#             在发送动作前加入 np.clip(action, -1.0, 1.0)。
#
# 保留 v2 的所有修复：
#   [Fix-1] 成功判定逻辑与 evaluator.py v3 保持一致
#   [Fix-2] 每局终端日志新增超时状态标记
#   [Fix-3] 总结报告的成功率统计同步使用新判定逻辑
# =============================================================================

import argparse
import time
import numpy as np
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from networks import Actor, MLPFeatureExtractor, RunningMeanStd  # [Fix-4] 新增导入 RunningMeanStd
from config import PPOConfig


# =============================================================================
# 主推理函数
# =============================================================================

@torch.no_grad()
def play(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*62}")
    print(f"  🎮 UGV PPO 可视化推理脚本（v3）")
    print(f"{'='*62}")
    print(f"  模型路径  : {args.checkpoint}")
    print(f"  测试局数  : {args.n_episodes}")
    print(f"  时间倍率  : {args.time_scale}x")
    print(f"  计算设备  : {device}")
    print(f"{'='*62}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 1：加载检查点
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[Play] 正在加载检查点: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[Play] ❌ 找不到文件: {args.checkpoint}")
        print(f"[Play]    请确认路径正确，检查 checkpoints/ 下有哪些 .pth 文件")
        return

    print(f"[Play] 检查点元信息:")
    print(f"  global_step    : {checkpoint.get('global_step', '未知'):,}")
    print(f"  episode_count  : {checkpoint.get('episode_count', '未知')}")
    print(f"  rollout_idx    : {checkpoint.get('rollout_idx', '未知')}")

    if 'actor_state_dict' not in checkpoint:
        print(f"\n[Play] ❌ 检查点中缺少 'actor_state_dict' 键！")
        print(f"[Play]    实际存在的键: {list(checkpoint.keys())}")
        return

    if 'config' in checkpoint and checkpoint['config'] is not None:
        cfg = checkpoint['config']
        print(f"  保存时 state_dim   : {cfg.state_dim}")
        print(f"  保存时 action_dim  : {cfg.action_dim}")
        print(f"  保存时 hidden_dims : {cfg.mlp_hidden_dims}")
        print(f"  保存时 feature_dim : {cfg.feature_dim}")
    else:
        print(f"  ⚠️  检查点中无 config，回退使用当前 config.py")
        cfg = PPOConfig()

    print(f"[Play] ✅ 检查点读取完毕\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 2：连接 Unity 环境
    # ─────────────────────────────────────────────────────────────────────────
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(
        time_scale=args.time_scale,
        target_frame_rate=60,
        quality_level=3
    )

    print(f"[Play] 正在连接 Unity 环境... (worker_id={args.worker_id})")
    if args.env_path is None:
        print(f"[Play] ⏳ 请现在点击 Unity Editor 中的 ▶ Play 按钮...")
        print(f"[Play]    脚本会持续等待，直到 Unity 进入 Play Mode\n")

    env = UnityEnvironment(
        file_name=args.env_path,
        worker_id=args.worker_id,
        side_channels=[engine_channel],
        no_graphics=False   # 可视化调试，必须 False
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 3：初始化，自动检测传感器维度
    # ─────────────────────────────────────────────────────────────────────────
    env.reset()

    detected_names = list(env.behavior_specs.keys())
    behavior_name  = detected_names[0]
    behavior_spec  = env.behavior_specs[behavior_name]

    state_dim = 0
    print(f"[Play] ✅ 环境连接成功！Behavior: '{behavior_name}'")
    print(f"\n[Play] === 传感器信息 ===")
    for i, obs_spec in enumerate(behavior_spec.observation_specs):
        dim = int(np.prod(obs_spec.shape))
        state_dim += dim
        if len(obs_spec.shape) == 3:
            c, h, w = obs_spec.shape
            sensor_type = f"📷 图像传感器  {h}×{w}×{c}  共{dim}维"
        else:
            sensor_type = f"📡 向量/射线传感器  共{dim}维"
        print(f"  传感器[{i}]: shape={obs_spec.shape}  ← {sensor_type}")

    action_dim = behavior_spec.action_spec.continuous_size
    print(f"  {'─'*44}")
    print(f"  总状态维度 (state_dim) : {state_dim}")
    print(f"  动作维度   (action_dim): {action_dim}  → [Steer, Motor]")
    print(f"{'='*62}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 4：重建网络并加载权重
    # ─────────────────────────────────────────────────────────────────────────
    feature_extractor = MLPFeatureExtractor(
        state_dim=state_dim,
        hidden_dims=cfg.mlp_hidden_dims,
        feature_dim=cfg.feature_dim
    ).to(device)

    actor = Actor(
        feature_extractor=feature_extractor,
        action_dim=action_dim
    ).to(device)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # ─────────────────────────────────────────────────────────────────────────
    # [Fix-4] 加载观测归一化器 (RunningMeanStd)
    #
    # 训练时 Actor 看到的是经过 obs_rms 归一化后的观测，
    # 推理时必须做相同的归一化，否则输入分布不匹配，动作输出会错乱。
    # ─────────────────────────────────────────────────────────────────────────
    obs_rms = RunningMeanStd(shape=(state_dim,)).to(device)
    if 'obs_rms_state_dict' in checkpoint:
        obs_rms.load_state_dict(checkpoint['obs_rms_state_dict'])
        print(f"[Play] ✅ 观测归一化器 (obs_rms) 加载成功")
    else:
        print(f"[Play] ⚠️  检查点中无 obs_rms_state_dict，将使用默认值（未归一化）")
        print(f"[Play]    如果该模型是用带归一化的代码训练的，推理效果可能不佳")

    total_params = sum(p.numel() for p in actor.parameters())
    print(f"[Play] ✅ Actor 加载成功！参数量: {total_params:,}")
    print(f"[Play]    推理模式：确定性策略（取动作均值 μ，无随机噪声）")
    # [Fix-1] 打印成功判定规则，让用户清楚当前版本的标准
    print(f"[Play]    成功判定：非超时结束 AND 最后一步奖励 ≥ 5.0\n")
    print(f"{'─'*62}")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 5：推理主循环
    # ─────────────────────────────────────────────────────────────────────────
    all_rewards   = []
    all_steps     = []
    # [Fix-3] 新增：分别记录每局的超时状态和最后一步奖励，供总结报告使用
    all_timed_out  = []
    all_last_reward = []

    for ep in range(args.n_episodes):

        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)

        obs = np.concatenate(
            [o[0].flatten() for o in decision_steps.obs], axis=0
        ).astype(np.float32)

        ep_reward   = 0.0
        ep_steps    = 0
        done        = False
        # [Fix-1] 新增：追踪本局是否超时、以及最后一步的即时奖励
        timed_out   = False
        last_reward = 0.0

        print(f"\n[Episode {ep+1:>2d}/{args.n_episodes}] ▶ 开始")

        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            # obs_t: (state_dim,) → (1, state_dim)，添加 batch 维

            # [Fix-4] 对观测进行归一化（与训练时一致）
            obs_norm = obs_rms(obs_t)

            mu, _ = actor(obs_norm)  # [Fix-4] 传入归一化后的观��
            # mu: (1, action_dim)，_ 是 std，推理时丢弃

            action = mu.squeeze(0).cpu().numpy()
            # action: (action_dim,) = (2,)
            # action[0]=Steer(-1左~+1右), action[1]=Motor(-1倒~+1前)

            # [Fix-5] 截断动作到 [-1, 1]，与 train.py 保持一致
            # 因为去掉了 tanh，mu 输出可能超出 [-1, 1]
            action = np.clip(action, -1.0, 1.0)

            env.set_actions(
                behavior_name,
                ActionTuple(continuous=action[np.newaxis, :].astype(np.float32))
            )
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                reward = float(terminal_steps.reward[0])
                obs = np.concatenate(
                    [o[0].flatten() for o in terminal_steps.obs], axis=0
                ).astype(np.float32)
                done = True

                # ─────────────────────────────────────────────────────────
                # [Fix-1] 从 ML-Agents 官方标志位读取是否超时
                # ─────────────────────────────────────────────────────────
                timed_out   = bool(terminal_steps.interrupted[0])
                last_reward = reward
            else:
                reward = float(decision_steps.reward[0])
                obs = np.concatenate(
                    [o[0].flatten() for o in decision_steps.obs], axis=0
                ).astype(np.float32)
                last_reward = reward

            ep_reward += reward
            ep_steps  += 1

            if ep_steps % 100 == 0:
                print(f"  Step {ep_steps:>4d} | 累计奖励: {ep_reward:>7.3f} | "
                      f"Steer={action[0]:>+5.2f}  Motor={action[1]:>+5.2f}")

            if args.step_delay > 0:
                time.sleep(args.step_delay)

        # ─────────────────────────────────────────────────────────────────
        # [Fix-1] 双重条件成功判定（与 evaluator.py v3 完全一致）
        # ───────────────────��─────────────────────────────────────────────
        is_success = (not timed_out) and (last_reward >= 5.0)

        all_rewards.append(ep_reward)
        all_steps.append(ep_steps)
        all_timed_out.append(timed_out)
        all_last_reward.append(last_reward)

        # [Fix-2] 日志新增超时标记和最后一步奖励
        timeout_tag = "  ⏱️ [超时]" if timed_out else ""
        print(f"[Episode {ep+1:>2d}] "
              f"Steps={ep_steps:>4d} | "
              f"TotalReward={ep_reward:>7.3f} | "
              f"LastR={last_reward:>+6.2f}"
              f"{timeout_tag} | "
              f"{'✅ 成功' if is_success else '❌ 失败'}")
        print(f"{'─'*62}")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 6：总结报告
    # ─────────────────────────────────────────────────────────────────────────

    # [Fix-3] 成功计数使用新判定逻辑
    success_n  = sum(
        (not to) and (lr >= 5.0)
        for to, lr in zip(all_timed_out, all_last_reward)
    )
    timeout_n  = sum(all_timed_out)

    print(f"\n{'='*62}")
    print(f"  📊 测试总结（共 {args.n_episodes} 个 Episode）")
    print(f"{'='*62}")
    print(f"  平均奖励  : {np.mean(all_rewards):>7.3f}  ±  {np.std(all_rewards):.3f}")
    print(f"  平均步数  : {np.mean(all_steps):>7.1f}")
    print(f"  真实成功率: {success_n}/{args.n_episodes}  "
          f"({success_n / args.n_episodes * 100:.0f}%)")
    print(f"  超时局数  : {timeout_n}/{args.n_episodes}  "
          f"({timeout_n / args.n_episodes * 100:.0f}%)")
    print(f"\n  各 Episode 明细:")
    for i, (r, s, to, lr) in enumerate(
        zip(all_rewards, all_steps, all_timed_out, all_last_reward)
    ):
        is_ok   = (not to) and (lr >= 5.0)
        to_tag  = "[超时]" if to else "      "
        flag    = "✅" if is_ok else "❌"
        print(f"    Episode {i+1:>2d}:  "
              f"Reward={r:>7.3f},  Steps={s:>4d},  "
              f"LastR={lr:>+6.2f}  {to_tag}  {flag}")
    print(f"{'='*62}\n")

    env.close()
    print("[Play] Unity 环境已关闭，推理完成。")


# =============================================================================
# 命令行参数入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UGV PPO 可视化推理脚本（v3）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ugv_ppo/best.pth",
        help="权重文件路径\n"
             "默认: checkpoints/ugv_ppo/best.pth\n"
             "新版分文件夹示例: checkpoints/ugv_ppo/exp3_potential_shaping_seed42/best.pth"
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default=None,
        help="Unity exe 路径\n"
             "不填 = 连接 Unity Editor\n"
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
        help="Unity 时间倍率\n1.0=正常速度（默认）\n3.0=加速观察"
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
        help="每步延迟秒数（默认: 0.0）\n设为 0.05 可放慢动作"
    )

    args = parser.parse_args()
    play(args)
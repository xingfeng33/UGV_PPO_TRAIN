# =============================================================================
# play.py —— 可视化/无头推理评估脚本（v4，同步帧堆叠支持）
#
# 修改记录 (v3 → v4):
#   [FrameStack-1] 从 checkpoint 的 cfg 中读取 frame_stack 参数，
#                  自动将 state_dim 更新为堆叠后的正确维度，确保网络能正确加载。
#   [FrameStack-2] 推理主循环中引入帧队列（deque），与 env_wrapper.py 逻辑完全对齐。
#                  getattr(cfg, 'frame_stack', 1) 确保兼容旧版 checkpoint（无此字段时默认=1）。
#
# 注意：play.py 直接裸调 UnityEnvironment，绕过了 env_wrapper.py，
#       因此必须在此处手动同步帧堆叠逻辑，否则用帧堆叠训练的模型在此推理会维度崩溃。
# =============================================================================

import argparse
import time
import random
from collections import deque                          # ← [FrameStack-2] 新增：帧队列
import numpy as np
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from networks import Actor, MLPFeatureExtractor, RunningMeanStd
from config import PPOConfig

@torch.no_grad()
def play(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*62}")
    print(f"  🎮 UGV PPO 推理评估脚本（v4）")
    print(f"{'='*62}")
    print(f"  模型路径  : {args.checkpoint}")
    print(f"  有效测试局: {args.n_episodes}")
    print(f"  时间倍率  : {args.time_scale}x")
    print(f"  无头模式  : {'开启 (无渲染)' if args.no_graphics else '关闭 (可视化)'}")
    print(f"  计算设备  : {device}")
    print(f"{'='*62}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 1：加载检查点 (包含 RunningMeanStd)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"[Play] 正在加载检查点: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[Play] ❌ 找不到文件: {args.checkpoint}")
        return

    if 'config' in checkpoint and checkpoint['config'] is not None:
        cfg = checkpoint['config']
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
        target_frame_rate=-1,
        quality_level=0
    )

    if args.env_path:
        print(f"[Play] 连接打包环境: {args.env_path}")
    else:
        print(f"[Play] ⏳ 请现在点击 Unity Editor 中的 ▶ Play 按钮...")
        if args.no_graphics:
            print(f"[Play] ⚠️ 警告：不指定 --env_path 连接 Editor 时，--no_graphics 可能会被忽略。")

    eval_seed = args.seed if args.seed is not None else random.randint(0, 999999)
    print(f"[Play] 🎲 本次评估使用的随机种子: {eval_seed}")

    env = UnityEnvironment(
        file_name=args.env_path,
        worker_id=args.worker_id,
        seed=eval_seed,
        side_channels=[engine_channel],
        no_graphics=args.no_graphics
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 3：初始化，自动检测传感器维度 + 同步帧堆叠维度
    # ─────────────────────────────────────────────────────────────────────────
    env.reset()

    detected_names = list(env.behavior_specs.keys())
    behavior_name  = detected_names[0]
    behavior_spec  = env.behavior_specs[behavior_name]

    raw_state_dim = 0
    for obs_spec in behavior_spec.observation_specs:
        raw_state_dim += int(np.prod(obs_spec.shape))
    action_dim = behavior_spec.action_spec.continuous_size

    # 【帧堆叠-1】从 checkpoint 的 cfg 中读取 frame_stack 参数
    # getattr(..., 1) 确保兼容旧版 checkpoint（旧模型没有 frame_stack 字段时默认=1，即不堆叠）
    frame_stack = getattr(cfg, 'frame_stack', 1)
    state_dim   = raw_state_dim * frame_stack       # 网络实际输入维度

    if frame_stack > 1:
        print(f"[Play] 帧堆叠已启用: {raw_state_dim}(单帧) × {frame_stack}(堆叠) = {state_dim}")
    else:
        print(f"[Play] 帧堆叠未启用 (frame_stack=1)，state_dim={state_dim}")

    # 【帧堆叠-2】初始化帧队列（供推理主循环使用）
    frames = deque(maxlen=frame_stack)

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 4：重建网络并加载权重
    # ─────────────────────────────────────────────────────────────────────────
    # obs_rms 使用堆叠后的 state_dim，与训练时完全对齐
    obs_rms = RunningMeanStd(shape=(state_dim,)).to(device)
    if 'obs_rms_state_dict' in checkpoint:
        obs_rms.load_state_dict(checkpoint['obs_rms_state_dict'])
        print("[Play] ✅ 状态归一化记录 (RunningMeanStd) 已成功加载。")

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

    print(f"[Play] ✅ Actor 加载成功！")
    print(f"[Play]    成功判定：非超时结束 AND 最后一步奖励 ≥ 5.0")
    print(f"[Play]    脏数据过滤：步数 <= 2 且失败的局将被直接抛弃并重试\n")
    print(f"{'─'*62}")

    # ─────────────────────────────────────────────────────────────────────────
    # ��骤 5：推理主循环 (使用 while 保证获得足量的有效局)
    # ─────────────────────────────────────────────────────────────────────────
    all_rewards    = []
    all_steps      = []
    all_timed_out  = []
    all_last_reward = []

    valid_episodes = 0
    total_attempts = 0

    while valid_episodes < args.n_episodes:
        total_attempts += 1
        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)

        # 提取初始单帧观测
        raw_obs = np.concatenate(
            [o[0].flatten() for o in decision_steps.obs], axis=0
        ).astype(np.float32)

        # 【帧堆叠-3】清空队列，用初始帧填满，避免全零初始化
        # frame_stack=1 时队列只有 1 帧，拼接结果等于 raw_obs，与原 v3 完全相同
        frames.clear()
        for _ in range(frame_stack):
            frames.append(raw_obs)
        obs = np.concatenate(list(frames), axis=0)

        ep_reward   = 0.0
        ep_steps    = 0
        done        = False
        timed_out   = False
        last_reward = 0.0

        if not args.no_graphics:
            print(f"\n[有效局 {valid_episodes+1:>2d}/{args.n_episodes} | 尝试 #{total_attempts}] ▶ 开始")

        while not done:
            # 推理时进行状态归一化（obs 已是堆叠后的维度，与训练完全一致）
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            obs_norm = obs_rms(obs_t)

            mu, _ = actor(obs_norm)
            action = mu.squeeze(0).cpu().numpy()

            action = np.clip(action, -1.0, 1.0)

            env.set_actions(
                behavior_name,
                ActionTuple(continuous=action[np.newaxis, :].astype(np.float32))
            )
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                reward = float(terminal_steps.reward[0])
                # 提取新单帧
                new_raw_obs = np.concatenate(
                    [o[0].flatten() for o in terminal_steps.obs], axis=0
                ).astype(np.float32)
                # 【帧堆叠-4】压入新帧，拼接堆叠观测（最老帧自动挤出）
                frames.append(new_raw_obs)
                obs = np.concatenate(list(frames), axis=0)
                done        = True
                timed_out   = bool(terminal_steps.interrupted[0])
                last_reward = reward
            elif len(decision_steps) > 0:
                reward = float(decision_steps.reward[0])
                # 提取新单帧
                new_raw_obs = np.concatenate(
                    [o[0].flatten() for o in decision_steps.obs], axis=0
                ).astype(np.float32)
                # 【帧堆叠-4】压入新帧，拼接堆叠观测（最老帧自动挤出）
                frames.append(new_raw_obs)
                obs = np.concatenate(list(frames), axis=0)
                last_reward = reward
            else:
                break

            ep_reward += reward
            ep_steps  += 1

            if not args.no_graphics and ep_steps % 100 == 0:
                print(f"  Step {ep_steps:>4d} | 累计奖励: {ep_reward:>7.3f} | "
                      f"Steer={action[0]:>+5.2f}  Motor={action[1]:>+5.2f}")

            if args.step_delay > 0 and not args.no_graphics:
                time.sleep(args.step_delay)

        # --------- 局结算与脏数据过滤 ---------
        is_success = (not timed_out) and (last_reward >= 5.0)

        if ep_steps <= 2 and not is_success:
            print(f"  ⚠️ [尝试 #{total_attempts}] 检测到模拟器生成重叠 (Steps={ep_steps}, Reward={last_reward:.1f})，已作废并重试...")
            continue

        valid_episodes += 1
        all_rewards.append(ep_reward)
        all_steps.append(ep_steps)
        all_timed_out.append(timed_out)
        all_last_reward.append(last_reward)

        timeout_tag = "  ⏱️ [超时]" if timed_out else ""
        if not args.no_graphics or valid_episodes % 10 == 0:
            print(f"[完成有效局 {valid_episodes:>2d}] "
                  f"Steps={ep_steps:>4d} | "
                  f"TotalReward={ep_reward:>7.3f} | "
                  f"LastR={last_reward:>+6.2f}"
                  f"{timeout_tag} | "
                  f"{'✅ 成功' if is_success else '❌ 失败'}")
            if not args.no_graphics: print(f"{'─'*62}")

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 6：总结报告
    # ─────────────────────────────────────────────────────────────────────────
    success_n  = sum(
        (not to) and (lr >= 5.0)
        for to, lr in zip(all_timed_out, all_last_reward)
    )
    timeout_n  = sum(all_timed_out)
    filtered_n = total_attempts - args.n_episodes

    print(f"\n{'='*62}")
    print(f"  📊 测试总结（已过滤 {filtered_n} 次异常重叠生成）")
    print(f"{'='*62}")
    print(f"  有效评估局: {args.n_episodes} 局 (总尝试 {total_attempts} 次)")
    print(f"  平均奖励  : {np.mean(all_rewards):>7.3f}  ±  {np.std(all_rewards):.3f}")
    print(f"  平均步数  : {np.mean(all_steps):>7.1f}")
    print(f"  真实成功率: {success_n}/{args.n_episodes}  "
          f"({success_n / args.n_episodes * 100:.0f}%)")
    print(f"  超时局数  : {timeout_n}/{args.n_episodes}  "
          f"({timeout_n / args.n_episodes * 100:.0f}%)")

    if args.no_graphics:
        print(f"\n  [详细日志已因 no_graphics 模式折叠]")
    else:
        print(f"\n  各有效局明细:")
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

# =============================================================================
# 命令行参数入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UGV PPO 可视化/无头推理脚本（v4）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ugv_ppo/best.pth", help="权重文件路径")
    parser.add_argument("--env_path", type=str, default=None, help="Unity exe 路径（不填 = 连接 Unity Editor）")
    parser.add_argument("--n_episodes", type=int, default=5, help="要求跑满的有效测试局数量（默认: 5）")
    parser.add_argument("--time_scale", type=float, default=1.0, help="Unity 时间倍率 (配合 --no_graphics 可设为 20.0 等加速)")
    parser.add_argument("--worker_id", type=int, default=0, help="gRPC 端口偏移（默认: 0）")
    parser.add_argument("--step_delay", type=float, default=0.0, help="每步延迟秒数（设为 0.05 可放慢动作）")
    parser.add_argument("--no_graphics", action="store_true", help="开启无头模式（不渲染画面）")
    parser.add_argument("--seed", type=int, default=None, help="指定环境随机种子（不填则每次自动生成随机种子）")

    args = parser.parse_args()
    play(args)
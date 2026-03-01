# =============================================================================
# play.py —— 可视化/无头推理评估脚本（v3，新增脏数据过滤与无头模式）
#
# 修改记录 (v2 → v3):
#   [Feature-1] 引入脏数据过滤：如果步数 <= 2 且失败，认定为 Unity 出生重叠 Bug，
#               直接作废该局，不计入评估统计，直到跑满 n_episodes 个有效局。
#   [Feature-2] 新增 --no_graphics 支持：允许在连接 exe 时关闭渲染，大幅提升评估速度。
#   [Feature-3] 引入随机种子/指定种子：打破固定生成规律。
# =============================================================================

import argparse
import time
import random
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
    print(f"  🎮 UGV PPO 推理评估脚本（v3）")
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
        # weights_only=False 确保可以加载 config 对象
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
        target_frame_rate=-1 if args.no_graphics else 60, # 无渲染时解除帧率限制
        quality_level=0 if args.no_graphics else 3        # 无渲染时降低画质级别
    )

    print(f"[Play] 正在连接 Unity 环境... (worker_id={args.worker_id})")
    if args.env_path is None:
        print(f"[Play] ⏳ 请现在点击 Unity Editor 中的 ▶ Play 按钮...")
        if args.no_graphics:
            print(f"[Play] ⚠️ 警告：不指定 --env_path 连接 Editor 时，--no_graphics 可能会被忽略。")
    
    # 确定随机种子
    eval_seed = args.seed if args.seed is not None else random.randint(0, 999999)
    print(f"[Play] 🎲 本次评估使用的随机种子: {eval_seed}")

    env = UnityEnvironment(
        file_name=args.env_path,
        worker_id=args.worker_id,
        seed=eval_seed,                  # 传入随机种子，打破时间循环
        side_channels=[engine_channel],
        no_graphics=args.no_graphics     # 开启/关闭无头模式
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 3：初始化，自动检测传感器维度
    # ─────────────────────────────────────────────────────────────────────────
    env.reset()

    detected_names = list(env.behavior_specs.keys())
    behavior_name  = detected_names[0]
    behavior_spec  = env.behavior_specs[behavior_name]

    state_dim = 0
    for obs_spec in behavior_spec.observation_specs:
        state_dim += int(np.prod(obs_spec.shape))
    action_dim = behavior_spec.action_spec.continuous_size

    # ─────────────────────────────────────────────────────────────────────────
    # 步骤 4：重建网络并加载权重
    # ─────────────────────────────────────────────────────────────────────────
    # 加载归一化模块 (极其重要，否则推理完全乱套)
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
    # 步骤 5：推理主循环 (使用 while 保证获得足量的有效局)
    # ─────────────────────────────────────────────────────────────────────────
    all_rewards   = []
    all_steps     = []
    all_timed_out  = []
    all_last_reward = []

    valid_episodes = 0
    total_attempts = 0  # 记录总共尝试了多少次（包含脏数据）

    # 使用 while 循环直到收集够有效的数据
    while valid_episodes < args.n_episodes:
        total_attempts += 1
        env.reset()
        decision_steps, _ = env.get_steps(behavior_name)

        obs = np.concatenate(
            [o[0].flatten() for o in decision_steps.obs], axis=0
        ).astype(np.float32)

        ep_reward   = 0.0
        ep_steps    = 0
        done        = False
        timed_out   = False
        last_reward = 0.0

        if not args.no_graphics:
            print(f"\n[有效局 {valid_episodes+1:>2d}/{args.n_episodes} | 尝试 #{total_attempts}] ▶ 开始")

        while not done:
            # 推理时进行状态归一化
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            obs_norm = obs_rms(obs_t)

            mu, _ = actor(obs_norm)
            action = mu.squeeze(0).cpu().numpy()
            
            # 【重要】推理时必须截断动作，防止超过物理边界
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
                timed_out   = bool(terminal_steps.interrupted[0])
                last_reward = reward
            elif len(decision_steps) > 0:
                reward = float(decision_steps.reward[0])
                obs = np.concatenate(
                    [o[0].flatten() for o in decision_steps.obs], axis=0
                ).astype(np.float32)
                last_reward = reward
            else:
                # 极端异常情况保护
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

        # 脏数据过滤器 (模拟真实世界无重叠的安全摆放)
        if ep_steps <= 2 and not is_success:
            print(f"  ⚠️ [尝试 #{total_attempts}] 检测到模拟器生成重叠 (Steps={ep_steps}, Reward={last_reward:.1f})，已作废并重试...")
            continue  # 跳过统计，直接进入下一个 while 循环

        # 只有走到这里，才算一个有效局
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
        description="UGV PPO 可视化/无头推理脚本（v3）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ugv_ppo/best.pth",
        help="权重文件路径"
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default=None,
        help="Unity exe 路径（不填 = 连接 Unity Editor）"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=5,
        help="要求跑满的有效测试局数量（默认: 5）"
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help="Unity 时间倍率 (配合 --no_graphics 可设为 20.0 等加速)"
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
        help="每步延迟秒数（设为 0.05 可放慢动作）"
    )
    parser.add_argument(
        "--no_graphics",
        action="store_true",
        help="开启无头模式（不渲染画面）。注意：仅对指定的 exe 环境生效，能极速跑完 100 局评估！"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="指定环境随机种子（不填则每次自动生成随机种子）"
    )

    args = parser.parse_args()
    play(args)
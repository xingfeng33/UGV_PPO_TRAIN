# =============================================================================
# 模块 2: 环境交互与封装 (Environment Wrapper) —— v3 帧堆叠版
#
# 修改记录 (v2 → v3):
#   [FrameStack-1] 引入 collections.deque，实现帧堆叠（Frame Stacking）机制。
#                  通过 config.frame_stack 参数灵活控制：
#                    - frame_stack=1：行为与 v2 完全相同，向后兼容
#                    - frame_stack=4：将最近 4 帧拼接后输出，解决 POMDP 遮挡问题
#   [FrameStack-2] __init__ 中更新 self.state_dim 为堆叠后的真实维度，
#                  保留 self.real_state_dim 记录单帧维度，供调试使用。
#   [FrameStack-3] reset() 中用初始帧填满队列，避免全零初始化导致的分布偏移。
#   [FrameStack-4] step() 中每步将新帧压入队列，自动挤出最老帧，拼接后返回。
# =============================================================================

# =============================================================================
# 模块 2: 环境交互与封装 (Environment Wrapper) —— v2 修复版
#
# 修复记录：
#   [Fix-1] 将 _env.reset() 移到 behavior_specs 访问之前（原始代码 Bug）
#   [Fix-2] 修复 behavior_name 赋值时机问题（AttributeError）
#   [Fix-3] 使用模糊匹配而非直接取 [0]，更健壮地处理 'UGV?team=0' 格式
#   [Fix-4] no_graphics 改为构造参数，调用方自行决定，不在类内硬编码
#   [Fix-5] 增加 worker_id 参数，支持评估时使用不同端口（根本性架构修复）
# =============================================================================

import numpy as np
from collections import deque                          # ← [FrameStack-1] 新增：帧队列
from typing import Tuple, Optional, Dict, Any

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from config import PPOConfig


class UGVSearchEnv:
    """
    Unity UGVSearch 环境的 Gym 风格封装器。

    v3 新增说明：
    - 支持帧堆叠（Frame Stacking），通过 config.frame_stack 控制
    - frame_stack=1 时完全等同于 v2，无任何额外开销

    v2 修复说明：
    - behavior_name 现在在 reset() 之后自动检测，不依赖 config 中的字符串精确匹配
    - worker_id 可从外部传入，解决训练/评估端口冲突问题
    """

    def __init__(self, config: PPOConfig,
                 time_scale: float = 20.0,
                 no_graphics: bool = False,
                 worker_id: Optional[int] = None):
        """
        Args:
            config      : PPOConfig 配置对象
            time_scale  : Unity 时间加速倍率
                          - 连接 Editor 调试：建议 1.0~5.0
                          - 连接打包的 .exe 训练：可设 20.0
            no_graphics : 是否无渲染模式
                          - 连接 Editor 时：必须 False（Editor 本身有渲染）
                          - 连接打包的 .exe 时：可设 True 加速
            worker_id   : gRPC 端口偏移量（base_port + worker_id）
                          - None 时使用 config.worker_id
                          - 评估环境传入不同值（如 config.worker_id + 1）避免端口冲突
        """
        self.config = config

        # 确定本实例使用的 worker_id（外部传入优先）
        effective_worker_id = worker_id if worker_id is not None else config.worker_id

        # ------------------------------------------------------------------
        # 步骤 1: 配置 Unity 引擎参数
        # ------------------------------------------------------------------
        engine_channel = EngineConfigurationChannel()
        engine_channel.set_configuration_parameters(
            time_scale=time_scale,
            target_frame_rate=-1,
            quality_level=0
        )

        # ------------------------------------------------------------------
        # 步骤 2: 启动/连接 Unity 环境
        # ------------------------------------------------------------------
        self._env = UnityEnvironment(
            file_name=config.env_path,
            worker_id=effective_worker_id,
            side_channels=[engine_channel],
            no_graphics=no_graphics
        )

        # ------------------------------------------------------------------
        # 步骤 3: 【关键修复】先 reset()，才能拉取 behavior_specs
        # ------------------------------------------------------------------
        self._env.reset()

        # ------------------------------------------------------------------
        # 步骤 4: 自动检测真实的 behavior 名称（模糊匹配，处理 'UGV?team=0' 格式）
        # ------------------------------------------------------------------
        detected_names = list(self._env.behavior_specs.keys())
        print(f"[Env] 检测到的 Behavior 名称列表: {detected_names}")

        if len(detected_names) == 0:
            raise RuntimeError("[Env] ❌ Unity 没有传回任何 Behavior！请检查 Unity 场景中是否有 Agent 且已启动 PlayMode。")

        matched = [name for name in detected_names if config.behavior_name in name]

        if len(matched) > 0:
            self.behavior_name = matched[0]
            print(f"[Env] 模糊匹配成功: config='{config.behavior_name}' → 实际='{self.behavior_name}'")
        else:
            self.behavior_name = detected_names[0]
            print(f"[Env] ⚠️  未找到包含 '{config.behavior_name}' 的名称，已自动使用第一个: '{self.behavior_name}'")
            print(f"[Env] ⚠️  建议将 config.behavior_name 改为 '{self.behavior_name.split('?')[0]}'")

        # ------------------------------------------------------------------
        # 步骤 5: 获取并打印环境规格（状态/动作维度）
        # ------------------------------------------------------------------
        behavior_spec = self._env.behavior_specs[self.behavior_name]

        print(f"\n[Env] === 传感器维度详情 ===")
        total_state_dim = 0
        for i, obs_spec in enumerate(behavior_spec.observation_specs):
            obs_dim = int(np.prod(obs_spec.shape))
            total_state_dim += obs_dim
            print(f"  传感器[{i}]: shape={obs_spec.shape}, 展平维度={obs_dim}")

        print(f"  单帧总维度: {total_state_dim}")
        print(f"  Config 中设置的 state_dim: {config.state_dim}")

        if total_state_dim != config.state_dim:
            print(f"\n[Env] ⚠️  维度不符！实际={total_state_dim}, config={config.state_dim}")
            print(f"[Env] ⚠️  若有图像传感器维度 (如 H×W×C)，说明场景中挂载了摄像头。")
            print(f"[Env] ⚠️  如暂时只想用向量观测训练，请在 Unity 中关闭/移除 Camera Sensor 组件。")
            print(f"[Env] ⚠️  已自动使用实际维度 {total_state_dim} 继续运行。\n")

        # ------------------------------------------------------------------
        # [FrameStack-2] 初始化帧堆叠机制
        #
        # self.real_state_dim : 单帧真实维度（保留用于调试，不变）
        # self.state_dim      : 堆叠后的网络输入维度（= real_state_dim * frame_stack）
        #                       train.py 通过 actual_state_dim = env.state_dim 获取此值，
        #                       从而正确初始化 PPOAgent 网络的第一层 Linear 层
        # frame_stack=1 时 state_dim == real_state_dim，与 v2 行为完全相同
        # ------------------------------------------------------------------
        self.real_state_dim = total_state_dim          # 单帧维度（调试用）
        self.frame_stack    = config.frame_stack       # 帧堆叠数（1=不启用）
        self.state_dim      = total_state_dim * self.frame_stack  # 网络实际输入维度

        # 初始化双端队列，maxlen 保证超出时自动挤出最老帧
        self.frames = deque(maxlen=self.frame_stack)

        self.real_action_dim = behavior_spec.action_spec.continuous_size
        self.action_dim = self.real_action_dim

        # 打印最终确认信息
        if self.frame_stack > 1:
            print(f"[Env] 状态维度: {self.real_state_dim}(单帧) × {self.frame_stack}(堆叠) = {self.state_dim}  ← 帧堆叠已启用")
        else:
            print(f"[Env] 状态维度: {self.state_dim}  (帧堆叠未启用，frame_stack=1)")
        print(f"[Env] 动作维度: {self.action_dim}  (连续动作: Steer + Motor)")
        print(f"[Env] ✅ 环境初始化完成! Behavior='{self.behavior_name}'\n")

        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """重置环境，返回初始堆叠观测。

        [FrameStack-3] 重置时用初始单帧填满整个队列，避免全零初始化带来的分布偏移。
        frame_stack=1 时：返回 shape=(real_state_dim,)，与 v2 完全相同。
        frame_stack=4 时：返回 shape=(real_state_dim*4,)。
        """
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.behavior_name)
        obs = self._extract_obs(decision_steps)   # 单帧，shape=(real_state_dim,)
        self._last_obs = obs

        # 【帧堆叠】清空队列，用初始帧重复填满
        # 这样避免了开局前几帧全是零向量导致的网络输入分布异常
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs)

        # 拼接所有帧后返回；frame_stack=1 时等价于直接返回 obs
        return np.concatenate(list(self.frames), axis=0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """发送动作，返回 (next_stacked_obs, reward, done, info)。

        [FrameStack-4] 每步将新单帧压入队列右侧，最老帧自动从左侧挤出，
                       拼接后作为网络输入返回。
        frame_stack=1 时：队列始终只有 1 帧，行为与 v2 完全相同。

        action: shape=(action_dim,)
            action[0] = Steer  ∈ [-1, 1]
            action[1] = Motor  ∈ [-1, 1]
        """
        # 增加 batch 维度：(action_dim,) → (1, action_dim)
        action_batch = action[np.newaxis, :].astype(np.float32)
        action_tuple = ActionTuple(continuous=action_batch)
        self._env.set_actions(self.behavior_name, action_tuple)
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self.behavior_name)

        if len(terminal_steps) > 0:
            next_obs = self._extract_obs_from_terminal(terminal_steps)
            reward   = float(terminal_steps.reward[0])
            done     = True
            info     = {"interrupted": bool(terminal_steps.interrupted[0])}
        elif len(decision_steps) > 0:
            next_obs = self._extract_obs(decision_steps)
            reward   = float(decision_steps.reward[0])
            done     = False
            info     = {}
        else:
            # [M-4 修复] 极端情况：两者都为空，返回上一次观测
            next_obs = self._last_obs
            reward   = 0.0
            done     = False
            info     = {"empty_steps": True}

        self._last_obs = next_obs

        # 【帧堆叠】将新帧压入队列右侧，最老帧自动从左侧挤出（deque maxlen 机制）
        # frame_stack=1 时队列始终只有 1 个元素，拼接结果等于 next_obs 本身
        self.frames.append(next_obs)
        stacked_obs = np.concatenate(list(self.frames), axis=0)

        return stacked_obs, reward, done, info

    def close(self):
        """安全关闭 Unity 环境"""
        try:
            self._env.close()
            print(f"[Env] Unity 环境已关闭 (behavior='{self.behavior_name}')")
        except Exception as e:
            print(f"[Env] 关闭环境时出现异常（可忽略）: {e}")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _extract_obs(self, decision_steps) -> np.ndarray:
        """提取并展平拼接所有传感器的观测，返回单帧，shape=(real_state_dim,)"""
        obs_list = decision_steps.obs
        flat_obs = np.concatenate(
            [obs[0].flatten() for obs in obs_list],
            axis=0
        ).astype(np.float32)
        return flat_obs

    def _extract_obs_from_terminal(self, terminal_steps) -> np.ndarray:
        """从 TerminalSteps 中提取观测（与 _extract_obs 对称），返回单帧"""
        obs_list = terminal_steps.obs
        flat_obs = np.concatenate(
            [obs[0].flatten() for obs in obs_list],
            axis=0
        ).astype(np.float32)
        return flat_obs
# =============================================================================
# 模块 2: 环境交互与封装 (Environment Wrapper)
#
# 核心职责:
#   1. 启动/连接 Unity ML-Agents 环境
#   2. 将 Unity 原生 API（DecisionSteps / TerminalSteps）
#      封装成类 Gym 接口: reset() / step() -> (obs, reward, done, info)
#   3. 自动检测并打印状态/动作空间维度，供用户校验 config.py 中的 state_dim
#
# 重要架构说明:
#   ML-Agents 的数据格式：
#   - obs 是一个列表，每个元素对应一种传感器
#   - obs[0]: 向量传感器（6维基础 + 射线展开后的1D拼接）
#            shape = (n_agents, state_dim)
#   - 在单智能体场景下 n_agents=1，我们取 [0] 去掉 batch 维度
# =============================================================================

# =============================================================================
# 模块 2: 环境交互与封装 (Environment Wrapper) —— v2 修复版
#
# 修复记录：
#   [Fix-1] 将 _env.reset() ���到 behavior_specs 访问之前（原始代码 Bug）
#   [Fix-2] 修复 behavior_name 赋值时机问题（AttributeError）
#   [Fix-3] 使用模糊匹配而非直接取 [0]，更健壮地处理 'UGV?team=0' 格式
#   [Fix-4] no_graphics 改为构造参数，调用方自行决定，不在类内硬编码
#   [Fix-5] 增加 worker_id 参数，支持评估时使用不同端口（根本性架构修复）
# =============================================================================

import numpy as np
from typing import Tuple, Optional, Dict, Any

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from config import PPOConfig


class UGVSearchEnv:
    """
    Unity UGVSearch 环境的 Gym 风格封装器。

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
                          - 注意：使用独立端口时，Unity Editor 端也需要对应配置，
                            或使用打包的第二个 Unity 实例
        """
        self.config = config

        # 确定本实例使用的 worker_id
        # 外部传入优先，否则使用 config 中的默认值
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
        print(f"[Env] 正在连接 Unity 环境... worker_id={effective_worker_id}")
        self._env = UnityEnvironment(
            file_name=config.env_path,
            worker_id=effective_worker_id,
            side_channels=[engine_channel],
            no_graphics=no_graphics
        )

        # ------------------------------------------------------------------
        # 步骤 3: 【关键修复】先 reset()，才能拉取 behavior_specs
        #
        # ML-Agents 的 behavior_specs 字典在 reset() 之前是空的。
        # Unity 需要先进入 PlayMode 并完成一帧初始化，
        # Python 端才能通过 gRPC 收到完整的环境描述信息。
        # ------------------------------------------------------------------
        self._env.reset()

        # ------------------------------------------------------------------
        # 步骤 4: 自动检测真实的 behavior 名称
        #
        # Unity ML-Agents 会在用户设置的名称后自动追加 "?team=0"
        # 例如：Unity 中设置 "UGV" → Python 端收到 "UGV?team=0"
        # 因此不能直接用 config.behavior_name 去精确匹配，
        # 而应该用"包含关系"做模糊匹配，找到对应的真实名称
        # ------------------------------------------------------------------
        detected_names = list(self._env.behavior_specs.keys())
        print(f"[Env] 检测到的 Behavior 名称列表: {detected_names}")

        if len(detected_names) == 0:
            raise RuntimeError("[Env] ❌ Unity 没有传回任何 Behavior！请检查 Unity 场景中是否有 Agent 且已启动 PlayMode。")

        # 模糊匹配：找到名称中包含 config.behavior_name 的项
        # 例如 config.behavior_name="UGV" 能匹配 "UGV?team=0"
        matched = [name for name in detected_names if config.behavior_name in name]

        if len(matched) > 0:
            # 匹配到了：使用第一个匹配项
            self.behavior_name = matched[0]
            print(f"[Env] 模糊匹配成功: config='{config.behavior_name}' → 实际='{self.behavior_name}'")
        else:
            # 完全匹配不到：降级使用第一个，并给出明确警告
            self.behavior_name = detected_names[0]
            print(f"[Env] ⚠️  未找到包含 '{config.behavior_name}' 的名称，已自动使用第一个: '{self.behavior_name}'")
            print(f"[Env] ⚠️  建议将 config.behavior_name 改为 '{self.behavior_name.split('?')[0]}'")

        # ------------------------------------------------------------------
        # 步骤 5: 获取并打印环境规格（状态/动作维度）
        # ------------------------------------------------------------------
        behavior_spec = self._env.behavior_specs[self.behavior_name]

        # 自动计算各传感器的真实���度，并逐一打印（帮助诊断维度异常）
        print(f"\n[Env] === 传感器维度详情 ===")
        total_state_dim = 0
        for i, obs_spec in enumerate(behavior_spec.observation_specs):
            dim = int(np.prod(obs_spec.shape))
            total_state_dim += dim
            # 根据 shape 判断传感器类型
            if len(obs_spec.shape) == 1:
                sensor_type = "向量传感器 (Vector / RayPerception)"
            elif len(obs_spec.shape) == 3:
                sensor_type = f"图像传感器 (Camera) {obs_spec.shape[1]}×{obs_spec.shape[2]}×{obs_spec.shape[0]}"
            else:
                sensor_type = f"未知类型 shape={obs_spec.shape}"
            print(f"  传感器 [{i}]: shape={obs_spec.shape}, 展平维度={dim}  ← {sensor_type}")
        print(f"  ─────────────────────────────────────")
        print(f"  总状态维度 (所有传感器展平后拼接): {total_state_dim}")
        print(f"  Config 中设置的 state_dim: {config.state_dim}")

        if total_state_dim != config.state_dim:
            print(f"\n[Env] ⚠️  维度不符！实际={total_state_dim}, config={config.state_dim}")
            print(f"[Env] ⚠️  若有图像传感器维度 (如 H×W×C)，说明场景中挂载了摄像头。")
            print(f"[Env] ⚠️  如暂时只想用向量观测训练，请在 Unity 中关闭/移除 Camera Sensor 组件。")
            print(f"[Env] ⚠️  已自动使用实际维度 {total_state_dim} 继续运行。\n")

        self.real_state_dim = total_state_dim
        self.state_dim = total_state_dim

        self.real_action_dim = behavior_spec.action_spec.continuous_size
        self.action_dim = self.real_action_dim

        print(f"[Env] 动作维度: {self.action_dim}  (连续动作: Steer + Motor)")
        print(f"[Env] ✅ 环境初始化完成! Behavior='{self.behavior_name}'\n")

        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """重置环境，返回初始观测 shape=(state_dim,)"""
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.behavior_name)
        obs = self._extract_obs(decision_steps)
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        发送动作，返回 (next_obs, reward, done, info)。

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
        return next_obs, reward, done, info

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
        """提取并展平拼接所有传感器的观测"""
        obs_list = decision_steps.obs
        flat_obs = np.concatenate(
            [obs[0].flatten() for obs in obs_list],
            axis=0
        ).astype(np.float32)
        return flat_obs

    def _extract_obs_from_terminal(self, terminal_steps) -> np.ndarray:
        """从 TerminalSteps 中提取观测（与 _extract_obs 对称）"""
        obs_list = terminal_steps.obs
        flat_obs = np.concatenate(
            [obs[0].flatten() for obs in obs_list],
            axis=0
        ).astype(np.float32)
        return flat_obs
#!/usr/bin/env python3
"""
灵巧手圆柱抓取 — PPO 强化学习控制脚本

输入:  GelSight 指尖触觉传感器的 2D 形变矩阵（每个指尖一张 64×64 深度图）
        + 关节位置/速度（N 指 × 2~3 关节）
输出: 各关节电机转矩
目标: 稳定抓住圆柱体，不掉落

依赖: torch, numpy, wandb (可选)
"""

import math, random, time, warnings, itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class PPOConfig:
    # 触觉传感器参数
    n_fingers: int = 3                  # 灵巧手手指数（如三指夹爪）
    tactile_size: int = 64              # GelSight 输出分辨率 64×64
    n_joints_per_finger: int = 3        # 每指关节数（含基座）
    n_joints: int = field(init=False)   # 总关节数
    latent_dim: int = 128               # Tactile encoder 输出维度

    # PPO 超参数
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 512              # 每轮收集步数
    n_episodes_per_eval: int = 5        # 评估局数

    # 训练
    max_timesteps: int = 60000
    eval_interval: int = 2              # 每 N 轮评估
    save_interval: int = 20             # 每 N 轮保存 checkpoint
    device: str = "auto"
    seed: int = 42

    def __post_init__(self):
        self.n_joints = self.n_fingers * self.n_joints_per_finger
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
#  GelSight 触觉编码器（CNN）
# ═══════════════════════════════════════════════════════════════

class TactileEncoder(nn.Module):
    """将 GelSight 2D 形变矩阵编码为低维特征向量。

    输入:  (B, C=1, H=64, W=64)   — 单通道深度/形变图
    输出:  (B, latent_dim)         — 触觉特征
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),   # 64→30
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # 30→14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # 14→6
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2), # 6→2
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                      # 2→1×1
            nn.Flatten(),                                 # → 128
        )
        self.fc = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        h = self.cnn(x)        # (B, 128)
        return self.fc(h)      # (B, latent_dim)


# ═══════════════════════════════════════════════════════════════
#  Actor-Critic 网络
# ═══════════════════════════════════════════════════════════════

class GraspPPONetwork(nn.Module):
    """PPO Actor-Critic: 触觉 + 关节状态 → 转矩 + 价值。

    观测空间:
      - tactile_imgs: (n_fingers, 1, 64, 64)    每个指尖一张深度图
      - joint_pos:    (n_joints,)                关节位置
      - joint_vel:    (n_joints,)                关节速度

    动作空间:  (n_joints,)   ∈ [-1, 1]  归一化转矩
    """

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg

        # 触觉编码器（每指共享权重 → 对所有指求和或拼接）
        self.tactile_encoder = TactileEncoder(latent_dim=cfg.latent_dim)

        # 关节状态编码
        state_dim = cfg.n_joints * 2  # pos + vel
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 融合层: 触觉特征 + 关节编码 → 联合特征
        fused_dim = cfg.latent_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Actor 头 — 输出高斯均值 + 对数标准差
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, cfg.n_joints),
            nn.Tanh(),                   # 输出归一化到 [-1, 1]
        )
        self.log_std = nn.Parameter(torch.full((cfg.n_joints,), -1.0))

        # Critic 头
        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, tactile_imgs: torch.Tensor,
                joint_pos: torch.Tensor,
                joint_vel: torch.Tensor) -> tuple:
        """返回 (action_mean, log_std, value)."""
        # 逐指编码触觉图像
        B = tactile_imgs.shape[0]
        # tactile_imgs: (B, n_fingers, 1, H, W)
        # 将手指维度与 batch 合并
        imgs = tactile_imgs.view(-1, 1, self.cfg.tactile_size, self.cfg.tactile_size)
        tactile_feats = self.tactile_encoder(imgs)  # (B*n_fingers, latent_dim)
        # 对全手指求和 → (B, latent_dim)
        tactile_feats = tactile_feats.view(B, self.cfg.n_fingers, -1).sum(dim=1)

        # 关节编码
        state_feats = self.state_encoder(
            torch.cat([joint_pos, joint_vel], dim=-1)
        )

        # 融合
        fused = self.fusion(torch.cat([tactile_feats, state_feats], dim=-1))

        # Actor
        mean = self.actor_mean(fused)         # (B, n_joints)
        log_std = self.log_std.expand_as(mean)  # (B, n_joints)

        # Critic
        value = self.critic(fused).squeeze(-1)   # (B,)

        return mean, log_std, value

    def get_action_and_value(self, tactile_imgs: torch.Tensor,
                             joint_pos: torch.Tensor,
                             joint_vel: torch.Tensor,
                             action: Optional[torch.Tensor] = None) -> dict:
        """动作采样或计算给定动作的 log_prob / entropy / value."""
        mean, log_std, value = self(tactile_imgs, joint_pos, joint_vel)
        std = log_std.exp()
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return {
            "action":      action,
            "log_prob":    log_prob,
            "entropy":     entropy,
            "value":       value,
            "mean":        mean,
            "std":         std,
        }


# ═══════════════════════════════════════════════════════════════
#  简化圆柱抓取模拟环境
# ═══════════════════════════════════════════════════════════════

class CylinderGraspEnv:
    """2D 圆柱抓取模拟环境（简化物理）。

    状态:
      - 圆柱 x 方向偏移（偏离掌心中心的位移）
      - 圆柱 z 方向偏移（重力方向下滑距离）
      - 各指法向力 N_i（基于触觉深度推导）
      - 各指尖与圆柱的接触角 θ_i

    观测:
      - tactile_imgs: (3, 1, 64, 64)  模拟的 GelSight 形变图
      - joint_pos:    (9,)  各关节位置（归一化）
      - joint_vel:    (9,)  各关节速度（归一化）

    动作:
      - joint_torque: (9,)  ∈ [-1, 1]  归一化转矩命令

    奖励:
      - 圆柱不滑落（z 偏移 < 阈值）
      - 法向力在合适范围内
      - 惩罚过大的力矩 / 急动
    """

    def __init__(self, cfg: PPOConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # 圆柱物理
        self.cylinder_mass = 0.1           # kg（轻，让学握力不那么难）
        self.cylinder_radius = 0.04        # m
        self.gravity = 9.81
        self.friction_coeff = 1.0          # 静摩擦系数
        self.target_normal_force = 2.0     # N — 理想法向力

        # 手指几何（简化 3 指 120° 分布）
        self.finger_angles = np.array([-120, 0, 120]) * math.pi / 180
        self.finger_stiffness = 200.0      # N/m — 接触刚度
        self.max_torque = 0.5              # N·m — 电机最大转矩

        # 状态
        self.reset()

    def reset(self):
        """重置环境，返回观测。"""
        # 圆柱状态
        self.x_offset = self.rng.normal(0, 0.001)  # 微小初始对不准
        self.z_offset = 0.0
        self.z_velocity = 0.0

        # 关节 — 手指张开（需要 PPO 学握紧）
        self.joint_pos = np.full(self.cfg.n_joints, 0.0)
        self.joint_vel = np.zeros(self.cfg.n_joints)
        self.prev_action = np.zeros(self.cfg.n_joints)

        # 接触力（基于初始位置）
        grip = np.mean(self.joint_pos[:self.cfg.n_fingers])
        self.contact_forces = self._compute_contact_forces(grip)

        # 模拟触觉图像缓存
        self._tactile_imgs = np.zeros(
            (self.cfg.n_fingers, 1, self.cfg.tactile_size, self.cfg.tactile_size),
            dtype=np.float32,
        )
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple:
        """执行一步物理模拟。

        Returns:
            obs, reward, done, info
        """
        # ── 1. 更新关节 ──
        torque = np.clip(action, -1, 1) * self.max_torque
        # 简单二阶动力学: 位置 ← 位置 + 速度; 速度 ← 速度 + (转矩 - 阻尼)/惯量
        damping = 0.5
        inertia = 0.02
        accel = (torque - damping * self.joint_vel) / inertia
        self.joint_vel += accel * 0.01     # dt = 10ms
        self.joint_pos += self.joint_vel * 0.01
        self.joint_pos = np.clip(self.joint_pos, -1, 1)
        self.joint_vel *= 0.98             # 速度衰减

        # ── 2. 计算接触力 ──
        grip_closure = np.mean(self.joint_pos[:self.cfg.n_fingers])
        self.contact_forces = self._compute_contact_forces(grip_closure)

        # ── 3. 更新圆柱动力学 ──
        total_normal = np.sum(self.contact_forces)
        # 摩擦力 = μ × 法向力（与滑移方向相反）
        friction_force = self.friction_coeff * total_normal

        # 重力趋势: 下滑力 = mg - 摩擦力 (近似)
        sliding_force = self.cylinder_mass * self.gravity - friction_force
        if sliding_force > 0:
            accel_z = sliding_force / self.cylinder_mass
            self.z_velocity += accel_z * 0.01
        else:
            # 摩擦力足以支撑 — 减速
            self.z_velocity *= 0.9
        self.z_offset += self.z_velocity * 0.01
        self.z_offset = max(0, self.z_offset)

        # 水平扰动（模拟外界干扰）
        disturbance = self.rng.normal(0, 0.002)
        self.x_offset += disturbance
        self.x_offset = np.clip(self.x_offset, -0.02, 0.02)

        # ── 4. 生成 GelSight 触觉图像 ──
        self._render_tactile()

        # ── 5. 计算奖励 ──
        reward, info = self._compute_reward(action, torque)

        # ── 6. 终止判断 ──
        done = self.z_offset > 0.05        # 滑落了 5cm

        obs = self._get_obs()
        return obs, reward, done, info

    # ── 私有方法 ──

    def _compute_contact_forces(self, grip_closure: float) -> np.ndarray:
        """基于手指闭合程度计算每指法向力。

        目标: grip=0.0 → 几乎滑落（0.3N/指，总0.9N < mg=0.98N）
              grip=0.3 → 刚好抓紧（~1.0N/指，总3.0N > 0.98N）
              grip=0.5 → 牢固抓紧（~1.9N/指）
              grip=1.0 → 强力抓紧（~4.5N/指）
        """
        grip = max(-1.0, min(1.0, grip_closure))
        # 非线性: 0→0.3, 0.3→1.0, 0.5→1.9, 1.0→4.5
        base = 0.3 + 2.0 * max(0, grip) + 2.0 * max(0, grip) ** 2
        # x_offset 导致的不均匀性
        per_finger = np.full(self.cfg.n_fingers, base, dtype=np.float32)
        for i in range(self.cfg.n_fingers):
            angle = self.finger_angles[i]
            mismatch = abs(self.x_offset * math.cos(angle) * 50)
            per_finger[i] = max(0.2, per_finger[i] * (1 - 0.3 * mismatch))
        return per_finger

    def _render_tactile(self):
        """基于接触仿真实时渲染 GelSight 形变图。

        每个指尖生成一个 64×64 单通道深度图:
          - 接触区域 → 高斯状凸起（深度值与法向力成正比）
          - 接触中心根据 x_offset 偏移
          - 噪声模拟传感器噪声
        """
        imgs = np.zeros((self.cfg.n_fingers, 1,
                         self.cfg.tactile_size, self.cfg.tactile_size),
                        dtype=np.float32)
        for i in range(self.cfg.n_fingers):
            force = self.contact_forces[i]
            amp = min(force / 10.0, 1.0)   # 归一化幅度

            # 接触中心（在传感器上的像素坐标）
            cx = self.cfg.tactile_size // 2
            cy = self.cfg.tactile_size // 2
            # 根据水平偏移偏移接触中心
            shift = int(self.x_offset * 500)
            cx += int(shift * math.cos(self.finger_angles[i]))
            cy += int(shift * math.sin(self.finger_angles[i]))
            cx = np.clip(cx, 5, self.cfg.tactile_size - 5)
            cy = np.clip(cy, 5, self.cfg.tactile_size - 5)

            # 2D 高斯
            sigma = 8.0 + 4.0 * (1 - amp)  # 力越大接触区域越大
            Y, X = np.ogrid[:self.cfg.tactile_size, :self.cfg.tactile_size]
            dist2 = (X - cx)**2 + (Y - cy)**2
            gauss = amp * np.exp(-dist2 / (2 * sigma**2))

            # 添加传感器噪声
            noise = self.rng.normal(0, 0.02, gauss.shape)
            imgs[i, 0] = np.clip(gauss + noise, 0, 1)

        self._tactile_imgs = imgs

    def _compute_reward(self, action, torque):
        """奖励函数设计。"""
        reward = 0.0
        info = {}

        # (a) 主要目标: 不掉落
        stay_alive = 1.0 - 10.0 * self.z_offset
        reward += stay_alive

        # (b) 法向力奖惩 — 接近目标力
        force_error = np.mean(np.abs(self.contact_forces
                                     - self.target_normal_force))
        force_penalty = -0.02 * force_error
        reward += force_penalty

        # (c) 力均匀性 — 三指力需均匀
        force_std = np.std(self.contact_forces)
        uniformity_penalty = -0.05 * force_std
        reward += uniformity_penalty

        # (d) 力矩消耗惩罚
        torque_penalty = -0.01 * np.mean(np.abs(torque))
        reward += torque_penalty

        # (e) 急动惩罚（动作变化率）
        jerk = np.mean(np.abs(action - self.prev_action))
        jerk_penalty = -0.005 * jerk
        reward += jerk_penalty
        self.prev_action = action.copy()

        info["reward_alive"] = stay_alive
        info["reward_force"] = force_penalty
        info["reward_uniform"] = uniformity_penalty
        info["contact_forces"] = self.contact_forces.copy()
        info["z_offset"] = self.z_offset
        return reward, info

    def _get_obs(self) -> dict:
        return {
            "tactile_imgs": self._tactile_imgs.copy(),
            "joint_pos":    self.joint_pos.copy().astype(np.float32),
            "joint_vel":    self.joint_vel.copy().astype(np.float32),
        }


# ═══════════════════════════════════════════════════════════════
#  经验回放缓冲区 (GAE)
# ═══════════════════════════════════════════════════════════════

@dataclass
class RolloutBuffer:
    """存储 N 步经验，用于 PPO 更新。"""
    cfg: PPOConfig

    def __post_init__(self):
        self.capacity = self.cfg.buffer_size
        self.clear()

    def clear(self):
        size = self.capacity
        self.tactile_imgs = np.zeros(
            (size, self.cfg.n_fingers, 1, self.cfg.tactile_size, self.cfg.tactile_size),
            dtype=np.float32)
        self.joint_pos    = np.zeros((size, self.cfg.n_joints), dtype=np.float32)
        self.joint_vel    = np.zeros((size, self.cfg.n_joints), dtype=np.float32)
        self.actions      = np.zeros((size, self.cfg.n_joints), dtype=np.float32)
        self.log_probs    = np.zeros(size, dtype=np.float32)
        self.rewards      = np.zeros(size, dtype=np.float32)
        self.dones        = np.zeros(size, dtype=bool)
        self.values       = np.zeros(size, dtype=np.float32)
        self.returns      = np.zeros(size, dtype=np.float32)
        self.advantages   = np.zeros(size, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def store(self, obs, action, log_prob, reward, done, value):
        i = self.idx % self.capacity
        self.tactile_imgs[i] = obs["tactile_imgs"]
        self.joint_pos[i]    = obs["joint_pos"]
        self.joint_vel[i]    = obs["joint_vel"]
        self.actions[i]      = action
        self.log_probs[i]    = log_prob
        self.rewards[i]      = reward
        self.dones[i]        = done
        self.values[i]       = value
        self.idx += 1
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """在 episode 结束后用 GAE(lambda) 计算 advantages。"""
        n = min(self.size, self.capacity)
        adv = np.zeros(n, dtype=np.float32)
        next_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                delta = (self.rewards[t]
                         + gamma * last_value * next_non_terminal
                         - self.values[t])
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                delta = (self.rewards[t]
                         + gamma * self.values[t + 1] * next_non_terminal
                         - self.values[t])
            next_gae = delta + gamma * lam * next_non_terminal * next_gae
            adv[t] = next_gae
        self.advantages[:n] = adv
        self.returns[:n] = adv + self.values[:n]

    def get_batches(self, batch_size: int):
        """随机打乱并返回 mini-batch 迭代器。"""
        n = min(self.size, self.capacity)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                k: torch.from_numpy(getattr(self, k)[batch_idx])
                for k in ["tactile_imgs", "joint_pos", "joint_vel",
                          "actions", "log_probs", "returns", "advantages"]
            }

    def is_full(self) -> bool:
        return self.size >= self.capacity


# ═══════════════════════════════════════════════════════════════
#  PPO 训练器
# ═══════════════════════════════════════════════════════════════

class PPOTrainer:
    """单智能体 PPO 训练器。"""

    def __init__(self, cfg: PPOConfig, env: CylinderGraspEnv):
        self.cfg = cfg
        self.env = env
        self.device = torch.device(cfg.device)

        self.net = GraspPPONetwork(cfg).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = RolloutBuffer(cfg)

        self.timestep = 0
        self.episode = 0
        self.best_reward = -float("inf")

    # ── 观测 → Tensor ──

    def _to_tensor(self, obs: dict) -> dict:
        return {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)
            for k, v in obs.items()
        }

    # ── 数据收集 ──

    def collect_rollout(self) -> dict:
        """收集完整 buffer 的经验用于 PPO 更新。"""
        self.buffer.clear()
        ep_returns = []
        ep_sum = 0.0
        done = True

        while self.buffer.size < self.cfg.buffer_size:
            if done:
                obs = self.env.reset()
                ep_sum = 0.0

            obs_t = self._to_tensor(obs)

            with torch.no_grad():
                out = self.net.get_action_and_value(
                    obs_t["tactile_imgs"],
                    obs_t["joint_pos"],
                    obs_t["joint_vel"],
                )
            action_np = out["action"].cpu().numpy().squeeze(0)
            log_prob_np = out["log_prob"].item()
            value_np = out["value"].item()
            next_obs, reward, done, info = self.env.step(action_np)

            self.buffer.store(obs, action_np, log_prob_np, reward, done, value_np)
            ep_sum += reward
            self.timestep += 1

            if done:
                ep_returns.append(ep_sum)

            obs = next_obs

        last_value = 0.0 if done else (
            self.net.get_action_and_value(
                torch.from_numpy(obs["tactile_imgs"]).unsqueeze(0).to(self.device),
                torch.from_numpy(obs["joint_pos"]).unsqueeze(0).to(self.device),
                torch.from_numpy(obs["joint_vel"]).unsqueeze(0).to(self.device),
            )["value"].item()
        )
        self.buffer.compute_gae(last_value, self.cfg.gamma, self.cfg.gae_lambda)

        ep_ret = np.mean(ep_returns) if ep_returns else 0.0
        return {"episode_return": ep_ret,
                "episode_length": self.buffer.size}

    # ── PPO 更新 ──

    def update(self) -> dict:
        """用当前 buffer 执行 PPO 更新。"""
        losses = []

        for _ in range(self.cfg.ppo_epochs):
            for batch in self.buffer.get_batches(self.cfg.batch_size):
                # 移动到设备
                b = {k: v.to(self.device) for k, v in batch.items()}

                out = self.net.get_action_and_value(
                    b["tactile_imgs"],
                    b["joint_pos"],
                    b["joint_vel"],
                    b["actions"],
                )

                # 重要性采样比率
                ratio = torch.exp(out["log_prob"] - b["log_probs"])

                # PPO-clip 策略损失
                adv = b["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                obj = ratio * adv
                obj_clipped = torch.clamp(ratio, 1 - self.cfg.clip_epsilon,
                                          1 + self.cfg.clip_epsilon) * adv
                policy_loss = -torch.min(obj, obj_clipped).mean()

                # 价值损失
                v_pred = out["value"]
                v_target = b["returns"]
                value_loss = F.mse_loss(v_pred, v_target)

                # 熵奖励（鼓励探索）
                entropy_loss = -out["entropy"].mean()

                # 总损失
                loss = (policy_loss
                        + self.cfg.vf_coef * value_loss
                        + self.cfg.ent_coef * entropy_loss)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         self.cfg.max_grad_norm)
                self.optim.step()

                losses.append({
                    "loss":         loss.item(),
                    "policy_loss":  policy_loss.item(),
                    "value_loss":   value_loss.item(),
                    "entropy":      -entropy_loss.item(),
                    "approx_kl":    (ratio - 1 - torch.log(ratio)).mean().item(),
                })

        self.buffer.clear()
        avg = {k: np.mean([l[k] for l in losses]) for k in losses[0]}
        return avg

    # ── 评估 ──

    def evaluate(self) -> dict:
        """运行 N 局评估（无探索噪声）。"""
        returns = []
        lengths = []
        forces = []
        z_offsets = []

        for _ in range(self.cfg.n_episodes_per_eval):
            obs = self.env.reset()
            done = False
            ep_ret = 0
            ep_len = 0
            ep_forces = []

            while not done and ep_len < 500:  # 最长 500 步
                obs_t = self._to_tensor(obs)
                with torch.no_grad():
                    out = self.net.get_action_and_value(
                        obs_t["tactile_imgs"],
                        obs_t["joint_pos"],
                        obs_t["joint_vel"],
                    )
                action_np = out["mean"].cpu().numpy().squeeze(0)  # 无噪声
                obs, reward, done, info = self.env.step(action_np)
                ep_ret += reward
                ep_len += 1
                ep_forces.append(info["contact_forces"].copy())

            returns.append(ep_ret)
            lengths.append(ep_len)
            if ep_forces:
                forces.append(np.mean(ep_forces, axis=0))
            z_offsets.append(info.get("z_offset", 0))

        return {
            "return_mean":    np.mean(returns),
            "return_std":     np.std(returns),
            "length_mean":    np.mean(lengths),
            "success_rate":   np.mean([l >= 499 for l in lengths]),
            "force_mean":     np.mean(forces, axis=0).tolist() if forces else [],
            "z_offset_mean":  np.mean(z_offsets),
        }

    # ── 保存/加载 ──

    def save(self, path: str):
        torch.save({
            "net_state_dict": self.net.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "timestep": self.timestep,
            "episode": self.episode,
            "cfg": self.cfg,
        }, path)

    def load(self, path: str):
        cp = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(cp["net_state_dict"])
        self.optim.load_state_dict(cp["optim_state_dict"])
        self.timestep = cp["timestep"]
        self.episode = cp["episode"]

    # ── 主训练循环 ──

    def train(self, log_fn=None):
        """主训练循环。"""
        print(f"开始训练 — 设备: {self.device}")
        print(f"观测: {self.cfg.n_fingers}指 × {self.cfg.tactile_size}² GelSight "
              f"+ {self.cfg.n_joints} 关节")
        print(f"动作: {self.cfg.n_joints} 维转矩  "
              f"buffer: {self.cfg.buffer_size}  "
              f"max_steps: {self.cfg.max_timesteps}")
        print("-" * 60)

        while self.timestep < self.cfg.max_timesteps:
            # 收集
            rollout_info = self.collect_rollout()
            self.episode += 1

            # 更新
            loss_info = self.update()

            # 日志
            ep_ret = rollout_info["episode_return"]
            ep_len = rollout_info["episode_length"]

            if self.episode % self.cfg.eval_interval == 0:
                eval_info = self.evaluate()
                msg = (f"[{self.timestep:7d} | ep {self.episode:4d}]  "
                       f"ret={ep_ret:6.1f}  len={ep_len:3d}  "
                       f"eval_ret={eval_info['return_mean']:6.1f}  "
                       f"success={eval_info['success_rate']:.1%}  "
                       f"z={eval_info['z_offset_mean']:.4f}")
                print(msg)

                if eval_info["return_mean"] > self.best_reward:
                    self.best_reward = eval_info["return_mean"]
                    self.save("grasp_ppo_best.pt")

            if self.episode % self.cfg.save_interval == 0:
                self.save(f"grasp_ppo_ep{self.episode}.pt")

            if log_fn:
                log_fn(self.episode, ep_ret, loss_info)

        print(f"\n训练完成！最佳评估奖励: {self.best_reward:.1f}")
        self.save("grasp_ppo_final.pt")
        return self.net


# ═══════════════════════════════════════════════════════════════
#  WandB 日志（可选）
# ═══════════════════════════════════════════════════════════════

def wandb_log_fn(episode, ep_ret, loss_info):
    try:
        import wandb
        wandb.log({
            "episode":          episode,
            "episode_return":   ep_ret,
            **{f"loss/{k}": v for k, v in loss_info.items()},
        })
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════
#  Demo / 评估模式
# ═══════════════════════════════════════════════════════════════

def demo_trained_policy(cfg: PPOConfig, checkpoint_path: str,
                        max_steps: int = 500):
    """加载训练好的策略并演示抓取（打印时序信息）。"""
    env = CylinderGraspEnv(cfg)
    net = GraspPPONetwork(cfg).to(cfg.device)
    cp = torch.load(checkpoint_path, map_location=cfg.device)
    net.load_state_dict(cp["net_state_dict"])
    net.eval()

    obs = env.reset()
    total_reward = 0.0

    print(f"\n演示 — 已加载检查点 (ep={cp['episode']}, step={cp['timestep']})")
    print(f"{'步':>4}  {'力[0]':>6} {'力[1]':>6} {'力[2]':>6}  "
          f"{'z偏移':>6}  {'奖励':>6}  {'动作(均值)':>20}")

    for step in range(max_steps):
        obs_t = {
            k: torch.from_numpy(v).unsqueeze(0).to(cfg.device)
            for k, v in obs.items()
        }
        with torch.no_grad():
            out = net.get_action_and_value(
                obs_t["tactile_imgs"],
                obs_t["joint_pos"],
                obs_t["joint_vel"],
            )
        action_np = out["mean"].cpu().numpy().squeeze(0)
        obs, reward, done, info = env.step(action_np)
        total_reward += reward

        if step % 50 == 0:
            f = info["contact_forces"]
            a_str = " ".join(f"{a:6.3f}" for a in action_np[:3])
            print(f"{step:4d}  {f[0]:6.2f} {f[1]:6.2f} {f[2]:6.2f}  "
                  f"{info['z_offset']:6.4f}  {reward:6.2f}  {a_str}")

        if done:
            print(f"\n💥 圆柱掉落！步数: {step}")
            break

    print(f"\n总奖励: {total_reward:.1f}")
    return total_reward


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GelSight 灵巧手 PPO 抓取控制")
    parser.add_argument("--mode", choices=["train", "demo", "bench"],
                        default="train", help="运行模式")
    parser.add_argument("--checkpoint", default="grasp_ppo_best.pt",
                        help="加载的检查点路径（demo/bench）")
    parser.add_argument("--n-fingers", type=int, default=3,
                        help="手指数量")
    parser.add_argument("--tactile-size", type=int, default=64,
                        help="GelSight 分辨率")
    parser.add_argument("--max-steps", type=int, default=500_000,
                        help="训练总步数 (train) 或最大步数 (demo)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--wandb", action="store_true", help="启用 WandB 日志")
    args = parser.parse_args()

    cfg = PPOConfig(
        n_fingers=args.n_fingers,
        tactile_size=args.tactile_size,
        max_timesteps=args.max_steps,
        seed=args.seed,
    )

    # 随机种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if args.mode == "train":
        if args.wandb:
            try:
                import wandb
                wandb.init(project="grasp-ppo", config=vars(cfg))
            except ImportError:
                print("⚠️  WandB 未安装，跳过日志")
                args.wandb = False

        env = CylinderGraspEnv(cfg)
        trainer = PPOTrainer(cfg, env)
        log_fn = wandb_log_fn if args.wandb else None
        trainer.train(log_fn=log_fn)

    elif args.mode == "demo":
        demo_trained_policy(cfg, args.checkpoint,
                            max_steps=args.max_steps)

    elif args.mode == "bench":
        """快速基准: 无学习, 仅用随机动作测试环境."""
        env = CylinderGraspEnv(cfg)
        returns = []
        for ep in range(20):
            obs = env.reset()
            done = False
            ep_ret = 0
            while not done:
                action = np.random.uniform(-1, 1, cfg.n_joints).astype(np.float32)
                obs, reward, done, info = env.step(action)
                ep_ret += reward
            returns.append(ep_ret)
        print(f"随机策略基准 — 20 局平均奖励: {np.mean(returns):.1f} "
              f"± {np.std(returns):.1f}")

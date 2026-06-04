#!/Users/lix/miniconda3/bin/python3
"""
灵巧手抓取可视化 — 实时显示 GelSight 触觉图 + 抓取状态

用法:  ./visualize_grasp.py                          # 随机策略
       ./visualize_grasp.py --checkpoint grasp_ppo_best.pt  # 训练好的策略
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

# CJK 字体配置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from train_grasp_ppo import PPOConfig, CylinderGraspEnv, GraspPPONetwork
import torch


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="训练好的检查点路径")
    p.add_argument("--max-steps", type=int, default=300, help="最大步数")
    p.add_argument("--fps", type=int, default=20, help="帧率")
    p.add_argument("--save", default=None, help="保存为 GIF/MP4 文件路径")
    return p.parse_args()


def make_grasp_diagram(ax, cfg, env):
    """绘制抓取俯视图：圆柱 + 三指位置（关节驱动）+ 力向量。"""
    ax.clear()
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(-0.08, 0.08)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_title("抓取俯视图", color="white", fontsize=11)

    # 圆柱
    cx = 0
    cy = 0
    cyl = Circle((cx, cy), env.cylinder_radius, facecolor="#4a90d9",
                 edgecolor="#7bb3e0", linewidth=2, alpha=0.8)
    ax.add_patch(cyl)
    # 目标圆柱（半透明参考）
    target = Circle((0, 0), env.cylinder_radius, facecolor="none",
                    edgecolor="#4a90d9", linewidth=1, linestyle="--", alpha=0.3)
    ax.add_patch(target)

    # 手指 + 力 — 位置由关节决定
    colors = ["#ff6b6b", "#51cf66", "#ffd43b"]
    labels = ["拇指", "食指", "中指"]
    for i in range(cfg.n_fingers):
        angle = env.finger_angles[i]
        # 关节 grip 值 → 手指径向位置
        grip = float(env.joint_pos[i])         # [-1, 1]
        grip_norm = max(0, min(1, (grip + 1) / 2))  # 归一化到 [0, 1]
        # grip=0(开) → 距离远, grip=1(闭) → 贴住圆柱
        max_dist = env.cylinder_radius + 0.025
        min_dist = env.cylinder_radius + 0.001
        base_dist = max_dist - (max_dist - min_dist) * grip_norm
        fx = base_dist * np.cos(angle)
        fy = base_dist * np.sin(angle)

        # 手指圆（大小也随 grip 变化）
        finger_radius = 0.004 + 0.004 * grip_norm
        finger = Circle((fx, fy), finger_radius, facecolor=colors[i],
                        edgecolor="white", linewidth=1.5)
        ax.add_patch(finger)
        ax.text(fx * 1.5, fy * 1.5, labels[i], color=colors[i],
                fontsize=8, ha="center", va="center")

        # 法向力箭头（指向圆柱中心）
        force = float(env.contact_forces[i])
        # 箭头长度与力成正比，最小可见
        arrow_len = 0.002 + 0.015 * min(force / 5.0, 1.0)
        ax.arrow(fx, fy, -arrow_len * np.cos(angle), -arrow_len * np.sin(angle),
                 head_width=0.004, head_length=0.004,
                 color=colors[i], alpha=min(0.3 + 0.7 * force / 5.0, 1.0),
                 linewidth=1.5)

        # 标注力大小
        label_offset = 0.008
        lx = fx - (arrow_len + 0.006) * np.cos(angle)
        ly = fy - (arrow_len + 0.006) * np.sin(angle)
        ax.text(lx, ly, f"{force:.2f}N", color=colors[i],
                fontsize=7, ha="center", va="center", alpha=0.9)

    grip_mean = float(np.mean(env.joint_pos[:cfg.n_fingers]))
    ax.text(0, -0.070, f"z偏移: {env.z_offset*1000:.1f}mm | "
            f"握力: {grip_mean:.2f} | "
            f"接触力: {float(np.mean(env.contact_forces)):.2f}N/指",
            color="white", fontsize=9, ha="center")


def make_tactile_view(ax, env, finger_idx, title):
    """绘制单个指尖的 GelSight 形变热图。"""
    img = env._tactile_imgs[finger_idx, 0]  # (64, 64)
    ax.clear()
    ax.imshow(img, cmap="inferno", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, color="white", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    # 边框
    for spine in ax.spines.values():
        spine.set_color("#444")


def main():
    args = parse_args()
    cfg = PPOConfig(max_timesteps=args.max_steps)

    # 初始化
    env = CylinderGraspEnv(cfg)
    net = GraspPPONetwork(cfg).to(cfg.device)
    net.eval()

    # 加载检查点
    if args.checkpoint and os.path.exists(args.checkpoint):
        cp = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
        net.load_state_dict(cp["net_state_dict"])
        print(f"已加载检查点 (ep={cp.get('episode','?')}, step={cp.get('timestep','?')})")

    # ── 构建画布 ──
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#16213e")
    fig.suptitle("GelSight 灵巧手抓取模拟 — PPO 控制",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    # 布局: 左上=俯视图  右三=触觉图  底部=时序曲线
    gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.35,
                           left=0.05, right=0.98, top=0.90, bottom=0.08)

    ax_top = fig.add_subplot(gs[0, 0:2])  # 俯视图
    ax_tactile = [
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
    ]
    ax_force = fig.add_subplot(gs[1, 0:3])  # 力时序
    ax_height = fig.add_subplot(gs[1, 3:5])  # z偏移时序

    # 数据缓存
    force_history = [[] for _ in range(cfg.n_fingers)]
    z_history = []
    step_history = []
    step = 0

    def init_anim():
        """初始化动画（返回所有要刷新的 artist）。"""
        return []

    def update_anim(frame):
        nonlocal step

        # ── 一步模拟 ──
        if step == 0:
            env.reset()

        # 准备观测 Tensor
        obs = {
            "tactile_imgs": torch.from_numpy(env._tactile_imgs).unsqueeze(0).float().to(cfg.device),
            "joint_pos": torch.from_numpy(env.joint_pos).unsqueeze(0).float().to(cfg.device),
            "joint_vel": torch.from_numpy(env.joint_vel).unsqueeze(0).float().to(cfg.device),
        }
        with torch.no_grad():
            out = net.get_action_and_value(
                obs["tactile_imgs"], obs["joint_pos"], obs["joint_vel"])

        action = out["mean"].cpu().numpy().squeeze(0)
        _, reward, done, info = env.step(action)

        # ── 更新时序数据 ──
        for i in range(cfg.n_fingers):
            force_history[i].append(info["contact_forces"][i])
        z_history.append(info["z_offset"] * 1000)  # mm
        step_history.append(step)

        # 历史窗口（最近 200 步）
        window = max(1, len(step_history) - 200)
        s_hist = step_history[window:]
        z_hist = z_history[window:]
        f_hist = [f[window:] for f in force_history]

        # ── 俯视图 ──
        make_grasp_diagram(ax_top, cfg, env)

        # ── 触觉图 ──
        labels = ["拇指 GelSight", "食指 GelSight", "中指 GelSight"]
        for i in range(cfg.n_fingers):
            make_tactile_view(ax_tactile[i], env, i, labels[i])

        # ── 力时序 ──
        ax_force.clear()
        colors = ["#ff6b6b", "#51cf66", "#ffd43b"]
        labels_f = ["拇指", "食指", "中指"]
        for i in range(cfg.n_fingers):
            if len(s_hist) > 1:
                ax_force.plot(s_hist, f_hist[i], color=colors[i],
                              label=labels_f[i], linewidth=1.2)
        ax_force.axhline(y=env.target_normal_force, color="white",
                         linestyle="--", linewidth=0.8, alpha=0.4,
                         label=f"目标 {env.target_normal_force}N")
        ax_force.set_facecolor("#1a1a2e")
        ax_force.set_title("指尖法向力 (N)", color="white", fontsize=10)
        ax_force.set_xlabel("步", color="#aaa", fontsize=8)
        ax_force.set_ylabel("力 (N)", color="#aaa", fontsize=8)
        ax_force.legend(loc="upper right", fontsize=7,
                        labelcolor="white", facecolor="#222", edgecolor="#444")
        ax_force.tick_params(colors="#aaa", labelsize=7)
        ax_force.set_ylim(0, max(8, max(max(f) for f in f_hist) + 1) if f_hist[0] else 8)
        for spine in ax_force.spines.values():
            spine.set_color("#444")

        # ── z偏移时序 ──
        ax_height.clear()
        if len(s_hist) > 1:
            ax_height.plot(s_hist, z_hist, color="#74b9ff", linewidth=1.5)
        ax_height.axhline(y=50, color="#ff6b6b", linestyle="--",
                          linewidth=0.8, alpha=0.5, label="掉落阈值")
        ax_height.set_facecolor("#1a1a2e")
        ax_height.set_title("圆柱下落 z偏移 (mm)", color="white", fontsize=10)
        ax_height.set_xlabel("步", color="#aaa", fontsize=8)
        ax_height.set_ylabel("z (mm)", color="#aaa", fontsize=8)
        ax_height.legend(loc="upper left", fontsize=7,
                         labelcolor="white", facecolor="#222", edgecolor="#444")
        ax_height.tick_params(colors="#aaa", labelsize=7)
        ax_height.set_ylim(0, 60)
        for spine in ax_height.spines.values():
            spine.set_color("#444")

        # ── 步数信息 ──
        status = f"步 {step}  奖励 {reward:.1f}  "
        status += "✓ 抓住" if not done else "✗ 掉落"
        ax_top.set_xlabel(status, color="white" if not done else "#ff6b6b",
                          fontsize=10, fontweight="bold")

        step += 1
        if step >= args.max_steps or done:
            step = 0
            for i in range(cfg.n_fingers):
                force_history[i].clear()
            z_history.clear()
            step_history.clear()

        return []

    # ── 显示窗口 ──
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # ── 主循环 ──
    print("可视化运行中... 关闭窗口退出")
    for _ in range(args.max_steps):
        update_anim(0)
        plt.pause(1.0 / args.fps)
        if not plt.fignum_exists(fig.number):
            break

    print("可视化结束")


if __name__ == "__main__":
    main()

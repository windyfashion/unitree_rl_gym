# G1 / unitree_rl_gym 学习笔记

> 基于 `unitree_rl_gym` 项目中 G1 环境、奖励设计与部署相关代码的讨论整理。  
> 主要文件：`legged_gym/envs/g1/g1_env.py`、`g1_config.py`、`legged_robot.py`、`deploy/deploy_mujoco/deploy_mujoco.py`

---

## 目录

1. [步态相位 `leg_phase`](#1-步态相位-leg_phase)
2. [左右腿控制：有无预设步态曲线](#2-左右腿控制有无预设步态曲线)
3. [张量索引：`feet_pos[:, :, 2]`](#3-张量索引feet_pos---2)
4. [接触判定：`contact` 与 `torch.norm`](#4-接触判定contact-与-torchnorm)
5. [奖励 `_reward_feet_swing_height`](#5-奖励-_reward_feet_swing_height)
6. [奖励 `_reward_contact`](#6-奖励-_reward_contact)
7. [奖励 `_reward_contact_no_vel` 与参考系](#7-奖励-_reward_contact_no_vel-与参考系)
8. [观测设计：普通观测 vs 特权观测](#8-观测设计普通观测-vs-特权观测)
9. [相位观测 `sin_phase` / `cos_phase` 的意义](#9-相位观测-sin_phase--cos_phase-的意义)
10. [LSTM 与循环策略网络](#10-lstm-与循环策略网络)
11. [概念速查表](#11-概念速查表)

---

## 1. 步态相位 `leg_phase`

### 1.1 是什么

`leg_phase` 表示**每条腿在当前步态周期中的归一化相位**，形状为 `(num_envs, 2)`：

- 第 0 列：左腿相位 `phase_left`
- 第 1 列：右腿相位 `phase_right`

取值范围 **[0, 1)**，0 表示周期起点，接近 1 时回到下一周期起点。

### 1.2 如何计算

在 `G1Robot._post_physics_step_callback` 中每仿真步更新：

```python
period = 0.8          # 一个完整步态周期 0.8 秒
offset = 0.5          # 左右腿相位差半个周期（交替步态）
self.phase = (self.episode_length_buf * self.dt) % period / period
self.phase_left = self.phase
self.phase_right = (self.phase + offset) % 1
self.leg_phase = torch.cat([self.phase_left.unsqueeze(1),
                            self.phase_right.unsqueeze(1)], dim=-1)
```

| 变量 | 含义 |
|------|------|
| `period` | 开环步态时钟周期（秒） |
| `self.phase` | 全局相位，由仿真时间驱动，与真实触地无关 |
| `offset = 0.5` | 右腿比左腿晚半个周期，模拟左右交替 |
| `leg_phase` | 拼成 per-leg 相位，主要用于奖励 |

### 1.3 用途

主要用于 `_reward_contact`：根据相位判断脚**应该**处于支撑相还是摆动相，再与实际接触力对比。

**注意**：`leg_phase` **不直接进入** `obs_buf`，策略看到的是全局 `sin/cos(phase)`（见第 9 节）。

---

## 2. 左右腿控制：有无预设步态曲线

### 2.1 结论

**没有**预先给定的关节角或足端轨迹曲线。控制链路为：

```
RL 策略 → action（12 维）
       → target_angle = action × action_scale + default_dof_pos
       → PD 控制 → 关节力矩
```

`default_dof_pos` 来自 `g1_config` 的 `default_joint_angles`，是**固定站立偏置**，不随时间变化。

### 2.2 `phase` / `leg_phase` 的角色

| 机制 | 作用 |
|------|------|
| 开环时钟 `phase` | 观测中的 sin/cos；奖励中的支撑/摆动期望 |
| **不是** | 播放关节样条、足端轨迹、CPG 输出 |

步态由策略在奖励塑形下**学出来**，而非跟踪预录曲线。

### 2.3 与「步态曲线控制」的对比

| 本仓库 | 典型步态曲线方案 |
|--------|------------------|
| 参考 = 常数默认姿态 + 策略增量 | \(q_{ref}(t)\) 或 \(x_{foot}(t)\) 随相位变化 |
| 相位用于奖励/观测提示 | 控制器显式跟踪曲线 |

---

## 3. 张量索引：`feet_pos[:, :, 2]`

### 3.1 语法说明

这是 **PyTorch / NumPy 多维张量索引**，不是 Python 语言独有特性。

`feet_pos` 形状约为 `(num_envs, 脚数量, 3)`，最后一维为 **x, y, z**。

```python
feet_pos[:, :, 2]
#  │    │   └── 第 3 个分量：z（高度）
#  │    └── 所有脚
#  └── 所有并行环境
```

| 写法 | 结果形状 | 含义 |
|------|----------|------|
| `:` | 保留该维全部 | 不缩减这一维 |
| 整数 `2` | 去掉该维 | 只取 z 坐标 |
| `[:, :, 2]` | `(N, 脚)` | 每只脚的竖直位置 |

### 3.2 在代码中的用途

```python
pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
```

只在**脚未着地**时惩罚高度与 0.08 m 的偏差（摆动相抬脚高度）。

---

## 4. 接触判定：`contact` 与 `torch.norm`

### 4.1 数据来源

```python
# legged_robot.py
self.contact_forces = ... .view(self.num_envs, -1, 3)
# 形状：(num_envs, 刚体数, 3)，最后一维为 Fx, Fy, Fz（牛顿）
```

`feet_indices`：脚底刚体（如 `ankle_roll`）在刚体维上的下标。

### 4.2 典型写法

```python
contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
```

| 步骤 | 形状 | 含义 |
|------|------|------|
| `[:, feet_indices, :3]` | `(N, 脚, 3)` | 每只脚的接触力向量 |
| `torch.norm(..., dim=2)` | `(N, 脚)` | 合力大小 \(\sqrt{F_x^2+F_y^2+F_z^2}\) |
| `> 1.` | `(N, 脚)` bool | 大于 1 N 视为着地 |

`1.` 是经验阈值，非物理常数。别处也有仅用竖直分量 `contact_forces[..., 2] > 1` 的写法。

### 4.3 `~contact` 与布尔运算

`~` 对 bool 张量按位取反：未着地 → True，用于掩码乘法。

---

## 5. 奖励 `_reward_feet_swing_height`

```python
contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
return torch.sum(pos_error, dim=1)
```

- **目的**：摆动相（未着地）时，鼓励脚抬到约 **0.08 m** 高度。
- **配置**：`feet_swing_height = -20.0`（负 scale → 惩罚高度误差）。

---

## 6. 奖励 `_reward_contact`

```python
for i in range(self.feet_num):
    is_stance = self.leg_phase[:, i] < 0.55   # 相位 < 0.55 → 期望支撑相
    contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    res += ~(contact ^ is_stance)            # 一致加分，不一致减分
```

### 6.1 逻辑

- `contact ^ is_stance`：异或，**不一致**为 True。
- `~(...)`：一致时为 True → 奖励增加。

即：**该撑的时候着地、该摆的时候离地**。

### 6.2 与相位观测的关系

- 使用 **`leg_phase`（环境内部）**，不依赖策略是否“看到”相位。
- 相位观测（sin/cos）是为了让策略**更容易学会**配合这项奖励，而非奖励生效的前提。

---

## 7. 奖励 `_reward_contact_no_vel` 与参考系

### 7.1 代码

```python
contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
penalize = torch.square(contact_feet_vel[:, :, :3])
return torch.sum(penalize, dim=(1, 2))
```

```python
# feet_vel 来源
self.feet_vel = self.feet_state[:, :, 7:10]  # 刚体状态 7:10
```

### 7.2 参考系（重要）

| 项目 | 说明 |
|------|------|
| **参考系** | **世界系 / 惯性系**（Isaac Gym 刚体线速度标准定义） |
| **速度对象** | 脚连杆（如 `ankle_roll`）原点的线速度，**非**接触点速度 |
| **意图（名义上）** | 抑制着地脚滑动、拖脚、蹭地 |
| **配置** | `contact_no_vel = -0.2`，权重相对较小 |

### 7.3 理论辨析：不同参考系下的“脚速度”

| 量 | 无打滑、平地行走时 |
|----|-------------------|
| 接触点相对地面 | 理想 ≈ **0** |
| 脚连杆原点相对世界系 | 可能 **≠ 0**（身体越过支撑脚，连杆几何点在动） |
| 脚相对机身 | 通常 **明显 ≠ 0** |

因此：**当前实现 ≠ 严格的零滑移约束** \(v_{\text{foot/ground}} = 0\)，而是对世界系下脚连杆速度的启发式惩罚。

更严谨的做法示例：

- 只惩罚**平行于地面**的分量（\(v_x, v_y\)）；
- 使用**接触点速度**（正运动学 + 接触点偏移）；
- \(v_{\text{foot}} - v_{\text{ground}}\)（平地时 ground 为 0）。

### 7.4 `unsqueeze(-1)` 的作用

`contact` 形状 `(N, 脚)` → `unsqueeze(-1)` → `(N, 脚, 1)`，与 `feet_vel` `(N, 脚, 3)` 广播相乘：着地脚保留速度，悬空脚置零。

---

## 8. 观测设计：普通观测 vs 特权观测

### 8.1 两套缓冲区

| 缓冲区 | 使用者 | G1 大致维度 |
|--------|--------|-------------|
| `obs_buf` | **Actor（策略）**，部署时可用 | 47 |
| `privileged_obs_buf` | **Critic（价值网络）**，仅训练 | 50 |

### 8.2 差异

`privileged_obs_buf` 比 `obs_buf` **多一项**：

```python
self.base_lin_vel * self.obs_scales.lin_vel   # 机体线速度（仅 critic）
```

其余：角速度、投影重力、指令、关节位/速、上一步 action、sin/cos 相位等一致。

### 8.3 术语：非对称 Actor-Critic / 特权信息（Privileged Information）

- **思想**：训练时让 Critic 看到更多“上帝视角”状态，估值更准、训练更稳；Actor 只使用部署时可获得的传感器信息，利于 **sim2real**。
- **线速度**：仿真中易获取，真机常需估计（噪声、延迟），故默认不放进 Actor 观测。

### 8.4 实践建议

| 场景 | 建议 |
|------|------|
| 使用非对称训练（默认） | 保持线速度仅在 `privileged_obs_buf` |
| **关闭**非对称，Actor/Critic 共用 `obs_buf` | 若线速度对任务重要且真机可获得 → **应加入** `obs_buf` |
| 真机无法获得线速度 | 不宜在 `obs_buf` 加入，避免训练/部署分布不一致 |

---

## 9. 相位观测 `sin_phase` / `cos_phase` 的意义

### 9.1 代码

```python
sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
# 拼入 obs_buf 与 privileged_obs_buf 末尾（各 1 维，共 2 维）
```

部署时 `deploy_mujoco.py` 用同样公式、同样 `period = 0.8` 生成相位。

### 9.2 作用（为何加入观测）

1. **步态时钟 / 节拍器**：告诉策略当前处于周期哪一段（支撑/摆动倾向）。
2. **打破对称、稳定周期步态**：便于学 \( \text{action} = f(\text{state}, \sin\phi, \cos\phi) \)。
3. **弥补无记忆网络的缺陷**：纯 MLP 难以从单帧推断“第几拍”；与 LSTM 可互补。

用 sin/cos 而非裸 `phase`：避免 0/1 边界不连续，且两维可唯一确定相位。

### 9.3 与 `_reward_contact` 的关系

| 问题 | 答案 |
|------|------|
| 是否专门为 `_reward_contact` 才加？ | **不完全是**；主要服务于策略的时序/周期结构，奖励是配套 |
| 去掉相位观测，`_reward_contact` 还有效吗？ | **环境侧仍有效**（`leg_phase` 在环境内计算）；但策略难以对齐隐含时钟，**更难学、样本效率更低** |
| 左右腿相位是否都进观测？ | **否**；仅**全局** sin/cos，左右区分靠 `leg_phase` 在奖励侧 + 策略从本体感觉学习 |

---

## 10. LSTM 与循环策略网络

### 10.1 定义

**LSTM**（Long Short-Term Memory，长短期记忆网络）：一种 **RNN（循环神经网络）** 变体，通过**门控机制**在长序列上缓解梯度消失，能跨多步保留有用信息。

### 10.2 与本项目的关系

`g1_config.py`：

```python
policy_class_name = "ActorCriticRecurrent"
rnn_type = 'lstm'
rnn_hidden_size = 64
rnn_num_layers = 1
```

策略每步：`(obs_t, h_{t-1}) → action_t, h_t`，隐藏状态 \(h\) 在步间传递。

### 10.3 与 MLP、相位观测的对比

| 结构 | 能否感知“步态第几拍” |
|------|---------------------|
| MLP + 当前观测 | 困难 |
| MLP + sin/cos 相位 | 较容易（显式时钟） |
| LSTM + 当前观测 | 可从历史推断，不保证学好 |
| LSTM + 相位 | 通常最省事 |

### 10.4 部署注意

加载 `policy_lstm_*.pt` 时需维护并每步更新 **hidden state**（及 cell state）；环境 reset 时应清零，否则记忆污染。

---

## 11. 概念速查表

| 术语 | 简要说明 |
|------|----------|
| **PD 控制** | 比例-微分控制：\(\tau = K_p(q_{target}-q) - K_d \dot{q}\) |
| **action_scale** | 动作缩放；目标角 = action × scale + default_angle |
| **开环相位** | 由仿真时间驱动，不反馈真实触地 |
| **支撑相 / 摆动相** | 本项目中 `leg_phase < 0.55` 近似为支撑，否则为摆动期望 |
| **Privileged obs** | 仅 Critic 训练时可见的额外状态 |
| **PPO** | 近端策略优化，本仓库默认 RL 算法 |
| **dim=2 的 norm** | 对最后一维（xyz）求向量模长 |
| **广播 (broadcasting)** | 不同形状张量按规则自动扩展后逐元素运算 |
| **世界系** | 固定在仿真世界坐标系，不随机器人运动 |

---

## 附录：G1 观测维度构成（约 47 维）

| 块 | 内容 | 维数（约） |
|----|------|------------|
| 角速度 | `base_ang_vel`（缩放） | 3 |
| 重力方向 | `projected_gravity` | 3 |
| 指令 | `commands[:, :3]` | 3 |
| 关节位置偏差 | `dof_pos - default` | 12 |
| 关节速度 | `dof_vel` | 12 |
| 上一步动作 | `actions` | 12 |
| 相位 | `sin_phase`, `cos_phase` | 2 |

特权观测在此基础上 **+3**（`base_lin_vel`）→ 约 50 维。

---

## 附录：主要奖励项（G1）

| 奖励名 | scale（config） | 作用概要 |
|--------|-----------------|----------|
| `tracking_lin_vel` | 1.0 | 跟踪线速度指令 |
| `contact` | 0.18 | 相位期望与触地一致 |
| `feet_swing_height` | -20.0 | 摆动相脚高约 0.08 m |
| `contact_no_vel` | -0.2 | 着地时压低脚连杆世界系速度（启发式） |
| `hip_pos` | -1.0 | 限制髋部关节偏离 |
| `alive` | 0.15 | 存活奖励 |

---

*文档生成日期：2026-05-15*

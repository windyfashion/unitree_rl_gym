# Legged Gym / Unitree RL Gym 学习笔记

> 主题：观测缩放（`cmd_scale` / `obs_scales`）、奖励项（`tracking_ang_vel`）、部署对齐、TensorBoard 解读、MuJoCo 转弯偏慢处理，以及 `unsqueeze` 含义。  
> 基于 `unitree_rl_gym` 代码与讨论整理。

---

## 1. 核心概念速查

| 名称 | 所在位置 | 作用 |
|------|----------|------|
| `obs_scales` | `legged_robot_config.py` → `normalization.obs_scales` | 训练时观测各维的固定缩放系数 |
| `commands_scale` | `legged_robot.py` 初始化 | `[lin_vel, lin_vel, ang_vel]`，用于缩放 `commands` 进观测 |
| `cmd_scale` | 部署 YAML（如 `deploy_mujoco/configs/g1.yaml`） | 部署时与训练 `commands_scale` 对齐 |
| `rewards.scales.tracking_*` | `g1_config.py` 等 | **奖励权重**，与观测缩放无关 |

**易混点：** `cmd_scale` ≠ `tracking_ang_vel`。前者改的是**网络输入**；后者改的是**优化目标里某项奖励的重要性**。

---

## 2. 观测缩放（Observation Scaling）

### 2.1 训练里怎么做

G1 环境在 `g1_env.py` 的 `compute_observations` 中拼接观测，例如：

- `base_ang_vel * obs_scales.ang_vel`
- `commands[:, :3] * commands_scale`（`commands_scale` 由 `obs_scales.lin_vel` 与 `obs_scales.ang_vel` 组成）
- `(dof_pos - default) * obs_scales.dof_pos`
- `dof_vel * obs_scales.dof_vel`
- `actions`、相位 `sin/cos` 等

`commands_scale` 在 `legged_robot.py` 中定义为：

```python
self.commands_scale = torch.tensor(
    [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
    device=self.device
)
```

默认配置（`legged_robot_config.py`）示例：

| 观测分量 | 默认 scale |
|----------|------------|
| 线速度 / 指令 vx, vy | `lin_vel = 2.0` |
| 角速度 / 指令 yaw | `ang_vel = 0.25` |
| 关节位置偏差 | `dof_pos = 1.0` |
| 关节速度 | `dof_vel = 0.05` |

### 2.2 部署里怎么做（MuJoCo）

`deploy_mujoco.py` 中：

```python
obs[6:9] = cmd * cmd_scale   # cmd_scale 默认 [2.0, 2.0, 0.25]
```

`cmd` 使用**物理单位**（m/s、rad/s）；`cmd_scale` 必须与训练时 `commands_scale` **完全一致**。

### 2.3 常见误解：要不要把 cmd 乘 1/scale？

**不要。** 若希望机器人以 **1 rad/s** 偏航运动，应设 `cmd` 第三维为 **1.0**，而不是 `4.0`（即不要除以 `0.25` 再输入）。

- 物理指令：`cmd_z = 1.0` rad/s  
- 网络看到：`1.0 × 0.25 = 0.25`（与训练时一致）  
- 若误填 `cmd_z = 4.0`，网络会认为收到了远大于训练分布的指令。

---

## 3. 为什么要加 scale？（设计思路）

### 3.1 分层设计

```
物理仿真 / 奖励计算  →  真实单位（m, s, rad）
        ↓
观测组装时 × 固定常数  →  送进策略网络的向量 o
        ↓
MLP π(o) / V(o)       →  深度学习优化
```

- **下层**：动力学、跟踪奖励仍用物理量，语义正确。  
- **上层**：仅对策略输入做线性缩放，改善数值条件。

### 3.2 具体收益（与深度学习相关）

1. **各维数值量级接近**：避免某一维（如大线速度）在 MLP 第一层主导激活，另一维（小角速度）几乎像常数。  
2. **优化更稳定**：Adam 等自适应方法下，各维有效步长不会差到离谱；减轻激活饱和。  
3. **与观测噪声一致**：`add_noise` 时各维信噪比更可控。  
4. **部署一致性**：train / sim2sim / sim2real 使用同一套 scale，避免分布漂移。

### 3.3 会不会「消除」物理信息？

- 在 scale **非零** 且 **train/deploy 一致** 时，这是**可逆的线性变换**，信息未丢失。  
- 网络在训练中已适配「缩放后的坐标」；部署必须复现同一变换。  
- **会出问题的情况**：训练与部署 scale 不一致 → 相当于换了输入坐标系。

### 3.4 与「梯度 / 贡献度」的关系（表述要准确）

- RL（PPO）主要对**网络参数**求梯度，环境通常**不可微**，不存在「对机身线速度误差直接反传」。  
- 若从局部看 \(\mathbf{o}_i = s_i x_i\)，则 \(\partial \pi / \partial x_i = s_i \cdot \partial \pi / \partial o_i\)：**scale 越大，该物理维在输入空间越敏感**。  
- 这不等于「线速度奖励梯度一定比角速度强」；**跟踪权重**由 `rewards.scales.tracking_*` 决定，与 `obs_scales` 是两套机制。

---

## 4. 奖励项 `tracking_ang_vel`

### 4.1 配置含义

`g1_config.py` 中：

```python
tracking_lin_vel = 1.0
tracking_ang_vel = 0.5
```

表示在总回报中**更强调线速度跟踪、相对弱化角速度跟踪**。这是常见任务取舍，**不是写反了**，也不是观测 scale 抄错。

### 4.2 是否生效？—— 完整计算链路

1. **`_parse_cfg`**：`reward_scales = class_to_dict(cfg.rewards.scales)`  
2. **`_prepare_reward_function`**（`__init__` 调用）：  
   - `scale == 0` 的项删除；  
   - 非零项：`reward_scales[key] *= dt`；  
   - 注册 `_reward_<name>`，如 `tracking_ang_vel` → `_reward_tracking_ang_vel`  
3. **每步 `post_physics_step` → `compute_reward`**：  
   ```python
   rew = self.reward_functions[i]() * self.reward_scales[name]
   self.rew_buf += rew
   self.episode_sums[name] += rew
   ```
4. **该项数学形式**（`legged_robot.py`）：  
   ```python
   ang_vel_error = (commands[:, 2] - base_ang_vel[:, 2]) ** 2
   return exp(-ang_vel_error / tracking_sigma)
   ```
   - `commands[:, 2]`：期望偏航角速度（rad/s）  
   - `base_ang_vel[:, 2]`：基座坐标系下实际偏航角速度  
   - `tracking_sigma`：默认 0.25，控制 \(\exp\) 对误差的敏感度  

### 4.3 TensorBoard 中 `rew_tracking_ang_vel ≈ 0.2` 如何理解

回合结束时（`reset_idx`）：

```python
extras['episode']['rew_' + key] = mean(episode_sums[key]) / max_episode_length_s
```

即：**整回合该项奖励之和 / 回合秒数** = 每秒平均贡献。

每步实际累加：

\[
\text{rew}_t = \exp(-e^2/\sigma) \times (0.5 \times dt)
\]

长时间平均后，日志量纲近似：

\[
\text{rew\_tracking\_ang\_vel} \approx 0.5 \times \mathbb{E}[\exp(-e^2/\sigma)]
\]

- 若曲线 **≈ 0.2**，则 \(\mathbb{E}[\exp] \approx 0.2 / 0.5 = 0.4\)：有跟踪，但误差不算小，**不代表项未启用**。  
- 与 `tracking_lin_vel`（系数 1.0）比，数值天生约低一半系数，不宜直接比绝对值判断「谁更重要」。

---

## 5. MuJoCo 部署：转弯偏慢怎么处理

### 5.1 推荐顺序

1. **先查指令**：`g1.yaml` 中 `cmd_init` 第三维（如 `0.5` rad/s）是否偏小；训练范围一般为 `ang_vel_yaw ∈ [-1, 1]`。  
2. **在训练范围内加大 `cmd`**：例如改为 `1.0`，不要用改 `cmd_scale`「骗」网络。  
3. **仍不够**：扩大 `commands.ranges.ang_vel_yaw` 并**重新训练**；可提高 `tracking_ang_vel` 权重。  
4. **次要**：微调 `kps`/`kds`，小步尝试，避免一次拉太大导致抖动。

### 5.2 不推荐

- 增大 `cmd_scale` 第三维以「加快转弯」—— 破坏 train/deploy 观测分布。  
- 超过训练最大偏航指令仍期望策略表现良好—— 属于分布外。

### 5.3 诊断

对比 `cmd` 第三维与 `d.qvel` 中偏航角速度（与 `deploy_mujoco.py` 中 `omega` 一致）：

- 实际 ω 已接近 cmd → 提高物理指令或重训更大偏航范围。  
- cmd 很大但 ω 跟不上 → 策略 / 奖励 / sim 差异，应调训练而非 scale。

---

## 6. PyTorch：`unsqueeze` 是什么

### 6.1 含义

**`unsqueeze(dim)`**：在指定维度 **插入一个长度为 1 的新维度**，不改变元素个数，只改变 **shape**。

### 6.2 示例

```python
x = torch.tensor([1.0, 2.0, 3.0])   # shape: (3,)
y = x.unsqueeze(0)                   # shape: (1, 3)  — 前面加 batch 维
z = x.unsqueeze(1)                   # shape: (3, 1)  — 每元素成一列
```

在 `g1_env.py` 中：

```python
sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
```

`self.phase` 形状为 `(num_envs,)`，`.unsqueeze(1)` 后变为 `(num_envs, 1)`，便于与 `(num_envs, num_obs)` 的 `obs_buf` 在最后一维 `torch.cat` 拼接。

### 6.3 与 `squeeze` 的关系

- **`squeeze`**：去掉长度为 1 的维度（反向操作）。  
- **`unsqueeze(i)`** ≈ **`view` / `reshape` 增加一维**，语义更直观。

---

## 7. 部署配置与训练默认值对照（G1）

| 项目 | 训练 (`obs_scales` / `commands_scale`) | 部署 (`g1.yaml`) |
|------|----------------------------------------|------------------|
| 线速度 / cmd xy | 2.0 | `cmd_scale[0:2] = 2.0` |
| 角速度 / cmd yaw | 0.25 | `cmd_scale[2] = 0.25` |
| 机体角速度观测 | `ang_vel_scale = 0.25` | `ang_vel_scale: 0.25` |
| 关节位置 | 1.0 | `dof_pos_scale: 1.0` |
| 关节速度 | 0.05 | `dof_vel_scale: 0.05` |
| 动作 | `action_scale = 0.25` | `action_scale: 0.25` |

---

## 8. 一句话记忆

| 问题 | 答案 |
|------|------|
| `cmd_scale` 干什么？ | 把物理指令缩放到训练时网络见过的观测尺度；**cmd 仍用真实单位**。 |
| 和 `tracking_ang_vel` 关系？ | 无关；后者是奖励权重。 |
| TensorBoard ~0.2 正常吗？ | 常表示 \(\exp\) 跟踪项平均约 0.4 量级，需结合 `0.5×dt` 与除以秒数理解。 |
| 转弯慢怎么办？ | 先加大 `cmd`（在训练范围内），再考虑重训与奖励权重，别改 `cmd_scale`。 |
| `unsqueeze`？ | 增加一个 size=1 的维度，便于 batch / cat。 |

---

## 9. 关键代码索引

| 文件 | 内容 |
|------|------|
| `legged_gym/envs/base/legged_robot_config.py` | `obs_scales`、`commands.ranges`、`rewards.scales` |
| `legged_gym/envs/base/legged_robot.py` | `commands_scale`、`_prepare_reward_function`、`compute_reward`、`_reward_tracking_ang_vel` |
| `legged_gym/envs/g1/g1_env.py` | G1 观测拼接、`unsqueeze(1)` |
| `legged_gym/envs/g1/g1_config.py` | G1 奖励权重覆盖 |
| `deploy/deploy_mujoco/deploy_mujoco.py` | `cmd * cmd_scale`、观测组装 |
| `deploy/deploy_mujoco/configs/g1.yaml` | 部署 scale 与 `cmd_init` |

---

*文档生成日期：2026-05-15*

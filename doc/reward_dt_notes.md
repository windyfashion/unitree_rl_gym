# Reward 计算中是否需要乘以 `dt`

## 核心结论

在强化学习、机器人控制、物理仿真、轨迹优化等连续时间系统中，reward 是否乘以 `dt` 取决于 reward 的语义。

**持续型 reward / cost 应该乘以 `dt`。**

**事件型 reward / penalty 通常不乘以 `dt`。**

可以用一句话判断：

> 这个 reward 是“每秒产生多少”，还是“每发生一次给多少”？

如果是“每秒产生多少”，乘 `dt`。

如果是“每发生一次给多少”，不乘 `dt`。

---

## 为什么持续型 reward 要乘 `dt`

连续时间任务的目标通常可以写成：

```text
maximize ∫ r(x(t), u(t)) dt
```

离散化之后，对应的是：

```text
maximize Σ r(x_k, u_k) * dt
```

也就是说，如果某个 reward 表示的是单位时间内的 reward rate，那么每一步实际获得的 reward 应该是：

```python
reward = reward_rate * dt
```

这样做的主要好处是：

1. reward 的量纲更加清楚；
2. 改变仿真步长时，总 reward 不会因为步数变化而明显改变；
3. reward 权重不容易和控制频率强绑定；
4. 同一套 reward 更容易迁移到不同 `dt`、frame skip 或控制频率下。

---

## 不乘 `dt` 会发生什么

假设每一步都有一个 tracking cost：

```python
reward = -abs(v - v_target)
```

如果不乘 `dt`：

- `dt = 0.02` 时，1 秒有 50 步；
- `dt = 0.01` 时，1 秒有 100 步。

同样的误差持续 1 秒，`dt = 0.01` 的累计 reward 绝对值会大约变成两倍。

这意味着任务目标本身会随着仿真频率改变。

如果乘以 `dt`：

```python
reward = -abs(v - v_target) * dt
```

那么同样的误差持续 1 秒，不同 `dt` 下的累计 reward 会更加接近。

---

## 应该乘 `dt` 的 reward 类型

以下 reward / cost 通常属于持续型，建议乘以 `dt`：

```python
alive_reward
position_tracking_cost
velocity_tracking_cost
orientation_tracking_cost
torque_cost
power_cost
energy_cost
action_smoothness_cost
joint_velocity_cost
joint_acceleration_cost
contact_force_cost
base_height_cost
```

典型写法：

```python
reward_rate = 0.0

reward_rate += alive_weight
reward_rate -= pos_weight * pos_error
reward_rate -= vel_weight * vel_error
reward_rate -= torque_weight * np.sum(torque ** 2)
reward_rate -= action_weight * np.sum(action ** 2)

reward = reward_rate * dt
```

这里的 `reward_rate` 可以理解为：

```text
单位：reward / second
```

而最终的 `reward` 是：

```text
单位：reward / step
```

---

## 不应该乘 `dt` 的 reward 类型

以下 reward / penalty 通常属于事件型，不建议乘以 `dt`：

```python
success_bonus
failure_penalty
fall_penalty
collision_penalty
goal_reached_bonus
termination_penalty
reset_penalty
```

典型写法：

```python
reward = reward_rate * dt

if success:
    reward += success_bonus

if fallen:
    reward -= fall_penalty

if collision:
    reward -= collision_penalty
```

原因是这些奖励或惩罚表示的是“一次事件”的价值，而不是“每秒持续产生”的价值。

例如：

```python
success_bonus = 100.0
```

含义是：

```text
成功一次，奖励 100
```

而不是：

```text
每秒成功奖励 100
```

因此不应该再乘以 `dt`。

---

## 推荐代码结构

比较清晰的 reward 结构是把持续型项和事件型项分开：

```python
def compute_reward(obs, action, prev_action, done, info, dt):
    reward_rate = 0.0

    # 持续型 reward / cost
    reward_rate += alive_weight
    reward_rate -= tracking_weight * tracking_error(obs)
    reward_rate -= torque_weight * torque_cost(obs)
    reward_rate -= smooth_weight * np.sum((action - prev_action) ** 2)

    reward = reward_rate * dt

    # 事件型 reward / penalty
    if info["success"]:
        reward += success_bonus

    if info["fallen"]:
        reward -= fall_penalty

    if info["collision"]:
        reward -= collision_penalty

    return reward
```

这种写法的语义非常明确：

```text
reward_rate：连续时间 reward rate
reward：当前 step 实际 reward
事件奖励：一次性加减，不随 dt 缩放
```

---

## `gamma` 也要考虑时间尺度

如果 reward 使用了 `dt` 缩放，并且希望不同控制频率下的任务尽量一致，那么 discount factor 也应该按时间尺度处理。

如果希望设置的是“每秒折扣系数”：

```python
gamma_per_second = 0.99
```

那么每一步的折扣应该是：

```python
gamma_step = gamma_per_second ** dt
```

例如：

```python
dt = 0.02
gamma_per_second = 0.99

gamma_step = gamma_per_second ** dt
```

此时：

```text
gamma_step ≈ 0.999799
```

也可以用连续折扣率：

```python
gamma_step = np.exp(-lambda_ * dt)
```

这样可以保持折扣的时间意义一致。

---

## 工程实践中的注意点

如果 `dt` 永远固定，不乘 `dt` 也可以训练成功。

但这样会带来一个问题：

```text
reward 权重和控制频率强绑定
```

一旦修改：

```python
dt
control_frequency
frame_skip
physics_substeps
```

原来的 reward 权重可能就需要重新调整。

因此，如果任务是连续控制任务，尤其是机器人或物理仿真任务，更推荐从一开始就使用：

```python
reward = reward_rate * dt
```

---

## 常见判断表

| Reward / Cost 类型 | 是否乘 `dt` | 原因 |
| --- | --- | --- |
| alive reward | 是 | 每秒持续存在 |
| tracking error | 是 | 持续跟踪误差 |
| torque cost | 是 | 持续能耗或控制代价 |
| action smoothness cost | 是 | 持续控制平滑约束 |
| joint velocity cost | 是 | 持续运动代价 |
| contact force cost | 是 | 持续接触力代价 |
| success bonus | 否 | 一次性成功事件 |
| fall penalty | 否 | 一次性失败事件 |
| collision penalty | 通常否 | 如果按碰撞事件计数，则不乘 |
| terminal penalty | 否 | episode 结束时的一次性惩罚 |

---

## 最终建议

在连续控制环境中，推荐使用以下规则：

```text
持续型 reward / cost：乘 dt
事件型 reward / penalty：不乘 dt
```

推荐代码模式：

```python
reward = reward_rate * dt

if event_happened:
    reward += event_bonus_or_penalty
```

如果希望不同 `dt` 下训练目标更加一致，还应配合：

```python
gamma_step = gamma_per_second ** dt
```

这样 reward 和 discount 都具有清晰的时间尺度含义。

# exp12 指标改善分析：为什么离 SOTA 还有差距，怎么改

> 创建：2026-06-12
> 背景：exp12-B 在 ep8 joint avgMinADE ≈ 1.94m 进入慢速区。focal 口径（新增打印）预计 ~0.9–1.1m。
> AV2 参照：focal-agent minADE(K=6) SOTA ~0.65m；multi-agent avgMinADE SOTA ~0.7–0.9m。
> 即：当前模型距 SOTA 约 **1.5–2×**，是真实差距，非纯指标口径问题。

---

## 一、根因诊断（按对最终指标的影响排序）

### 🔴 Tier 1：场景尺度被 LLM 架构压死（最可能的主因）

`max_agents=6, max_lanes=20`。这是**玩具尺度**——AV2 真实场景常有数十个 agent、上百条 lane polyline。SOTA（QCNet / MTR）建模 30–60+ agent、全图 polyline。

证据：富文本里出现 "Nearby agents within 50m: 20 total"，但模型只喂 6 个。joint avgMinADE 对全体 scored agent 求平均，**被截断丢弃的远端 agent 缺乏交互上下文，预测必然差**。

**为什么不能简单调大？** —— 这是 LLM-in-the-loop 设计的**结构性张力**：
```
LLM 输入序列 = [text | agent_emb×N | lane_emb×L | agt_tokens×N | scn]
```
N、L 每增大都直接拉长 28 层 LLM 的序列 → 计算量与显存爆炸（当前已 1.5s/it、84.8/96GB）。把 N=6→32、L=20→128 会让 LLM 部分增重 ~5×，单 epoch 从 23min → ~2h，且大概率 OOM。

**这是核心矛盾：LLM 把每个 agent/lane 当 token 过 28 层，从根上限制了场景尺度，而场景尺度正是 SOTA 的命脉。**

### 🟠 Tier 2：mode / anchor 覆盖太粗

K=6 个 mode，oracle-winner 占比 m2≈33% 一家独大，其余 8–23%。joint minADE 是 oracle-over-K，**K 个 mode 张不开真实多模态时，minADE 直接触底**。当前 anchor 偏静态（anchor[1]≈(3,0) 静止、anchor[0]≈(85,0) 高速直行），6 个锚点覆盖所有场景类型过于粗糙。

### 🟡 Tier 3：表达力 / 容量细节

- Bézier degree-5（6 控制点）对 6s 复杂机动（变道+转弯的曲率叠加）可能偏平滑。
- `n_levels=2` 的 Level-k 交互层数偏少。
- LoRA r=32 对 0.6B 模型够用，非瓶颈。

---

## 二、改善杠杆（ROI × 成本）

### 方案 A：解耦「场景尺度」与「LLM 序列」⭐ 最高 ROI，治本

**核心思想**：LLM 不再吃全部 agent/lane token，只吃压缩输入产出 `h_scn`；全分辨率的 encoder + decoder 在 LLM 之外对 N=32 全场景运行。

```
LLM 输入  = [text | (少量 pooled scene tokens) | scn]      ← 序列短，可控
decoder   = 全场景 AgentEncoder(N=32) + LaneEncoder(L=128) + Level-k
                ↑ 用 h_scn 做 FiLM/cross-attn 条件，不进 LLM
```

- **收益**：场景尺度可拉到 SOTA 量级，同时保留「LLM 场景引导」论点。
- **成本**：中等重构（model forward 需拆开 N-耦合），但不动训练框架。
- **风险**：改完需重训；h_scn 的条件注入方式要重新调。
- **与论文的关系**：反而**强化**创新点一——证明 LLM 引导在「正常尺度」decoder 上依然有效，比当前玩具尺度更有说服力。

### 方案 B：温和扩容（不改架构，先验证天花板）

在当前架构内把 `max_agents 6→12`、`max_lanes 20→40`，batch 相应减半（54→24），看 minADE 是否明显下降。
- **收益**：快速验证「尺度是否是主因」——若 12 agent 就明显改善，方案 A 的价值被证实。
- **成本**：低（只改超参），但 LLM 序列变长，速度/显存吃紧，可能需降 batch + 加 grad_accum。
- **定位**：诊断性实验（exp13），为方案 A 提供依据。

### 方案 C：mode / anchor 增强（低成本，叠加增益）

- `--num_modes 6→10`，扩大 oracle 覆盖（几乎零成本，直接降 minADE）。
- anchor 初始化改为**对训练集 GT 终点做 k-means**，替代/预热当前 EMA codebook——给更贴数据分布的锚点。
- **收益**：minADE 通常能吃到 0.1–0.2m，便宜。
- **成本**：极低。

### 方案 D：表达力细调（边际）

- Bézier degree 5→7（8 控制点）。
- `n_levels 2→3`。
- **收益**：边际，0.05m 量级。最后再调。

---

## 三、建议执行顺序

1. **先做方案 C**（num_modes=10 + k-means anchor）——零成本，叠加在任何后续实验上。
2. **方案 B 诊断**（max_agents=12）——确认尺度是不是主因。一个 epoch 就能看出趋势。
3. 若 B 确认尺度主因 → **投入方案 A 重构**（解耦 LLM 与场景尺度），这是冲 SOTA 的唯一治本路径。
4. 方案 D 留到最后定型时调。

## 四、对论文的提醒

**绝对 SOTA 不是这篇论文的必要条件**。核心贡献是 exp12-B − exp12-A 的 LLM 场景引导净增益（创新点一），这个 delta 在任何场景尺度下都成立。若冲 SOTA 成本过高，可：
- 在「受控尺度」下报告完整消融（诚实标注 max_agents=N 的设定），论点依然完整；
- 把方案 A 列为 future work / 或作为第二篇的工程贡献。

---

---

## 五、【2026-06-12 修订】SOTA 冲刺路线（已定为主线，放弃消融分支）

### 关键事实更正：场景扩容免费 + 序列开销被高估

1. **无需重新预处理**：`_get_lane_features` 读 pkl 全量 lane 只取 `[:max_lanes]`（`av2_hybrid_dataset.py:408`）；
   `neighbors_info` 全场景排序后取 `[:max_agents-1]`（`:300`）。调大 max_agents/max_lanes 只是少截断。
2. **序列开销 = max_text_len + 2N + L + 1**。当前 633 中 **600 是文本**；max_agents 6→20、max_lanes 20→64
   后仅 705（+11%）。**文本是计算大头，场景扩容几乎免费**（之前"N→32 重 5×"的判断作废）。
3. **全是现成参数**：`--max_agents --max_lanes --num_modes --n_levels --n_bezier_ctrl`，exp13 零代码改动。

### exp13：场景尺度（最高 ROI，立即可跑）

**num_modes 保持 6**：AV2 标准评测是 minADE₁/minADE₆，K 必须与评测对齐；K=10 训练按 K=6 报告
是指标错配（虚增 oracle 覆盖），违背「超参贴合实际评测」原则。模式覆盖不作为涨点手段。

**文本/数值 agent 数解耦（2026-06-12 本地实测后新增）**：本地 `debug/demo_text_exp13.py`
测得文本 token 随文本列举的 agent 数线性增长（每 agent ≈ +28 token）：max_agents=6→425
median，=20→682 median（57.5% 超 600 被截断）。多出来的全是 "parked, low influence" 停放车，
属 DPI 冗余（数值通道已有）。故新增 `--max_text_agents`（dataset 解耦文本列举数与数值通道
agent 数）：**数值吃满 20 agent，文本只列 top-5 影响力 agent**。实测 max_agents=20 +
max_text_agents=6 时文本回到 425 median / 489 max → **max_text_len=512 即 0% 截断**。
序列 512+2×20+64+1=617 ≈ exp12-B 的 633，**场景尺度 3× 几乎零额外计算**。

| 旋钮 | exp12 | exp13 | 理由 |
|---|---|---|---|
| max_agents | 6 | **20** | 数值通道：Level-k 交互 + 编码上下文变富（joint 指标对全 scored agent 平均）|
| max_lanes | 20 | **64** | 地图约束是 AV2 minADE 主驱动；免费扩容 |
| **max_text_agents** | (6) | **6（不变）** | 文本只列 top-5；与数值解耦，防序列爆长 |
| max_text_len | 600 | **512** | 解耦后文本 max 489，512 即 0% 截断；序列回到 ~617 |
| num_modes | 6 | **6（不变）** | 对齐 AV2 K=6 评测 |
| train_batch / grad_accum | 54/1 | **48/1** | 序列≈不变；数值编码器 +20 agent/64 lane 占少量额外显存 |
| early_stop_patience | 5 | **8** | 更大容量收敛更慢，防早停误杀 |

其余同 exp12-B（LoRA r32、T_0=40、scene_guidance_scale=1.0）。**纯场景尺度扩容,单步最大增益来源。**

代码改动（已落地，无需重预处理）：
- `data_av2/av2_hybrid_dataset.py`：`__init__` 加 `max_text_agents`；`compose_scene_text` 将
  文本列举（`text_neighbors`）与数值 neighbor_ids（`selected`）解耦。
- `train_hybrid.py`：加 `--max_text_agents`，透传给 train/val dataset。

### exp14（exp13 之后，需少量代码）
- ~~k-means anchor 初始化~~ **【已弃用，2026-06-12】**：两点原因——(1) exp12 后 AnchorCodebook
  是**纯诊断**，不进预测路径（mode query = base_q + scene_proj(h_scn)，轨迹基于每 agent
  gt_anchor），对它 k-means 对 minADE 零影响；(2) 多样性来源 `base_q`(mode_embeds) 已是随机
  初始化 + 反向传播的**自学习**涌现，符合"无监督/不预计算"的设计哲学。补充澄清：在 train set
  上挖锚点本身**不是**测试集泄漏（标准做法、实际可行），但在本架构里无效，故弃用。
- **n_levels 2→3、n_bezier_ctrl 6→8**：交互层数 + 轨迹表达力（纯容量，全自学习；显存允许时叠加）。
- **若模式覆盖被证实为瓶颈**，用**在线、不预计算**的多样性机制（与自学习理念一致）：
  mode 间排斥/多样性正则、winner-takes-most 软分配、scene-level 熵正则——均训练中涌现，无需提前看 GT。

### exp15（若仍短于 SOTA，需重构）
- **AgentEncoder 时序建模升级**：当前 per-frame MLP + **MaxPool**（丢时序顺序）→ 换轻量 temporal attention / attention-pool。SOTA 普遍用时序 transformer，这可能是继尺度之后的下一个瓶颈。
- **数据增强**：轨迹旋转、agent dropout（AV2 标准涨点手段）。

### 执行原则
- 一次主改一个大杠杆（exp13 = 尺度+模式），便于归因增益来源。
- 每个 exp 跑到 minADE 平台（patience 触发）再决定下一步。

## 待办
- [x] focal-agent minADE 打印已加入 `train_hybrid.py`
- [ ] **exp13：max_agents=20 + max_lanes=64（num_modes=6 不变；主线，立即跑）**
- [ ] exp14：n_levels=3 + bezier=8（k-means anchor 已弃用）
- [ ] exp15（备选）：AgentEncoder 时序升级 + 数据增强

# exp12 改进方案：Scenario-Query Joint Decoder（2026-06-10）

## 诊断回顾（exp11 9m plateau 的根因）

| # | 问题 | 证据 |
|---|---|---|
| P1 | **anchor 绑定的 mode 语义与 joint 协议冲突**：`mode_e = base_q + anchor_mlp(anchor_k)` 给每个 agent 注入同一终点先验；场景共享 winner mode 无法同时满足异质 agent | minFDE/minADE≈2.4（直线射线特征）；Level 层间差异消失 |
| P2 | **Bézier 控制点尺度病态**：MLP head 初始输出 O(1)m，目标 span 0–90m | ep1–7 线性 ~1m/ep 收敛后停滞 |
| P3 | **loss 粒度不一致**：anchor_cls 教 per-agent 语义，reg/cls 教 per-scene 语义；soft α=0.1 把所有 mode 拉向每个场景 GT | loss 持续降、joint minADE 不动 |
| P4 | **LLM 无场景级引导通道**：只能逐 agent 加修正向量，mode query 被 anchor 占据 | corr/enc≈1.19 但指标停滞 |
| P5 | LLM 梯度被饿：gn_llm pre-clip=10 vs clip=1.0；last-batch padding artifact | ep10 log |

设计目标（用户确认）：**multi-agent joint 预测**；LLM 的角色 = **理解场景、引导预测**。

---

## 改动集

### A. MLPDecoder 重构（核心）

**A1. 移除 anchor 注入，mode query 改为 LLM 场景条件化**

```python
# 旧: mode_e = base_q[k] + anchor_mlp(anchor_k)        # agent 无关的终点先验
# 新: mode_e = base_q[k] + scene_proj_k(h_scn)         # LLM 场景理解塑形 K 个 joint scenario
```

- `h_scn [B, H]`：LLM 序列新增 1 个学习的 `[SCN]` scene summary token（排在 AGT tokens 之后），
  取其 hidden state。LLM 因此获得显式的场景级输出通道。
- `scene_proj`: `Linear(H, K*H)`（或 K 个独立小投影），**零初始化** → 训练起点等价于
  纯 `base_q`，与既有零初始化残差模式（llm_correction_proj / lane_cross / out_projs）一致。
- mode 语义变化：从"去 anchor_k 的射线"→"第 k 种 joint 未来假设"，每个 agent 在
  scenario k 下由自身特征决定行为 → 与场景共享 winner 协议天然一致。

**A2. CV（恒速）先验残差化 Bézier 控制点**

```python
v_last = agent_features[..., -1, 2:4]                  # [B, N, 2] focal 系，数据已有
# 恒速直线的 degree-5 Bézier 等价控制点（均匀分布在速度射线上，C0=0 天然成立）:
cv_ctrl[i] = v_last * T_horizon * i / (n_ctrl - 1)     # i = 0..5, T_horizon=6.0s
ctrl = cv_ctrl.unsqueeze(mode_dim) + self.mlp(x).view(BN, K, n_ctrl, 2)   # MLP 只预测残差
```

- 残差量级 ≈ 几米（对恒速外推的偏离），与 MLP 初始化尺度匹配 → 解除 P2
- 先验逐 agent 条件化（每个 agent 自己的速度），适配 joint mode；静止 agent 先验=0，退化安全
- 替代了 anchor 先验丢失的几何信息，且不再有"agent 无关"的缺陷

### B. Loss 重构

- **删除 `anchor_cls_loss`**（per-agent anchor 语义已不存在）；AnchorCodebook 保留为
  纯诊断（观察终点分布），不再进入前向与 loss。
- **`soft_wta_alpha` 0.1 → 0.0**（纯 scene-level WTA，SceneTransformer/joint 预测标准做法）。
  α>0 会把所有 mode 拉向条件均值，直接对抗 mode 多样性，且是 P3 loss-metric 脱耦的来源之一。
- 保留：position SmoothL1 + scene winner、scene-level cls CE。训练目标与 val 指标完全同构。

### C. 训练卫生

- train DataLoader `drop_last=True`（消除 last-batch padding 梯度尖峰，已排队两轮）
- `--llm_grad_clip` 1.0 → 2.5（ep10 实测 pre-clip norm 达 10，clip=1.0 把 LLM 更新截到 10%）
- 新增诊断：`diag_scn_ratio = ‖scene_proj(h_scn)‖ / ‖base_q‖`（监控 LLM 场景引导的激活进度，
  对应原 corr/enc 的监控模式）

---

## 问题 → 改动映射

| 问题 | 解决它的改动 |
|---|---|
| P1 anchor-joint 冲突 | A1（移除注入）+ B（删 anchor_cls） |
| P2 控制点尺度 | A2（CV 残差） |
| P3 loss 脱耦 | B（α=0，粒度统一为 scene） |
| P4 LLM 无引导通道 | A1（[SCN] token → mode query） |
| P5 梯度卫生 | C |

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 失去 anchor_cls 后 mode collapse（K 个 mode 退化成一个） | 纯 scene-WTA 本身是 joint 预测标准配置；监控每 epoch 的 winner mode 使用直方图熵；若坍缩，回退方案 = 加场景级 winner 分布最大熵正则（不回 per-agent anchor） |
| LLM scene token 冷启动慢 | scene_proj 零初始化，与既有残差模式一致；diag_scn_ratio 监控激活速度 |
| CV 先验对转弯 agent 偏差大 | 残差 MLP 学偏离量；Bézier 平滑性保证修正轨迹合理；转弯偏离 ~10-20m 仍远小于 0-90m 的裸回归 |
| 早期收敛可能比 exp11 慢（少了 anchor 几何先验） | CV 先验提供了更强的逐 agent 几何先验，预计反而更快 |

---

## 消融设计（论文归因，关键）

| 实验 | 配置 | 归因 |
|---|---|---|
| exp11（已有） | anchor modes + 裸 Bézier | baseline |
| exp12-A | 新 decoder，`scene_proj` 强制 scale=0 | 隔离 **decoder 重构**（A2+B）的增益 |
| exp12-B | 完整（LLM scene guidance 开启） | 隔离 **LLM 场景引导**（A1）的增益 |

exp12-B − exp12-A = LLM scene guidance 的净贡献——这是"LLM 用于轨迹预测"论点的直接证据，
与 enriched text 实验（new_exp_20260608.md）互补。

---

## 涉及文件

- `simpl/hybrid_llm_model.py`：`MLPDecoder`（mode query、CV 先验、forward 签名加 `v_last`）；
  `HybridLLMPredictor.forward`（提取 v_last、[SCN] token、h_scn 传递）；`_build_inputs_embeds`
  （序列 +1 token，AGT 位置索引不变，SCN 在末尾）；诊断
- `simpl/av2_llm_loss.py`：移除 anchor_cls 分支；codebook 参数变 optional
- `simpl/hybrid_loss.py`：权重透传同步
- `train_hybrid.py` / `train_hybrid_local.py`：args 同步（α 默认 0、llm_grad_clip 默认 2.5、
  `--scene_guidance_scale` 消融开关）、drop_last、诊断打印
- 需**从头训练**（mode query 与 decoder head 语义全变，不可 resume）

## 训练命令（exp12-B，4×GPU）

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
torchrun --nproc_per_node=4 train_hybrid.py \
  --features_dir data_av2/feature \
  --max_text_len 512 \
  --train_batch_size 72 --val_batch_size 72 \
  --train_epoches 40 --val_interval 1 \
  --n_levels 2 --flash_attn --dist_backend gloo \
  --llm_lr 5e-5 --gru_lr 1e-4 \
  --grad_clip 5.0 --llm_grad_clip 2.5 \
  --lora_r 32 --lora_alpha 64 --lora_targets all-linear \
  --soft_wta_alpha 0.0 \
  --warmup_epochs 5 --T_0 20 --early_stop_patience 10 \
  --num_workers 8 \
  --exp_name exp12_scenario_joint \
  --ckpt_dir saved_models/exp12_scenario_joint \
  --logger_writer
```

（exp12-A 消融：同命令 + `--scene_guidance_scale 0.0`，exp_name 改 `exp12a_no_scn`）

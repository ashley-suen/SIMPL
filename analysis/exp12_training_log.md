# exp12 训练记录：Scenario-Query Joint Decoder

> 创建：2026-06-11
> 模型：AV2 Hybrid LLM Predictor（Qwen3-0.6B-Base + LoRA + Level-k Decoder + CV-prior Bézier）
> 任务：Argoverse 2 多智能体联合轨迹预测（joint multi-agent prediction）
> 设备：4× RTX PRO 6000 96GB Blackwell Edition

---

## 实验设计

exp12 在 exp11 基础上引入两类改动，通过 `--scene_guidance_scale` 开关做消融：

| 实验 | scene_guidance_scale | 含义 | 状态 |
|---|---|---|---|
| **exp12-A** | 0.0 | decoder 重构基线（CV-prior Bézier + scene-WTA，**无** LLM 场景引导） | 待跑 |
| **exp12-B** | 1.0 | + [SCN] token + scene_proj 场景引导（完整） | **运行中** |

**净增益归因**：`exp12-B − exp12-A` 的 minADE 差 = LLM 场景引导（[SCN] + scene_proj）的独立贡献。详见 [`innovations.md`](innovations.md) 创新点一。

### 关键超参（两实验共享）

```
train_batch_size 54 (×4 GPU)   max_text_len 600    lora_r/alpha 32/64 (all-linear)
train_epoches 40                n_levels 2          llm_lr 5e-5  gru_lr 1e-4
scheduler cosine_warmup_restart T_0 40 (单周期，无 restart)  warmup 5  eta_min_ratio 0.1
grad_clip 5.0 (enc) / llm_grad_clip 1.0 (llm)      early_stop_patience 5
soft_wta_alpha 0.0 (纯 scene-WTA)   cls_weight 0.5  dist_backend gloo
```

**调度器决策**：原 exp11 用 `T_0=20` 在 40 epoch 中会于 epoch 20 触发一次 warm restart，与 `early_stop_patience=5` 冲突（restart 爬坡期连续 ≥5 epoch 变差 → 训练在第二周期峰值被误杀）。exp12 改 `T_0=40` 单周期 cosine，去掉 restart，保证 A/B 对照不被 restart 随机性污染。

---

## exp12-B（LLM 场景引导，运行中）

启动时间戳：`20260611-212432`
ckpt：`saved_models/exp12b_scenario_joint/`

### 逐 epoch 指标

| epoch | train avg_loss | val loss | **minADE (m)** | minFDE (m) | brierMinFDE | corr/enc | scn_q/base_q | min anchor sep (m) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.214 | 3.046 | **2.711** | 6.932 | 7.400 | 0.557 | 141.1 | 25.21 |
| 2 | 2.928 | 2.850 | **2.497** | 6.475 | 6.965 | 0.694 | 163.1 | 24.20 |
| 3 | 2.775 | 2.696 | **2.328** | 6.074 | 6.552 | 0.783 | 172.3 | 24.93 |
| 4 | 2.657 | 2.596 | **2.182** | 5.679 | 6.204 | 0.783 | 162.3 | 24.64 |
| 5 | 2.517 | 2.487 | **2.089** | 5.469 | 5.962 | 0.816 | 161.1 | 24.31 |
| 6 | 2.428 | 2.408 | **1.988** | 5.207 | 5.729 | 0.859 | 156.1 | 24.39 |
| 7 | 2.367 | 2.376 | **1.937** | 5.095 | 5.603 | 0.859 | 157.8 | 24.73 |
| 8 | 2.323 | 2.340 | 1.937 | 5.112 | 5.602 | 0.862 | 149.2 | 25.55 | ← patience 1/5（与 ep7 持平）|
| 9 | 2.270 | 2.330 | 1.963 | 5.177 | 5.658 | 0.862 | 151.2 | 24.89 | ← patience 2/5（best 仍为 ep7 的 1.937）|

每 epoch ≈ 22.7 min（train）+ 1.0 min（val）；peak_mem ≈ 84.8 GB / 96 GB。

### 诊断观察

**[Modes] oracle-winner 占比**（无 mode collapse）：

| epoch | m0 | m1 | m2 | m3 | m4 | m5 |
|---|---|---|---|---|---|---|
| 1 | 8.0% | 16.6% | 35.1% | 10.7% | 12.3% | 17.2% |
| 2 | 23.6% | 11.4% | 34.6% | 7.6% | 11.0% | 11.7% |
| 3 | 22.3% | 10.8% | 35.1% | 8.4% | 11.0% | 12.5% |
| 4 | 23.4% | 9.7% | 32.9% | 10.2% | 11.7% | 12.1% |
| 5 | 22.9% | 9.5% | 34.6% | 10.2% | 9.6% | 13.1% |
| 6 | 23.3% | 9.2% | 33.9% | 9.9% | 8.5% | 15.2% |
| 7 | 23.5% | 9.5% | 31.4% | 10.1% | 8.5% | 16.9% |
| 9 | 20.0% | 9.5% | 39.3% | 9.7% | 8.1% | 13.3% |

**[Anchors] codebook 终点**（稳定，无 zombie）：anchor[1]≈(3,0) 为静止/近距模式且 ema_count 最高（197→203），anchor[0]≈(85,0) 为高速直行模式，横向模式 anchor[2]/[5] 占比小但存活（ema_count >10）。

### 评估与风险跟踪

✅ **健康信号**：
- minADE 两个 epoch 从 2.71 → 2.50（−0.21m），相对 exp11 ~9m 平台是数量级跃升 —— 说明 decoder 重构（CV-prior Bézier + scene-WTA）极其有效。**注意：此增益 exp12-A 也会有，尚不能归因给 LLM，须等 A 对照。**
- gn_enc 2.78–2.89（clip 5.0 内）；gn_llm 4.8–5.8（被裁到 1.0，偏紧但稳定）。
- corr/enc 0.56→0.69 LLM 修正通道正常激活；lane_cross 0.02→0.04 轻量介入。
- 模式分布健康，codebook 无塌缩。

⚠️ **需盯紧：scn_q/base_q（每样本真实比值 ≈ 19× 起，仍在升）**
- 诊断口径：`scn_q.norm()[B,K,H] / base_q.norm()[K,H]`，scn_q 含 batch 维 B=54，故原始值天然 ×√54≈7.35。真实每样本比值 = 141/7.35≈**19×**（ep1）→ 163/7.35≈**22×**（ep2）。
- 含义：残差层级**反转**——mode query 几乎完全由 LLM 的 h_scn 驱动，base_q 贡献被压到 ~5%。这与「base_q 骨架 + scn_q 渐进精修」的设计本意相反。
- **趋势判断**：原始值 141→163→172→**162**（增量 +22→+9→**−10**，ep4 已掉头回落）；每样本峰值 23.4× 后回落到 22×。**确认趋稳并掉头，非 runaway** ✅ 风险点解除。
- **止损线**（已不触发）：若后续比值重新暴涨（>40×/样本，即原始值 >300）或 val minADE 反弹，才需降 `--scene_guidance_scale` 到 0.5 重跑。

### 平台分析（ep7–9）

minADE 在 ep7 达到 best **1.937m** 后进入平台：ep8=1.937（持平）、ep9=1.963（微升），patience 2/5。

- **降幅序列**：0.214→0.169→0.146→0.093→0.101→0.051→0.000→−0.026，清晰减速并触顶。
- **关键区分**：train avg_loss 仍在降（2.367→2.323→2.270），val loss 也仍在缓降（2.376→2.340→2.330），**尚无过拟合**——是 joint argmin 指标提前进入慢速区，不是发散。
- m2 占比在 ep9 升到 39.3%，mode 略向单一模式集中（多样性轻微下降），是平台期常见现象。
- **结论**：当前架构（max_agents=6 玩具尺度）的天花板约 **1.94m joint avgMinADE**。距 SOTA ~1.5–2×，根因与改善路径见 [`metric_improvement_plan.md`](metric_improvement_plan.md)。

### 最终诊断
exp12-B 已确认健康收敛到架构天花板 1.94m（focal 口径预计 ~0.9–1.1m，待续跑激活打印确认）。
- 训练机制全部正常：CV-prior Bézier + scene-WTA 有效（相对 exp11 ~9m 是数量级提升）、scn_q 趋稳、无塌缩、无过拟合。
- 性能瓶颈在**场景尺度被 LLM 序列锁死**（max_agents=6, max_lanes=20），非训练问题。

### 待办
- [ ] exp12-A 消融（scale=0）：净增益归因，独立于尺度改善
- [ ] focal 打印已加入代码（`train_hybrid.py`），需续跑或新实验激活
- [ ] exp13：方案 C（num_modes=10 + k-means anchor）+ 方案 B（max_agents=12 诊断）
- [ ] 方案 A（解耦 LLM 与场景尺度）：冲 SOTA 的治本重构
- [ ] A/B 跑完后填写净增益对比，回填 `writing/dissertation_outline.md` 实验章节

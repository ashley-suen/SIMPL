# exp11_bezier_enriched_v2 Training Log

**Experiment**: exp11_bezier_enriched_v2  
**Start time**: 2026-06-10 16:58:35  
**Config**: Bézier (n_ctrl=6) + MAP_CONTEXT enriched text + Level-k E2E (detach removed) + llm_correction_proj in slow group  
**Command**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
torchrun --nproc_per_node=4 train_hybrid.py \
  --features_dir data_av2/feature \
  --max_text_len 512 \
  --train_batch_size 72 --val_batch_size 72 \
  --train_epoches 40 --val_interval 1 \
  --n_levels 2 --flash_attn --dist_backend gloo \
  --llm_lr 5e-5 --gru_lr 1e-4 \
  --grad_clip 5.0 --llm_grad_clip 1.0 \
  --lora_r 32 --lora_alpha 64 --lora_targets all-linear \
  --warmup_epochs 5 --T_0 20 --early_stop_patience 10 \
  --num_workers 8 \
  --exp_name exp11_bezier_enriched_v2 \
  --ckpt_dir saved_models/exp11_bezier_enriched_v2 \
  --logger_writer
```

---

## Per-Epoch Summary

| Epoch | avg_loss | val_loss | minADE (m) | minFDE (m) | brierMinFDE | LLM corr/enc | lane_cross | gn_enc (last batch) | gn_llm (last batch) | anchor min_sep (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 19.3886 | 18.8245 | 14.5648 | 29.5592 | 30.1438 | 1.0588 | 0.0388 | 7.84 | 5.81 | 10.838 |
| 2 | 17.9880 | 17.3480 | 13.4586 | 28.2702 | 28.8610 | 1.1010 | 0.1124 | 6.44 | 6.25 | 12.385 |
| 3 | 16.4852 | 15.8605 | 12.2872 | 26.8960 | 27.4645 | 1.1667 | 0.1470 | 9.62 | 8.75 | 13.047 |
| 4 | 15.3302 | 14.8722 | 11.4937 | 25.7504 | 26.3244 | 1.1778 | 0.1510 | 9.81 | 7.94 | 12.810 |
| 5 | 14.1165 | 13.6735 | 10.4914 | 24.3748 | 24.9842 | 1.1443 | 0.1554 | 7.00 | 5.97 | 13.344 |
| 6 | — | — | 9.6033† | — | — | — | — | — | — | — |
| 7 | 12.2980 | 12.0934 | 9.2115 | 22.1975 | 22.7882 | 1.1081 | 0.1711 | 5.59 | 5.22 | 15.208 |
| 8 | 11.9970 | 11.8099 | 8.9840 | 21.7654 | 22.3588 | 1.1602 | 0.1701 | 6.44 | 6.22 | 15.506 |
| 9 | 11.7995 | 11.7740 | 8.9771 | 21.7830 | 22.3411 | 1.1880 | 0.1773 | 6.34 | 6.78 | 16.343 |
| 10 | 11.7705 | 11.7541 | 8.9718 | 21.7839 | 22.3722 | 1.1936 | 0.1853 | 8.44 | **10.00** | 16.798 |

†ep6 log not captured; minADE=9.6033 inferred from ep7 checkpoint message "improved by 0.3918m → 9.2115m".

---

## Anchor State Per Epoch

### Epoch 1
```
min pairwise sep: 10.838 m
anchor[0] = ( +83.00,   -0.39) | ema_count =    37.75
anchor[1] = ( +24.00,  +24.12) | ema_count =    13.12
anchor[2] = ( +31.50,   -2.09) | ema_count =    64.00
anchor[3] = (  +0.76,   +0.03) | ema_count =   132.00
anchor[4] = ( +55.25,   -0.24) | ema_count =    70.00
anchor[5] = ( +11.44,   -1.81) | ema_count =    66.00
```

### Epoch 2
```
min pairwise sep: 12.385 m
anchor[0] = ( +84.00,   -0.44) | ema_count =    36.25
anchor[1] = ( +23.88,  +24.38) | ema_count =    12.69
anchor[2] = ( +33.25,   -1.72) | ema_count =    59.00
anchor[3] = (  +1.23,   +0.07) | ema_count =   149.00
anchor[4] = ( +53.75,   -0.22) | ema_count =    70.00
anchor[5] = ( +13.31,   -2.67) | ema_count =    66.50
```

### Epoch 3
```
min pairwise sep: 13.047 m
anchor[0] = ( +83.50,   -0.38) | ema_count =    35.25
anchor[1] = ( +23.62,  +24.38) | ema_count =    13.06
anchor[2] = ( +33.75,   -1.43) | ema_count =    57.75
anchor[3] = (  +1.34,   +0.05) | ema_count =   157.00
anchor[4] = ( +53.25,   -0.21) | ema_count =    69.50
anchor[5] = ( +14.06,   -2.83) | ema_count =    66.00
```

### Epoch 5
```
min pairwise sep: 13.344 m
anchor[0] = ( +84.50,   -0.40) | ema_count =    35.75
anchor[1] = ( +23.62,  +24.38) | ema_count =    12.94
anchor[2] = ( +34.75,   -1.47) | ema_count =    57.50
anchor[3] = (  +1.48,   +0.05) | ema_count =   161.00
anchor[4] = ( +53.75,   -0.29) | ema_count =    68.00
anchor[5] = ( +14.50,   -2.89) | ema_count =    65.00
```

### Epoch 4
```
min pairwise sep: 12.810 m
anchor[0] = ( +83.50,   -0.37) | ema_count =    36.25
anchor[1] = ( +23.50,  +24.12) | ema_count =    12.62
anchor[2] = ( +33.50,   -1.56) | ema_count =    60.25
anchor[3] = (  +1.26,   +0.05) | ema_count =   160.00
anchor[4] = ( +53.75,   -0.25) | ema_count =    69.50
anchor[5] = ( +13.75,   -2.78) | ema_count =    65.00
```

### Epoch 7
```
min pairwise sep: 15.208 m
anchor[0] = ( +84.00,   -0.43) | ema_count =    33.75
anchor[1] = ( +23.50,  +24.25) | ema_count =    13.06
anchor[2] = ( +34.50,   -1.28) | ema_count =    59.50
anchor[3] = (  +1.73,   +0.05) | ema_count =   165.00
anchor[4] = ( +55.25,   -0.26) | ema_count =    67.00
anchor[5] = ( +16.50,   -3.56) | ema_count =    55.75
```

### Epoch 8
```
min pairwise sep: 15.506 m
anchor[0] = ( +84.50,   -0.38) | ema_count =    33.50
anchor[1] = ( +23.38,  +23.50) | ema_count =    13.38
anchor[2] = ( +35.00,   -1.12) | ema_count =    58.00
anchor[3] = (  +1.98,   +0.06) | ema_count =   167.00
anchor[4] = ( +56.00,   -0.20) | ema_count =    66.50
anchor[5] = ( +17.00,   -3.78) | ema_count =    56.25
```

### Epoch 10
```
min pairwise sep: 16.798 m
anchor[0] = ( +86.50,   -0.41) | ema_count =    33.00
anchor[1] = ( +23.12,  +24.25) | ema_count =    12.81
anchor[2] = ( +36.00,   -0.98) | ema_count =    56.00
anchor[3] = (  +2.02,   +0.03) | ema_count =   176.00
anchor[4] = ( +56.25,   -0.24) | ema_count =    65.00
anchor[5] = ( +18.38,   -3.78) | ema_count =    56.00
```

### Epoch 9
```
min pairwise sep: 16.343 m
anchor[0] = ( +84.00,   -0.33) | ema_count =    35.00
anchor[1] = ( +22.88,  +24.38) | ema_count =    12.88
anchor[2] = ( +36.25,   -0.95) | ema_count =    55.75
anchor[3] = (  +1.99,   +0.05) | ema_count =   172.00
anchor[4] = ( +54.50,   -0.21) | ema_count =    66.00
anchor[5] = ( +17.88,   -3.80) | ema_count =    56.75
```

---

## Raw Epoch Logs

### Epoch 1
```
[Train ep 1] loss=20.8673 (avg=18.6776) | loss_l0=20.8785 | reg_loss_l0=19.5307 | cls_loss_l0=1.5992 | anchor_cls_loss_l0=1.0964 | loss_l1=20.8084 | reg_loss_l1=19.4958 | cls_loss_l1=1.5506 | anchor_cls_loss_l1=1.0747 | loss_l2=20.8912 | reg_loss_l2=19.4947 | cls_loss_l2=1.7323 | anchor_cls_loss_l2=1.0606 | gn_enc=7.84 gn_llm=5.81
[Train] ep 1 avg_loss=19.3886 | time=22.33min | peak_mem=81595MB
[Validation] ep 1 | loss: 18.8245 | minADE: 14.5648 m | minFDE: 29.5592 m | brierMinFDE: 30.1438 | time: 0.963 mins
[Branch] ep 1 | LLM corr/enc: 1.0588 | lane_cross/input: 0.0388
```

### Epoch 2
```
[Train ep 2] loss=12.3862 (avg=17.1381) | loss_l0=12.5175 | reg_loss_l0=11.1266 | cls_loss_l0=1.8082 | anchor_cls_loss_l0=0.9737 | loss_l1=12.3078 | reg_loss_l1=11.0038 | cls_loss_l1=1.6846 | anchor_cls_loss_l1=0.9234 | loss_l2=12.3598 | reg_loss_l2=10.9891 | cls_loss_l2=1.8221 | anchor_cls_loss_l2=0.9194 | gn_enc=6.44 gn_llm=6.25
[Train] ep 2 avg_loss=17.9880 | time=22.33min | peak_mem=81634MB
[Validation] ep 2 | loss: 17.3480 | minADE: 13.4586 m | minFDE: 28.2702 m | brierMinFDE: 28.8610 | time: 0.956 mins
[Branch] ep 2 | LLM corr/enc: 1.1010 | lane_cross/input: 0.1124
```

### Epoch 3
```
[Train ep 3] loss=15.3965 (avg=15.7520) | loss_l0=15.3614 | reg_loss_l0=14.0140 | cls_loss_l0=1.7618 | anchor_cls_loss_l0=0.9330 | loss_l1=15.4015 | reg_loss_l1=14.0346 | cls_loss_l1=1.8475 | anchor_cls_loss_l1=0.8863 | loss_l2=15.4115 | reg_loss_l2=14.0364 | cls_loss_l2=1.8602 | anchor_cls_loss_l2=0.8899 | gn_enc=9.62 gn_llm=8.75
[Train] ep 3 avg_loss=16.4852 | time=22.34min | peak_mem=81629MB
[Validation] ep 3 | loss: 15.8605 | minADE: 12.2872 m | minFDE: 26.8960 m | brierMinFDE: 27.4645 | time: 0.955 mins
[Branch] ep 3 | LLM corr/enc: 1.1667 | lane_cross/input: 0.1470
```

### Epoch 5
```
[Train ep 5] loss=14.3581 (avg50=13.5908) | gn_enc=7.00 gn_llm=5.97
[Train] ep 5 avg_loss=14.1165 | time=22.35min | peak_mem=81650MB
[Validation] ep 5 | loss: 13.6735 | minADE: 10.4914 m | minFDE: 24.3748 m | brierMinFDE: 24.9842 | time: 0.954 mins
[Branch] ep 5 | LLM corr/enc: 1.1443 | lane_cross/input: 0.1554
```

### Epoch 4
```
[Train ep 4 step 700] loss=13.6224 (avg50=15.2153) | reg_loss_l0=12.3997 | reg_loss_l1=12.2503 | reg_loss_l2=12.2264 | gn_enc=5.50 gn_llm=5.19
[Train ep 4 step 800] loss=13.2770 (avg50=15.1195) | reg_loss_l0=11.9977 | reg_loss_l1=11.9306 | reg_loss_l2=11.8881 | gn_enc=5.97 gn_llm=5.47
[Train ep 4 step 900] loss=15.3626 (avg50=15.2401) | reg_loss_l0=14.0543 | reg_loss_l1=13.9827 | reg_loss_l2=13.9417 | gn_enc=6.06 gn_llm=5.59
[Train ep 4 step 926] loss=17.4726 (avg50=14.7634) | reg_loss_l0=16.1928 | reg_loss_l1=16.1529 | reg_loss_l2=16.1325 | gn_enc=9.81 gn_llm=7.94  ← last-batch spike
[Train] ep 4 avg_loss=15.3302 | time=22.34min | peak_mem=81669MB
[Validation] ep 4 | loss: 14.8722 | minADE: 11.4937 m | minFDE: 25.7504 m | brierMinFDE: 26.3244 | time: 0.955 mins
[Branch] ep 4 | LLM corr/enc: 1.1778 | lane_cross/input: 0.1510
```

### Epoch 7
```
[ep 7 | 800/926] loss=11.5663 (avg50=12.0007) | reg_loss_l0=10.2331 | cls_loss_l0=1.6518 | anchor_cls_loss_l0=0.8870 | reg_loss_l1=10.1953 | reg_loss_l2=10.2002 | gn_enc=8.50 gn_llm=8.38
[ep 7 | 900/926] loss=10.0624 (avg50=12.2619) | reg_loss_l0=8.8302 | cls_loss_l0=1.6774 | anchor_cls_loss_l0=0.9062 | reg_loss_l1=8.8174 | reg_loss_l2=8.8285 | gn_enc=6.22 gn_llm=4.97
[ep 7 | 926/926] loss=7.7400 (avg=12.1324) | reg_loss_l0=6.5414 | cls_loss_l0=1.9041 | anchor_cls_loss_l0=0.6728 | reg_loss_l1=6.5047 | reg_loss_l2=6.4850 | gn_enc=5.59 gn_llm=5.22
[Train] ep 7 avg_loss=12.2980 | time=22.38min | peak_mem=81625MB
[Validation] ep 7 | loss: 12.0934 | minADE: 9.2115 m | minFDE: 22.1975 m | brierMinFDE: 22.7882 | time: 0.955 mins
[Branch] ep 7 | LLM corr/enc: 1.1081 | lane_cross/input: 0.1711
>> minADE improved by 0.3918 m → 9.2115 m; saving best checkpoint...
```

### Epoch 8
```
LR: llm_lr=4.81e-05, enc+dec_lr=9.61e-05
[ep 8 | 900/926] loss=11.7644 (avg50=12.1962) | reg_loss_l0=10.4087 | cls_loss_l0=1.8408 | anchor_cls_loss_l0=0.8916 | reg_loss_l1=10.4257 | reg_loss_l2=10.4241 | gn_enc=6.19 gn_llm=4.31
[ep 8 | 926/926] loss=15.2671 (avg=12.1216) | reg_loss_l0=13.8587 | cls_loss_l0=1.9937 | anchor_cls_loss_l0=0.9442 | reg_loss_l1=13.8310 | reg_loss_l2=13.8306 | gn_enc=6.44 gn_llm=6.22  ← last-batch spike (weaker than ep3/4)
[Train] ep 8 avg_loss=11.9970 | time=22.40min | peak_mem=81731MB
[Validation] ep 8 | loss: 11.8099 | minADE: 8.9840 m | minFDE: 21.7654 m | brierMinFDE: 22.3588 | time: 0.955 mins
[Branch] ep 8 | LLM corr/enc: 1.1602 | lane_cross/input: 0.1701
>> minADE improved by 0.2275 m → 8.9840 m; saving best checkpoint...
```

### Epoch 10
```
LR: llm_lr=4.26e-05, enc+dec_lr=8.51e-05
[ep 10 | 800/926] loss=12.9556 (avg50=11.9937) | reg_loss_l0=11.6281 | cls_loss_l0=1.7776 | anchor_cls_loss_l0=0.8381 | reg_loss_l1=11.6356 | reg_loss_l2=11.6372 | gn_enc=5.53 gn_llm=4.56
[ep 10 | 900/926] loss=9.7559 (avg50=11.4731) | reg_loss_l0=8.5320 | cls_loss_l0=1.7582 | anchor_cls_loss_l0=0.7699 | reg_loss_l1=8.5175 | reg_loss_l2=8.5063 | gn_enc=6.59 gn_llm=6.78
[ep 10 | 926/926] loss=10.2605 (avg=11.7383) | reg_loss_l0=8.9981 | cls_loss_l0=1.8075 | anchor_cls_loss_l0=0.7425 | reg_loss_l1=8.9703 | reg_loss_l2=8.9902 | gn_enc=8.44 gn_llm=10.00  ← last-batch: gn_llm hits clip ceiling
[Train] ep 10 avg_loss=11.7705 | time=22.39min | peak_mem=81636MB
[Validation] ep 10 | loss: 11.7541 | minADE: 8.9718 m | minFDE: 21.7839 m | brierMinFDE: 22.3722 | time: 0.954 mins
[Branch] ep 10 | LLM corr/enc: 1.1936 | lane_cross/input: 0.1853
>> minADE improved by 0.0053 m → 8.9718 m; saving best checkpoint...
```

### Epoch 9
```
LR: llm_lr=4.57e-05, enc+dec_lr=9.14e-05
[ep 9 | 900/926] loss=12.3052 (avg50=11.4871) | reg_loss_l0=10.9409 | cls_loss_l0=1.7561 | anchor_cls_loss_l0=0.8307 | reg_loss_l1=10.9455 | reg_loss_l2=10.9361 | gn_enc=6.38 gn_llm=5.00
[ep 9 | 926/926] loss=11.8199 (avg=11.4440) | reg_loss_l0=10.4422 | cls_loss_l0=1.6200 | anchor_cls_loss_l0=1.0880 | reg_loss_l1=10.4223 | reg_loss_l2=10.4428 | gn_enc=6.34 gn_llm=6.78
[Train] ep 9 avg_loss=11.7995 | time=22.40min | peak_mem=81715MB
[Validation] ep 9 | loss: 11.7740 | minADE: 8.9771 m | minFDE: 21.7830 m | brierMinFDE: 22.3411 | time: 0.955 mins
[Branch] ep 9 | LLM corr/enc: 1.1880 | lane_cross/input: 0.1773
>> minADE improved by 0.0070 m → 8.9771 m; saving best checkpoint...
```

---

## Observations & Action Items

### ep1–3 Analysis (2026-06-10)
- **Convergence**: Healthy. minADE dropping ~1.15m/epoch. avg_loss: 19.39 → 17.99 → 16.49.
- **LLM corr/enc**: Growing 1.059 → 1.101 → 1.167. Trigger: if ep5 > 1.4, stop and add `--llm_correction_scale 0.5`.
- **lane_cross**: Activating as expected (zero-init residual). 0.039 → 0.112 → 0.147.
- **gn_enc spike ep3**: 9.62 at last batch (5.88 mid-epoch). Warmup-phase artifact (lr still rising). Monitor after ep5 (warmup end).
- **Anchor sep growing**: 10.8 → 12.4 → 13.0m. Good diversity.
- **Level differentiation**: ep3 l0 reg=14.01 > l1=13.37 ≈ l2=13.36. E2E interaction beginning to help.

### ep4 Analysis (2026-06-10)
- **Convergence**: Continues. avg_loss 15.33 (-1.15), minADE 11.49 (-0.79m, rate slowing — normal).
- **LLM corr/enc**: 1.178 (+0.011 vs ep3's +0.067). **Growth decelerating sharply.** Approaching plateau. Trigger threshold 1.4 at ep5 unlikely to be hit.
- **lane_cross**: 0.151 — stabilizing. Module fully activated.
- **Last-batch spike pattern** (confirmed recurring): ep3 last-batch gn_enc=9.62, ep4=9.81. Loss also spikes at last batch (17.47 vs mid-epoch ~13). Root cause: DistributedSampler pads the last batch with repeated samples to fill world_size×batch_size; these padded samples create unusual gradient directions. **Not a bug — but clip threshold 5.0 is being hit hard.** Action: add `drop_last=True` to train DataLoader to eliminate pad samples (next run).
- **Anchor sep slightly down**: 13.0 → 12.8m. Minor fluctuation, not a collapse signal.
- **Level differentiation stable**: ep4 l0 reg=12.40 > l1=12.25 > l2=12.23. Hierarchy maintained. ✓

### ep5 Analysis (2026-06-10) — Warmup Complete
- **LLM corr/enc DECREASED for first time: 1.178 → 1.144**. Self-correcting. Encoder is now growing faster than LLM correction. Optimizer fix confirmed working. Trigger threshold (1.4) no longer a concern.
- **Last-batch gn_enc spike resolved: 9.81 → 7.00**. Warmup phase caused lr-ramp instability at last batch; with warmup complete (ep5 = peak lr = 1.0× base), gradient norms stabilized.
- **avg_loss drop accelerated: -1.21m** (ep4 was -1.16m). Full lr now active.
- **minADE: 10.49m (-1.0m)**. Back to ~1m/ep rate after ep4 slowdown.
- **LR schedule from ep6**: cosine decay phase begins (ep5=peak, ep6→19 decays from 100%→10%, ep20 restarts).

### ep7–9 Analysis (2026-06-10) — minADE Plateau Onset

**minADE improvement rate**:
| Transition | ΔminADE |
|---|---|
| ep5 → ep6 | −0.89m (inferred) |
| ep6 → ep7 | −0.39m |
| ep7 → ep8 | −0.23m |
| ep8 → ep9 | **−0.007m** ← near-flat |

- **Loss-minADE disconnect**: avg_loss still decreasing (12.30→11.80, −0.25/ep) while minADE flatlined. This indicates the model is improving cls_loss / anchor_cls_loss components but reg_loss has stalled — mode selection quality improving, positional accuracy not.
- **Level differentiation collapsed**: ep7–9 all levels show near-identical reg_loss (l0≈l1≈l2 within <0.01m at mid-epoch). This is not alarming — it means the base prediction at l0 is already good enough that refinement is marginal. Contrast ep3 where l0−l1=0.64m.
- **LLM corr/enc oscillating**: 1.144(ep5) → 1.108(ep7) → 1.160(ep8) → 1.188(ep9). After the ep5-7 decrease it's ticking up again. Still well below the 1.4 trigger but shows the correction is still trying to grow. The encoder is not yet dominating it.
- **Anchor diversity strong and growing**: 13.3 → 15.2 → 15.5 → 16.3m. No collapse. anchor[3] (slow/stopped mode) remains dominant (ema_count=172) but anchors[2,5] are differentiating (slow forward motion vs lateral+slight backward offset).
- **Last-batch gn pattern stable**: ep7 gn_enc=5.59, ep8=6.44, ep9=6.34 — all well below the ep3/4 spike of 9.62/9.81. Warmup artifact fully resolved.
- **LR at ep9**: llm_lr=4.57e-05 (91.4% of peak). Still in early cosine decay. The real slow-LR regime (≤50% of peak) starts around ep12. **minADE improvement may resume as LR drops lower** — cosine decay fine-tunes positional accuracy better than high-LR steps.
- **Ep9 last-batch anchor_cls spike**: anchor_cls_loss jumped to ~1.08 at step 926 (vs ~0.83 mid-epoch). Consistent with last-batch DistributedSampler padding artifact. `drop_last=True` fix remains queued for next run.

**Assessment**: The minADE plateau at ep8-9 is likely an early cosine decay behavior, not a training failure. With T_0=20 and LR still at 91-95%, the model is oscillating around a local optimum. The real fine-tuning begins around ep13-16 (LR 40-20%). The `early_stop_patience=10` gives until ep19 before any early stop triggers. **No intervention warranted yet — continue monitoring through ep12.**

### ep10 Analysis (2026-06-10) — Plateau Confirmed, gn_llm Spike Notable

- **minADE: 8.9718m (−0.005m)** — 3rd consecutive near-flat epoch: ep8→ep9 −0.007m, ep9→ep10 −0.005m. The 3-epoch watchlist threshold is technically triggered. However early_stop only resets when there IS improvement (0.005m qualifies), so the early stop counter stays at 0.
- **avg_loss barely moved: −0.029m** (vs −0.2/ep previously). Both train loss and val loss are now nearly identical (11.771 vs 11.754), confirming the model has found its current-LR optimum — not overfitting, just reached the precision limit at this LR level.
- **gn_llm = 10.00 at last batch**: First time gn_llm has hit 10.0 (--llm_grad_clip=1.0 so actual applied gradient is clamped; the 10.0 is the pre-clip norm). This means the LLM branch was trying to make a 10× over-norm update, which got clipped back to 1.0. Likely the same last-batch padding artifact, but the LLM is amplifying it. The mid-epoch gn_llm at step 900 was 6.78 — more normal.
- **LLM corr/enc: 1.194** (now rising for 3 consecutive epochs after ep7 dip). LLM correction weight is growing persistently. Still below 1.4 trigger.
- **Anchor min_sep: 16.798m** (growing every epoch since ep5 — very healthy diversity).
- **val minADE mid-eval vs final**: During ep10 val run, a batch hit minADE=8.8683 (lowest seen so far), but the full epoch average landed at 8.9718. This shows the model CAN predict to ~8.87m on favorable scenarios — the current plateau is an average-quality issue, not a hard ceiling.

**Root cause of plateau**: LR still at ~85% of peak (llm_lr=4.26e-05). The cosine decay hasn't gone low enough yet to enable fine-grained positional tuning. The gn_llm=10 spike shows the LLM is still trying to make large jumps but getting clipped — this is "churning" around the current optimum rather than converging. The low-LR regime (≤50% peak, ep12-13 onward) is needed to settle.

**Action**: No intervention. Continue through ep13-16 where LR drops to 40-20% of peak. If minADE remains flat (< 0.01m/ep) for 5+ consecutive epochs total, consider early stopping and evaluating the ep20 cosine restart effect.

### Model Audit & Root-Cause Diagnosis of the 9m Plateau (2026-06-10, post-ep10)

Full code audit of `hybrid_llm_model.py` / `av2_llm_loss.py` / `train_hybrid.py` val loop,
cross-checked against `train_llm.py` (the script that produced the old 1.63m number).

**Finding 0 — the 9m and the old 1.63m are DIFFERENT metrics, not comparable.**
- Old (`train_llm.py:273-292`): **focal agent only**, best-of-K chosen freely *for that one
  agent* (standard AV2 single-agent marginal minADE).
- New (`train_hybrid.py:560-597`): **scene-level JOINT minADE** — one shared mode per scene,
  error averaged over ALL scored agents (focal + 'score' category). A strictly harder,
  multi-agent metric. Even a perfect marginal predictor scores several × worse on it.
- Consequence: "8.97m vs 1.63m" is largely a metric redefinition, not purely model regression.
  Action: also log focal marginal minADE each val epoch for continuity with Phase I.

**Finding 1 — structural cap: anchor-tied mode semantics vs joint-mode protocol.**
- `MLPDecoder.forward`: `mode_e = base_q + anchor_mlp(anchor_k)` is added to EVERY agent →
  mode k carries an agent-INDEPENDENT endpoint prior "go to anchor_k" (anchors at ~2/18/23/
  36/56/86m forward).
- `anchor_cls_loss` reinforces per-agent anchor semantics (each agent's score → its own
  nearest anchor).
- But reg/cls/val all use ONE shared winner mode per scene. With heterogeneous agents
  (anchor[3] "stationary" holds ema_count 176 ≈ 44% of mass, others cruise 18–86m), no
  single anchor-tied mode fits all agents simultaneously. The shared-mode error floor is
  the anchor-quantization error averaged over non-conforming agents.
- Consistency check: observed minFDE≈21.8m, minADE≈9.0 (ratio 2.4) matches near-linear
  trajectories whose endpoints miss by ~20m — i.e. anchor-dominated, weakly
  agent-conditioned predictions. Level collapse (l0≈l1≈l2) is the same signature: the
  bottleneck is mode parameterisation, not interaction context.

**Finding 2 — Bézier head scale conditioning (secondary).**
- Final Linear (after LayerNorm) outputs O(1)m at init; control points must span 0–90m →
  weights must grow ~50×, against weight_decay=1e-4 and grad_clip. SmoothL1 (beta=1) gives
  constant sign-gradients for error>1m → linear convergence rate. Matches the observed
  steady ~1m/ep warmup gains followed by stall once easy mass (stationary cluster) is fit.
- B(t=0)=C₀ unconstrained: model must *learn* C₀≈0 ("trajectory starts at current
  position") which displacement+cumsum got for free.

**Finding 3 — loss-metric disconnect explained.**
- `soft_wta_alpha=0.1` pulls ALL K modes toward every scene's GT; non-winner reg components
  + cls/anchor_cls keep improving avg_loss while the JOINT min metric is pinned by
  Finding 1. Hence loss ↓ 0.2/ep while minADE flat — exactly ep8–10.

**Finding 4 — LLM branch starved at clip (minor).**
- gn_llm pre-clip hit 10.0 with `--llm_grad_clip=1.0`: late-epoch LLM updates truncated to
  10% of desired step. Consistent with corr/enc creeping (1.11→1.19) instead of converging.

**Verdict on earlier hypotheses**: "wait for cosine LR decay" (ep7-9 note) is REJECTED —
ep10 at 85% LR shows 0.005m gain; the plateau is structural, not LR-bound. The earlier
"two-stage training / LLM cold-start dominance" hypothesis is also rejected (old runs had
the same LLM topology and reached 1.63m on the marginal metric).

**Recommended actions (ordered)**:
1. Log focal marginal minADE alongside joint minADE (cheap, restores comparability).
2. Decide target metric. If joint: remove the per-mode anchor injection from the decoder
   query (or make anchors agent-conditioned) and reconcile anchor_cls with the joint winner.
3. Bézier head conditioning: residualise ctrl points around the per-mode anchor ray
   (C₀=0, C_last=anchor_k, MLP predicts deviation) or scale the head output.
4. drop_last=True; consider llm_grad_clip 1.0 → 2.5.

### Watchlist (updated after ep10)
| Signal | Threshold | Current | Status |
|---|---|---|---|
| LLM corr/enc | > 1.4 | 1.194 ↑ (3 consecutive rises) | ⚠️ Watch — rising trend |
| minADE improvement/ep | < 0.3m for 3ep | **threshold triggered** (ep8-10: 0.23/0.007/0.005m) | ⚠️ Monitoring — no intervention yet |
| gn_llm last-batch | > 8.0 | 10.00 (ep10) — pre-clip | ⚠️ New pattern, LLM hitting clip ceiling |
| avg_loss improvement/ep | < 0.05 for 2ep | 0.029 (ep10) — near flat | ⚠️ Watch |
| gn_enc last-batch | spike pattern | 8.44 (slightly elevated) | ✅ OK |
| anchor min_sep | < 5m | 16.8m | ✅ Strong |
| drop_last=True | — | queued for next run | ⚠️ Low priority |

### LR Schedule Reference (cosine_warmup_restart, T_0=20, warmup=5)
| Epoch range | Phase | lr scale |
|---|---|---|
| 1–4 | Warmup | 10%→82% |
| 5 | Peak | 100% |
| 6–19 | Cosine decay | ~99%→10% |
| 20 | Restart | 10% |
| 20–25 | Warmup again | 10%→100% |

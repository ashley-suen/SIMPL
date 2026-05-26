# Plan A — Per-Agent Prompt Inspection

> Each agent receives its own independent prompt: **shared scene header + that agent's own trajectory**.

# Split: train  (`data_av2/features/train`, 20 scenes total)

---

## Scene 1/5 — `2a73c649-306b-4e1a-a045-15278cc69b2d` (dataset index 3)

**Agents written:** 6  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1275/2000 |
| 1 | NBR1 | 5 | True | 1352/2000 |
| 2 | NBR2 | 20 | True | 195/2000 |
| 3 | NBR3 | 1 | True | 184/2000 |
| 4 | NBR4 | 2 | True | 184/2000 |
| 5 | NBR5 | 4 | True | 184/2000 |

### Agent 0 — FOCAL (traj_id=0, 1275 tokens)

```text
Scenario '2a73c649-306b-4e1a-a045-15278cc69b2d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Right (Solid Line)

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-7.2,0.1)|V:1.6m/s -> t=1:(-7.1,0.1)|V:1.6m/s -> t=2:(-7.0,0.1)|V:1.5m/s -> t=3:(-6.9,0.1)|V:1.5m/s -> t=4:(-6.8,0.1)|V:1.5m/s -> t=5:(-6.7,0.1)|V:1.5m/s -> t=6:(-6.5,0.1)|V:1.5m/s -> t=7:(-6.4,0.1)|V:1.5m/s -> t=8:(-6.2,0.1)|V:1.5m/s -> t=9:(-6.0,0.1)|V:1.4m/s -> t=10:(-5.9,0.1)|V:1.4m/s -> t=11:(-5.7,0.1)|V:1.4m/s -> t=12:(-5.6,0.1)|V:1.4m/s -> t=13:(-5.5,0.1)|V:1.4m/s -> t=14:(-5.3,0.1)|V:1.4m/s -> t=15:(-5.2,0.1)|V:1.4m/s -> t=16:(-5.0,0.1)|V:1.4m/s -> t=17:(-4.9,0.1)|V:1.4m/s -> t=18:(-4.7,0.0)|V:1.4m/s -> t=19:(-4.6,0.0)|V:1.4m/s -> t=20:(-4.4,0.0)|V:1.4m/s -> t=21:(-4.3,0.0)|V:1.4m/s -> t=22:(-4.2,-0.0)|V:1.4m/s -> t=23:(-4.0,-0.0)|V:1.5m/s -> t=24:(-3.9,-0.0)|V:1.5m/s -> t=25:(-3.7,-0.0)|V:1.5m/s -> t=26:(-3.6,-0.1)|V:1.5m/s -> t=27:(-3.5,-0.1)|V:1.5m/s -> t=28:(-3.3,-0.1)|V:1.5m/s -> t=29:(-3.2,-0.1)|V:1.5m/s -> t=30:(-3.0,-0.1)|V:1.5m/s -> t=31:(-2.9,-0.1)|V:1.5m/s -> t=32:(-2.7,-0.1)|V:1.5m/s -> t=33:(-2.6,-0.1)|V:1.4m/s -> t=34:(-2.4,-0.1)|V:1.4m/s -> t=35:(-2.2,-0.0)|V:1.4m/s -> t=36:(-2.1,-0.0)|V:1.4m/s -> t=37:(-1.9,-0.0)|V:1.5m/s -> t=38:(-1.8,-0.0)|V:1.5m/s -> t=39:(-1.6,-0.0)|V:1.5m/s -> t=40:(-1.4,0.0)|V:1.6m/s -> t=41:(-1.3,0.0)|V:1.6m/s -> t=42:(-1.1,0.0)|V:1.6m/s -> t=43:(-1.0,0.0)|V:1.6m/s -> t=44:(-0.8,0.0)|V:1.6m/s -> t=45:(-0.6,0.0)|V:1.6m/s -> t=46:(-0.5,0.0)|V:1.6m/s -> t=47:(-0.3,0.0)|V:1.6m/s -> t=48:(-0.1,0.0)|V:1.6m/s -> t=49:(0.0,0.0)|V:1.6m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=5, 1352 tokens)

```text
Scenario '2a73c649-306b-4e1a-a045-15278cc69b2d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Right (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:10) [pos(t=49): far-right-front | motion: faster than ego | interaction: low influence]: t=0:(49.0,-17.0)|V:9.2m/s -> t=1:(48.6,-16.9)|V:9.2m/s -> t=2:(48.1,-16.8)|V:9.2m/s -> t=3:(47.4,-16.8)|V:9.3m/s -> t=4:(46.7,-16.7)|V:9.3m/s -> t=5:(45.9,-16.6)|V:9.4m/s -> t=6:(45.0,-16.4)|V:9.4m/s -> t=7:(44.1,-16.3)|V:9.4m/s -> t=8:(43.1,-16.2)|V:9.4m/s -> t=9:(42.1,-16.0)|V:9.4m/s -> t=10:(41.1,-15.9)|V:9.4m/s -> t=11:(40.2,-15.8)|V:9.4m/s -> t=12:(39.3,-15.7)|V:9.4m/s -> t=13:(38.5,-15.6)|V:9.4m/s -> t=14:(37.6,-15.5)|V:9.3m/s -> t=15:(36.7,-15.3)|V:9.2m/s -> t=16:(35.9,-15.2)|V:9.1m/s -> t=17:(35.0,-15.1)|V:9.0m/s -> t=18:(34.2,-15.0)|V:8.8m/s -> t=19:(33.4,-14.9)|V:8.6m/s -> t=20:(32.5,-14.8)|V:8.5m/s -> t=21:(31.7,-14.7)|V:8.3m/s -> t=22:(30.9,-14.6)|V:8.2m/s -> t=23:(30.1,-14.6)|V:8.1m/s -> t=24:(29.4,-14.5)|V:8.0m/s -> t=25:(28.6,-14.4)|V:7.9m/s -> t=26:(27.8,-14.3)|V:7.7m/s -> t=27:(27.1,-14.3)|V:7.6m/s -> t=28:(26.4,-14.2)|V:7.5m/s -> t=29:(25.7,-14.2)|V:7.4m/s -> t=30:(25.0,-14.1)|V:7.2m/s -> t=31:(24.3,-14.1)|V:7.1m/s -> t=32:(23.6,-14.1)|V:7.1m/s -> t=33:(22.9,-14.1)|V:7.0m/s -> t=34:(22.2,-14.0)|V:6.9m/s -> t=35:(21.6,-14.0)|V:6.8m/s -> t=36:(20.9,-14.0)|V:6.7m/s -> t=37:(20.3,-14.0)|V:6.7m/s -> t=38:(19.6,-14.1)|V:6.6m/s -> t=39:(19.0,-14.1)|V:6.6m/s -> t=40:(18.4,-14.1)|V:6.5m/s -> t=41:(17.8,-14.2)|V:6.5m/s -> t=42:(17.2,-14.2)|V:6.4m/s -> t=43:(16.6,-14.3)|V:6.3m/s -> t=44:(16.0,-14.3)|V:6.3m/s -> t=45:(15.5,-14.4)|V:6.1m/s -> t=46:(15.0,-14.4)|V:6.0m/s -> t=47:(14.5,-14.5)|V:5.6m/s -> t=48:(14.0,-14.5)|V:5.4m/s -> t=49:(13.5,-14.6)|V:5.3m/s
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=20, 195 tokens)

```text
Scenario '2a73c649-306b-4e1a-a045-15278cc69b2d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Right (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:8) (Entered scene at t=42) [pos(t=49): far-left-rear | motion: stationary | interaction: low influence (parked)]: t=42~49 Stationary at (-5.5, 9.2)
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=1, 184 tokens)

```text
Scenario '2a73c649-306b-4e1a-a045-15278cc69b2d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Right (Solid Line)

[AGENT HISTORY]
- Autonomous Vehicle (Impact:7) [pos(t=49): far-right-front | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (9.6, -9.0)
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=2, 184 tokens)

```text
Scenario '2a73c649-306b-4e1a-a045-15278cc69b2d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Right (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:7) [pos(t=49): far-right-front | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (9.3, -11.8)
[PREDICT]:
```

### Agent 5 — NBR5 (traj_id=4, 184 tokens)

```text
Scenario '2a73c649-306b-4e1a-a045-15278cc69b2d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Right (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:7) [pos(t=49): far-right-front | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (14.7, -6.7)
[PREDICT]:
```

---

## Scene 2/5 — `0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0` (dataset index 0)

**Agents written:** 6  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1290/2000 |
| 1 | NBR1 | 9 | True | 179/2000 |
| 2 | NBR2 | 18 | True | 190/2000 |
| 3 | NBR3 | 24 | True | 191/2000 |
| 4 | NBR4 | 2 | True | 181/2000 |
| 5 | NBR5 | 1 | True | 182/2000 |

### Agent 0 — FOCAL (traj_id=0, 1290 tokens)

```text
Scenario '0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0' in miami. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-24.4,2.6)|V:9.0m/s -> t=1:(-24.1,2.6)|V:9.0m/s -> t=2:(-23.6,2.6)|V:8.9m/s -> t=3:(-22.9,2.5)|V:8.8m/s -> t=4:(-22.1,2.5)|V:8.8m/s -> t=5:(-21.2,2.4)|V:8.7m/s -> t=6:(-20.2,2.3)|V:8.6m/s -> t=7:(-19.1,2.1)|V:8.5m/s -> t=8:(-18.0,2.0)|V:8.5m/s -> t=9:(-16.9,1.8)|V:8.4m/s -> t=10:(-15.8,1.7)|V:8.3m/s -> t=11:(-14.8,1.5)|V:8.2m/s -> t=12:(-13.8,1.3)|V:8.1m/s -> t=13:(-12.9,1.2)|V:8.1m/s -> t=14:(-12.1,1.0)|V:8.1m/s -> t=15:(-11.3,0.8)|V:8.0m/s -> t=16:(-10.6,0.7)|V:7.8m/s -> t=17:(-10.0,0.6)|V:7.4m/s -> t=18:(-9.4,0.6)|V:7.0m/s -> t=19:(-8.9,0.5)|V:6.7m/s -> t=20:(-8.2,0.5)|V:6.5m/s -> t=21:(-7.6,0.5)|V:6.2m/s -> t=22:(-7.0,0.4)|V:6.0m/s -> t=23:(-6.4,0.4)|V:5.8m/s -> t=24:(-5.9,0.3)|V:5.5m/s -> t=25:(-5.3,0.3)|V:5.3m/s -> t=26:(-4.8,0.3)|V:5.0m/s -> t=27:(-4.3,0.3)|V:4.9m/s -> t=28:(-3.9,0.2)|V:4.7m/s -> t=29:(-3.4,0.2)|V:4.5m/s -> t=30:(-3.0,0.2)|V:4.3m/s -> t=31:(-2.7,0.2)|V:4.1m/s -> t=32:(-2.3,0.2)|V:3.8m/s -> t=33:(-2.0,0.2)|V:3.6m/s -> t=34:(-1.7,0.1)|V:3.3m/s -> t=35:(-1.5,0.1)|V:3.0m/s -> t=36:(-1.2,0.1)|V:2.6m/s -> t=37:(-1.0,0.1)|V:2.4m/s -> t=38:(-0.9,0.1)|V:2.1m/s -> t=39:(-0.7,0.1)|V:1.9m/s -> t=40:(-0.6,0.1)|V:1.6m/s -> t=41:(-0.4,0.1)|V:1.4m/s -> t=42:(-0.3,0.1)|V:1.1m/s -> t=43:(-0.2,0.1)|V:0.9m/s -> t=44:(-0.2,0.0)|V:0.8m/s -> t=45:(-0.1,0.0)|V:0.6m/s -> t=46:(-0.1,0.0)|V:0.5m/s -> t=47:(-0.0,0.0)|V:0.3m/s -> t=48:(-0.0,0.0)|V:0.2m/s -> t=49:(0.0,0.0)|V:0.1m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=9, 179 tokens)

```text
Scenario '0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0' in miami. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:9) [pos(t=49): right-rear | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (-6.4, -2.6)
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=18, 190 tokens)

```text
Scenario '0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0' in miami. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:9) (Entered scene at t=1) [pos(t=49): far-left-rear | motion: stationary | interaction: low influence (parked)]: t=1~49 Stationary at (-4.7, 5.5)
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=24, 191 tokens)

```text
Scenario '0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0' in miami. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:7) (Entered scene at t=1) [pos(t=49): far-left-rear | motion: stationary | interaction: low influence (parked)]: t=1~49 Stationary at (-11.4, 5.7)
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=2, 181 tokens)

```text
Scenario '0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0' in miami. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:7) [pos(t=49): far-right-front | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (7.5, -11.9)
[PREDICT]:
```

### Agent 5 — NBR5 (traj_id=1, 182 tokens)

```text
Scenario '0925ba29-4dd0-43c7-9b5c-ae067c7fb7c0' in miami. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Autonomous Vehicle (Impact:7) [pos(t=49): far-right-front | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (14.5, -6.0)
[PREDICT]:
```

---

## Scene 3/5 — `6c7e15d8-5a1e-4e33-b64d-7fecfb076b60` (dataset index 8)

**Agents written:** 6  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1269/2000 |
| 1 | NBR1 | 2 | True | 1346/2000 |
| 2 | NBR2 | 6 | True | 174/2000 |
| 3 | NBR3 | 7 | True | 175/2000 |
| 4 | NBR4 | 37 | True | 187/2000 |
| 5 | NBR5 | 23 | True | 186/2000 |

### Agent 0 — FOCAL (traj_id=0, 1269 tokens)

```text
Scenario '6c7e15d8-5a1e-4e33-b64d-7fecfb076b60' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-8.3,1.5)|V:1.8m/s -> t=1:(-8.3,1.5)|V:1.8m/s -> t=2:(-8.2,1.4)|V:1.8m/s -> t=3:(-8.1,1.4)|V:1.8m/s -> t=4:(-8.0,1.3)|V:1.7m/s -> t=5:(-7.9,1.2)|V:1.7m/s -> t=6:(-7.7,1.1)|V:1.7m/s -> t=7:(-7.6,1.0)|V:1.7m/s -> t=8:(-7.5,0.9)|V:1.6m/s -> t=9:(-7.3,0.8)|V:1.6m/s -> t=10:(-7.2,0.6)|V:1.7m/s -> t=11:(-7.0,0.5)|V:1.7m/s -> t=12:(-6.9,0.4)|V:1.7m/s -> t=13:(-6.7,0.3)|V:1.7m/s -> t=14:(-6.6,0.2)|V:1.7m/s -> t=15:(-6.4,0.1)|V:1.7m/s -> t=16:(-6.3,0.0)|V:1.7m/s -> t=17:(-6.1,-0.0)|V:1.7m/s -> t=18:(-5.9,-0.1)|V:1.7m/s -> t=19:(-5.7,-0.2)|V:1.8m/s -> t=20:(-5.5,-0.2)|V:1.8m/s -> t=21:(-5.3,-0.2)|V:1.9m/s -> t=22:(-5.1,-0.2)|V:2.0m/s -> t=23:(-4.9,-0.2)|V:2.0m/s -> t=24:(-4.7,-0.2)|V:2.1m/s -> t=25:(-4.5,-0.2)|V:2.1m/s -> t=26:(-4.3,-0.2)|V:2.1m/s -> t=27:(-4.1,-0.2)|V:2.1m/s -> t=28:(-3.9,-0.1)|V:2.1m/s -> t=29:(-3.8,-0.1)|V:2.0m/s -> t=30:(-3.6,-0.1)|V:2.0m/s -> t=31:(-3.4,-0.1)|V:2.0m/s -> t=32:(-3.2,-0.1)|V:2.0m/s -> t=33:(-3.0,-0.0)|V:1.9m/s -> t=34:(-2.8,-0.0)|V:1.9m/s -> t=35:(-2.6,-0.0)|V:1.9m/s -> t=36:(-2.4,-0.0)|V:1.9m/s -> t=37:(-2.2,-0.0)|V:1.9m/s -> t=38:(-2.1,-0.0)|V:1.9m/s -> t=39:(-1.9,-0.0)|V:1.8m/s -> t=40:(-1.7,-0.0)|V:1.8m/s -> t=41:(-1.5,-0.0)|V:1.8m/s -> t=42:(-1.3,-0.0)|V:1.8m/s -> t=43:(-1.1,-0.0)|V:1.9m/s -> t=44:(-0.9,-0.0)|V:1.9m/s -> t=45:(-0.7,-0.0)|V:1.9m/s -> t=46:(-0.5,-0.0)|V:1.9m/s -> t=47:(-0.4,-0.0)|V:1.8m/s -> t=48:(-0.2,-0.0)|V:1.8m/s -> t=49:(0.0,0.0)|V:1.8m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=2, 1346 tokens)

```text
Scenario '6c7e15d8-5a1e-4e33-b64d-7fecfb076b60' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:19) [pos(t=49): far-right-rear | motion: faster than ego | interaction: low influence]: t=0:(-41.1,-0.5)|V:5.2m/s -> t=1:(-40.9,-0.5)|V:5.2m/s -> t=2:(-40.5,-0.6)|V:5.4m/s -> t=3:(-40.1,-0.6)|V:5.5m/s -> t=4:(-39.7,-0.7)|V:5.7m/s -> t=5:(-39.1,-0.8)|V:5.9m/s -> t=6:(-38.5,-0.9)|V:6.0m/s -> t=7:(-37.9,-1.0)|V:6.2m/s -> t=8:(-37.2,-1.2)|V:6.3m/s -> t=9:(-36.5,-1.3)|V:6.4m/s -> t=10:(-35.8,-1.4)|V:6.5m/s -> t=11:(-35.2,-1.6)|V:6.6m/s -> t=12:(-34.5,-1.7)|V:6.7m/s -> t=13:(-33.8,-1.8)|V:6.8m/s -> t=14:(-33.2,-1.9)|V:6.9m/s -> t=15:(-32.5,-2.1)|V:6.9m/s -> t=16:(-31.8,-2.2)|V:6.9m/s -> t=17:(-31.1,-2.3)|V:6.9m/s -> t=18:(-30.4,-2.5)|V:6.9m/s -> t=19:(-29.7,-2.6)|V:6.9m/s -> t=20:(-29.0,-2.8)|V:7.0m/s -> t=21:(-28.3,-2.9)|V:7.1m/s -> t=22:(-27.6,-3.1)|V:7.1m/s -> t=23:(-26.9,-3.2)|V:7.2m/s -> t=24:(-26.2,-3.4)|V:7.2m/s -> t=25:(-25.4,-3.5)|V:7.3m/s -> t=26:(-24.7,-3.7)|V:7.4m/s -> t=27:(-24.0,-3.8)|V:7.5m/s -> t=28:(-23.3,-4.0)|V:7.5m/s -> t=29:(-22.6,-4.1)|V:7.6m/s -> t=30:(-22.0,-4.2)|V:7.6m/s -> t=31:(-21.3,-4.3)|V:7.5m/s -> t=32:(-20.6,-4.4)|V:7.4m/s -> t=33:(-19.9,-4.5)|V:7.3m/s -> t=34:(-19.3,-4.6)|V:7.1m/s -> t=35:(-18.6,-4.7)|V:6.9m/s -> t=36:(-18.0,-4.8)|V:6.7m/s -> t=37:(-17.3,-4.9)|V:6.6m/s -> t=38:(-16.7,-5.0)|V:6.4m/s -> t=39:(-16.0,-5.2)|V:6.3m/s -> t=40:(-15.4,-5.3)|V:6.2m/s -> t=41:(-14.7,-5.4)|V:6.2m/s -> t=42:(-14.1,-5.5)|V:6.2m/s -> t=43:(-13.4,-5.6)|V:6.3m/s -> t=44:(-12.8,-5.7)|V:6.3m/s -> t=45:(-12.2,-5.8)|V:6.4m/s -> t=46:(-11.5,-5.9)|V:6.4m/s -> t=47:(-10.9,-6.0)|V:6.4m/s -> t=48:(-10.3,-6.1)|V:6.4m/s -> t=49:(-9.7,-6.2)|V:6.4m/s
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=6, 174 tokens)

```text
Scenario '6c7e15d8-5a1e-4e33-b64d-7fecfb076b60' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:10) [pos(t=49): follower | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (-0.5, -1.0)
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=7, 175 tokens)

```text
Scenario '6c7e15d8-5a1e-4e33-b64d-7fecfb076b60' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:9) [pos(t=49): right-front | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (0.4, -3.0)
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=37, 187 tokens)

```text
Scenario '6c7e15d8-5a1e-4e33-b64d-7fecfb076b60' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:9) (Entered scene at t=17) [pos(t=49): right-rear | motion: stationary | interaction: low influence (parked)]: t=17~49 Stationary at (-5.3, -2.8)
[PREDICT]:
```

### Agent 5 — NBR5 (traj_id=23, 186 tokens)

```text
Scenario '6c7e15d8-5a1e-4e33-b64d-7fecfb076b60' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:9) (Entered scene at t=4) [pos(t=49): far-right-front | motion: stationary | interaction: low influence (parked)]: t=4~49 Stationary at (2.0, -6.1)
[PREDICT]:
```

---

## Scene 4/5 — `650b08b9-2b6c-4de6-a1cb-8b3de3f235c7` (dataset index 7)

**Agents written:** 4  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1263/2000 |
| 1 | NBR1 | 1 | True | 1338/2000 |
| 2 | NBR2 | 6 | True | 181/2000 |
| 3 | NBR3 | 13 | True | 184/2000 |
| 4 | EMPTY | — | False | 1/2000 |
| 5 | EMPTY | — | False | 1/2000 |

### Agent 0 — FOCAL (traj_id=0, 1263 tokens)

```text
Scenario '650b08b9-2b6c-4de6-a1cb-8b3de3f235c7' in palo-alto. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-3.7,0.0)|V:2.1m/s -> t=1:(-3.6,0.0)|V:2.1m/s -> t=2:(-3.5,0.0)|V:2.1m/s -> t=3:(-3.3,-0.0)|V:2.1m/s -> t=4:(-3.2,-0.0)|V:2.0m/s -> t=5:(-3.0,-0.0)|V:2.0m/s -> t=6:(-2.8,-0.0)|V:1.9m/s -> t=7:(-2.6,-0.0)|V:1.8m/s -> t=8:(-2.4,-0.0)|V:1.7m/s -> t=9:(-2.3,-0.0)|V:1.6m/s -> t=10:(-2.1,-0.0)|V:1.6m/s -> t=11:(-2.0,-0.0)|V:1.5m/s -> t=12:(-1.9,-0.0)|V:1.4m/s -> t=13:(-1.8,-0.0)|V:1.4m/s -> t=14:(-1.7,-0.0)|V:1.3m/s -> t=15:(-1.5,-0.0)|V:1.2m/s -> t=16:(-1.4,-0.0)|V:1.0m/s -> t=17:(-1.3,-0.0)|V:0.8m/s -> t=18:(-1.2,-0.0)|V:0.7m/s -> t=19:(-1.2,-0.0)|V:0.6m/s -> t=20:(-1.1,-0.0)|V:0.6m/s -> t=21:(-1.1,-0.0)|V:0.6m/s -> t=22:(-1.0,-0.0)|V:0.6m/s -> t=23:(-0.9,-0.0)|V:0.5m/s -> t=24:(-0.9,-0.0)|V:0.4m/s -> t=25:(-0.8,-0.0)|V:0.3m/s -> t=26:(-0.7,-0.0)|V:0.3m/s -> t=27:(-0.7,-0.0)|V:0.3m/s -> t=28:(-0.7,-0.0)|V:0.3m/s -> t=29:(-0.7,-0.0)|V:0.3m/s -> t=30:(-0.6,-0.0)|V:0.3m/s -> t=31:(-0.6,-0.0)|V:0.2m/s -> t=32:(-0.6,-0.0)|V:0.2m/s -> t=33:(-0.6,-0.0)|V:0.2m/s -> t=34:(-0.6,-0.0)|V:0.1m/s -> t=35:(-0.6,-0.0)|V:0.1m/s -> t=36:(-0.6,-0.0)|V:0.1m/s -> t=37:(-0.6,-0.0)|V:0.1m/s -> t=38:(-0.5,-0.0)|V:0.2m/s -> t=39:(-0.5,-0.0)|V:0.2m/s -> t=40:(-0.4,-0.0)|V:0.2m/s -> t=41:(-0.4,-0.0)|V:0.2m/s -> t=42:(-0.3,-0.0)|V:0.2m/s -> t=43:(-0.2,-0.0)|V:0.2m/s -> t=44:(-0.2,-0.0)|V:0.2m/s -> t=45:(-0.1,-0.0)|V:0.2m/s -> t=46:(-0.1,-0.0)|V:0.2m/s -> t=47:(-0.1,-0.0)|V:0.2m/s -> t=48:(-0.0,-0.0)|V:0.2m/s -> t=49:(0.0,0.0)|V:0.2m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=1, 1338 tokens)

```text
Scenario '650b08b9-2b6c-4de6-a1cb-8b3de3f235c7' in palo-alto. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Autonomous Vehicle (Impact:18) [pos(t=49): follower | motion: similar speed to ego | interaction: low influence]: t=0:(-24.3,0.4)|V:2.4m/s -> t=1:(-24.1,0.4)|V:2.4m/s -> t=2:(-23.8,0.4)|V:4.7m/s -> t=3:(-23.5,0.4)|V:4.7m/s -> t=4:(-23.1,0.4)|V:4.6m/s -> t=5:(-22.7,0.4)|V:4.6m/s -> t=6:(-22.2,0.4)|V:4.6m/s -> t=7:(-21.7,0.4)|V:4.5m/s -> t=8:(-21.3,0.4)|V:4.4m/s -> t=9:(-20.8,0.4)|V:4.3m/s -> t=10:(-20.4,0.4)|V:4.2m/s -> t=11:(-19.9,0.4)|V:4.2m/s -> t=12:(-19.5,0.4)|V:4.1m/s -> t=13:(-19.1,0.4)|V:4.0m/s -> t=14:(-18.7,0.4)|V:3.9m/s -> t=15:(-18.4,0.4)|V:3.8m/s -> t=16:(-18.0,0.4)|V:3.7m/s -> t=17:(-17.6,0.4)|V:3.7m/s -> t=18:(-17.3,0.4)|V:3.6m/s -> t=19:(-16.9,0.4)|V:3.5m/s -> t=20:(-16.6,0.4)|V:3.4m/s -> t=21:(-16.3,0.4)|V:3.3m/s -> t=22:(-15.9,0.4)|V:3.2m/s -> t=23:(-15.6,0.4)|V:3.1m/s -> t=24:(-15.3,0.4)|V:2.9m/s -> t=25:(-15.0,0.4)|V:2.8m/s -> t=26:(-14.8,0.4)|V:2.7m/s -> t=27:(-14.5,0.4)|V:2.6m/s -> t=28:(-14.3,0.4)|V:2.5m/s -> t=29:(-14.0,0.4)|V:2.4m/s -> t=30:(-13.8,0.4)|V:2.2m/s -> t=31:(-13.6,0.4)|V:2.0m/s -> t=32:(-13.4,0.4)|V:1.9m/s -> t=33:(-13.2,0.4)|V:1.8m/s -> t=34:(-13.1,0.4)|V:1.7m/s -> t=35:(-12.9,0.4)|V:1.5m/s -> t=36:(-12.8,0.4)|V:1.4m/s -> t=37:(-12.7,0.4)|V:1.3m/s -> t=38:(-12.5,0.4)|V:1.2m/s -> t=39:(-12.4,0.4)|V:1.0m/s -> t=40:(-12.3,0.4)|V:0.9m/s -> t=41:(-12.3,0.4)|V:0.8m/s -> t=42:(-12.2,0.4)|V:0.7m/s -> t=43:(-12.1,0.4)|V:0.6m/s -> t=44:(-12.1,0.4)|V:0.6m/s -> t=45:(-12.0,0.4)|V:0.5m/s -> t=46:(-12.0,0.4)|V:0.3m/s -> t=47:(-12.0,0.4)|V:0.3m/s -> t=48:(-11.9,0.4)|V:0.2m/s -> t=49:(-11.9,0.4)|V:0.2m/s
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=6, 181 tokens)

```text
Scenario '650b08b9-2b6c-4de6-a1cb-8b3de3f235c7' in palo-alto. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Vehicle (Impact:8) (Entered scene at t=24) [pos(t=49): left-rear | motion: stationary | interaction: low influence (parked)]: t=24~49 Stationary at (-7.3, 4.2)
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=13, 184 tokens)

```text
Scenario '650b08b9-2b6c-4de6-a1cb-8b3de3f235c7' in palo-alto. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Vehicle (Impact:6) (Entered scene at t=47) [pos(t=49): far-right-front | motion: stationary | interaction: low influence (parked)]: t=47~49 Stationary at (14.1, -14.6)
[PREDICT]:
```

---

## Scene 5/5 — `a184c073-2b99-4a6b-81f4-5a10d1f95fdd` (dataset index 16)

**Agents written:** 6  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1299/2000 |
| 1 | NBR1 | 6 | True | 1298/2000 |
| 2 | NBR2 | 1 | True | 1292/2000 |
| 3 | NBR3 | 2 | True | 1322/2000 |
| 4 | NBR4 | 4 | True | 1358/2000 |
| 5 | NBR5 | 13 | True | 1401/2000 |

### Agent 0 — FOCAL (traj_id=0, 1299 tokens)

```text
Scenario 'a184c073-2b99-4a6b-81f4-5a10d1f95fdd' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection, Stop-and-Go Platooning

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Allowed-Left

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-15.5,2.5)|V:3.7m/s -> t=1:(-15.3,2.5)|V:3.7m/s -> t=2:(-15.1,2.4)|V:3.7m/s -> t=3:(-14.9,2.3)|V:3.7m/s -> t=4:(-14.5,2.3)|V:3.7m/s -> t=5:(-14.2,2.2)|V:3.7m/s -> t=6:(-13.8,2.1)|V:3.7m/s -> t=7:(-13.4,2.1)|V:3.7m/s -> t=8:(-13.0,2.0)|V:3.7m/s -> t=9:(-12.7,1.9)|V:3.7m/s -> t=10:(-12.3,1.8)|V:3.7m/s -> t=11:(-11.9,1.7)|V:3.7m/s -> t=12:(-11.6,1.6)|V:3.7m/s -> t=13:(-11.2,1.6)|V:3.7m/s -> t=14:(-10.9,1.5)|V:3.7m/s -> t=15:(-10.5,1.4)|V:3.7m/s -> t=16:(-10.2,1.3)|V:3.7m/s -> t=17:(-9.8,1.2)|V:3.7m/s -> t=18:(-9.4,1.1)|V:3.7m/s -> t=19:(-9.1,1.0)|V:3.7m/s -> t=20:(-8.7,1.0)|V:3.7m/s -> t=21:(-8.3,0.9)|V:3.6m/s -> t=22:(-8.0,0.8)|V:3.6m/s -> t=23:(-7.6,0.7)|V:3.7m/s -> t=24:(-7.3,0.7)|V:3.8m/s -> t=25:(-7.0,0.6)|V:3.9m/s -> t=26:(-6.7,0.5)|V:3.9m/s -> t=27:(-6.4,0.5)|V:3.7m/s -> t=28:(-6.2,0.4)|V:3.4m/s -> t=29:(-5.9,0.4)|V:3.2m/s -> t=30:(-5.6,0.3)|V:3.0m/s -> t=31:(-5.3,0.3)|V:3.0m/s -> t=32:(-5.0,0.2)|V:2.9m/s -> t=33:(-4.8,0.2)|V:2.9m/s -> t=34:(-4.5,0.2)|V:2.8m/s -> t=35:(-4.2,0.1)|V:2.8m/s -> t=36:(-3.8,0.1)|V:2.8m/s -> t=37:(-3.5,0.0)|V:2.8m/s -> t=38:(-3.2,0.0)|V:2.8m/s -> t=39:(-2.9,0.0)|V:2.9m/s -> t=40:(-2.6,-0.0)|V:2.9m/s -> t=41:(-2.3,-0.0)|V:2.9m/s -> t=42:(-2.0,-0.0)|V:2.9m/s -> t=43:(-1.7,-0.0)|V:2.9m/s -> t=44:(-1.4,-0.0)|V:3.0m/s -> t=45:(-1.1,-0.0)|V:3.0m/s -> t=46:(-0.9,-0.0)|V:3.0m/s -> t=47:(-0.6,-0.0)|V:2.9m/s -> t=48:(-0.3,-0.0)|V:2.9m/s -> t=49:(0.0,0.0)|V:2.9m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=6, 1298 tokens)

```text
Scenario 'a184c073-2b99-4a6b-81f4-5a10d1f95fdd' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection, Stop-and-Go Platooning

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Allowed-Left

[AGENT HISTORY]
- Vehicle (Impact:71) [pos(t=49): lead | motion: faster than ego | interaction: high influence (direct path blockage)]: t=0:(-11.2,8.5)|V:6.9m/s -> t=1:(-10.9,8.4)|V:6.9m/s -> t=2:(-10.5,8.3)|V:6.9m/s -> t=3:(-10.0,8.2)|V:6.9m/s -> t=4:(-9.4,8.0)|V:6.9m/s -> t=5:(-8.8,7.9)|V:6.9m/s -> t=6:(-8.1,7.7)|V:6.8m/s -> t=7:(-7.4,7.6)|V:6.8m/s -> t=8:(-6.7,7.4)|V:6.8m/s -> t=9:(-5.9,7.2)|V:6.8m/s -> t=10:(-5.3,7.0)|V:6.8m/s -> t=11:(-4.6,6.9)|V:6.8m/s -> t=12:(-3.9,6.7)|V:6.8m/s -> t=13:(-3.3,6.6)|V:6.8m/s -> t=14:(-2.6,6.4)|V:6.8m/s -> t=15:(-2.0,6.2)|V:6.7m/s -> t=16:(-1.4,6.1)|V:6.7m/s -> t=17:(-0.7,5.9)|V:6.7m/s -> t=18:(-0.1,5.8)|V:6.6m/s -> t=19:(0.5,5.6)|V:6.5m/s -> t=20:(1.2,5.4)|V:6.4m/s -> t=21:(1.8,5.3)|V:6.3m/s -> t=22:(2.4,5.1)|V:6.3m/s -> t=23:(3.0,5.0)|V:6.3m/s -> t=24:(3.6,4.8)|V:6.3m/s -> t=25:(4.2,4.7)|V:6.3m/s -> t=26:(4.8,4.5)|V:6.2m/s -> t=27:(5.4,4.4)|V:6.1m/s -> t=28:(6.0,4.3)|V:6.1m/s -> t=29:(6.6,4.1)|V:6.2m/s -> t=30:(7.2,4.0)|V:6.1m/s -> t=31:(7.8,3.9)|V:6.1m/s -> t=32:(8.4,3.7)|V:5.9m/s -> t=33:(8.9,3.6)|V:5.8m/s -> t=34:(9.5,3.5)|V:5.7m/s -> t=35:(10.1,3.3)|V:5.6m/s -> t=36:(10.6,3.2)|V:5.5m/s -> t=37:(11.2,3.1)|V:5.4m/s -> t=38:(11.7,2.9)|V:5.4m/s -> t=39:(12.3,2.8)|V:5.4m/s -> t=40:(12.8,2.7)|V:5.4m/s -> t=41:(13.3,2.6)|V:5.4m/s -> t=42:(13.8,2.4)|V:5.3m/s -> t=43:(14.3,2.3)|V:5.3m/s -> t=44:(14.8,2.2)|V:5.2m/s -> t=45:(15.4,2.0)|V:5.1m/s -> t=46:(15.9,1.9)|V:5.0m/s -> t=47:(16.4,1.8)|V:4.8m/s -> t=48:(16.9,1.7)|V:4.7m/s -> t=49:(17.4,1.5)|V:4.6m/s
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=1, 1292 tokens)

```text
Scenario 'a184c073-2b99-4a6b-81f4-5a10d1f95fdd' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection, Stop-and-Go Platooning

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Allowed-Left

[AGENT HISTORY]
- Autonomous Vehicle (Impact:32) [pos(t=49): left-front | motion: slower than ego | interaction: moderate influence (close proximity)]: t=0:(2.8,8.4)|V:2.3m/s -> t=1:(3.0,8.3)|V:2.3m/s -> t=2:(3.3,8.3)|V:4.6m/s -> t=3:(3.6,8.2)|V:4.5m/s -> t=4:(3.9,8.1)|V:4.4m/s -> t=5:(4.3,8.0)|V:4.4m/s -> t=6:(4.7,7.9)|V:4.3m/s -> t=7:(5.2,7.8)|V:4.3m/s -> t=8:(5.6,7.7)|V:4.2m/s -> t=9:(6.1,7.6)|V:4.2m/s -> t=10:(6.5,7.5)|V:4.1m/s -> t=11:(6.9,7.4)|V:4.1m/s -> t=12:(7.3,7.3)|V:4.0m/s -> t=13:(7.7,7.2)|V:4.0m/s -> t=14:(8.0,7.1)|V:3.9m/s -> t=15:(8.4,7.0)|V:3.8m/s -> t=16:(8.8,6.9)|V:3.8m/s -> t=17:(9.1,6.9)|V:3.7m/s -> t=18:(9.5,6.8)|V:3.7m/s -> t=19:(9.8,6.7)|V:3.6m/s -> t=20:(10.2,6.6)|V:3.5m/s -> t=21:(10.5,6.5)|V:3.5m/s -> t=22:(10.9,6.4)|V:3.4m/s -> t=23:(11.2,6.4)|V:3.4m/s -> t=24:(11.5,6.3)|V:3.2m/s -> t=25:(11.8,6.2)|V:3.2m/s -> t=26:(12.1,6.1)|V:3.1m/s -> t=27:(12.4,6.1)|V:3.0m/s -> t=28:(12.7,6.0)|V:3.0m/s -> t=29:(13.0,5.9)|V:2.9m/s -> t=30:(13.2,5.9)|V:2.8m/s -> t=31:(13.5,5.8)|V:2.7m/s -> t=32:(13.8,5.7)|V:2.7m/s -> t=33:(14.0,5.7)|V:2.5m/s -> t=34:(14.2,5.6)|V:2.5m/s -> t=35:(14.5,5.6)|V:2.4m/s -> t=36:(14.7,5.5)|V:2.3m/s -> t=37:(14.9,5.5)|V:2.3m/s -> t=38:(15.1,5.4)|V:2.2m/s -> t=39:(15.4,5.3)|V:2.1m/s -> t=40:(15.6,5.3)|V:2.1m/s -> t=41:(15.7,5.3)|V:2.1m/s -> t=42:(15.9,5.2)|V:1.9m/s -> t=43:(16.1,5.2)|V:1.8m/s -> t=44:(16.3,5.1)|V:1.7m/s -> t=45:(16.4,5.1)|V:1.6m/s -> t=46:(16.6,5.1)|V:1.4m/s -> t=47:(16.7,5.0)|V:1.4m/s -> t=48:(16.8,5.0)|V:1.3m/s -> t=49:(16.9,5.0)|V:1.2m/s
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=2, 1322 tokens)

```text
Scenario 'a184c073-2b99-4a6b-81f4-5a10d1f95fdd' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection, Stop-and-Go Platooning

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Allowed-Left

[AGENT HISTORY]
- Vehicle (Impact:20) [pos(t=49): far-left-front | motion: faster than ego | interaction: low influence]: t=0:(-14.9,12.6)|V:6.2m/s -> t=1:(-14.6,12.5)|V:6.1m/s -> t=2:(-14.2,12.4)|V:6.0m/s -> t=3:(-13.7,12.3)|V:5.9m/s -> t=4:(-13.2,12.2)|V:5.8m/s -> t=5:(-12.7,12.0)|V:5.8m/s -> t=6:(-12.1,11.9)|V:5.8m/s -> t=7:(-11.5,11.7)|V:5.8m/s -> t=8:(-10.9,11.5)|V:5.8m/s -> t=9:(-10.3,11.4)|V:5.8m/s -> t=10:(-9.7,11.2)|V:5.8m/s -> t=11:(-9.2,11.1)|V:5.7m/s -> t=12:(-8.6,10.9)|V:5.7m/s -> t=13:(-8.1,10.8)|V:5.5m/s -> t=14:(-7.6,10.7)|V:5.4m/s -> t=15:(-7.1,10.6)|V:5.3m/s -> t=16:(-6.6,10.4)|V:5.2m/s -> t=17:(-6.2,10.3)|V:5.0m/s -> t=18:(-5.7,10.2)|V:4.9m/s -> t=19:(-5.2,10.1)|V:4.7m/s -> t=20:(-4.8,10.0)|V:4.6m/s -> t=21:(-4.3,9.9)|V:4.5m/s -> t=22:(-3.9,9.8)|V:4.4m/s -> t=23:(-3.5,9.6)|V:4.3m/s -> t=24:(-3.0,9.5)|V:4.3m/s -> t=25:(-2.6,9.4)|V:4.2m/s -> t=26:(-2.2,9.3)|V:4.1m/s -> t=27:(-1.8,9.3)|V:4.0m/s -> t=28:(-1.4,9.2)|V:4.0m/s -> t=29:(-1.0,9.1)|V:3.9m/s -> t=30:(-0.7,9.0)|V:3.9m/s -> t=31:(-0.3,8.9)|V:3.8m/s -> t=32:(0.1,8.8)|V:3.8m/s -> t=33:(0.4,8.7)|V:3.7m/s -> t=34:(0.8,8.6)|V:3.6m/s -> t=35:(1.2,8.5)|V:3.6m/s -> t=36:(1.5,8.5)|V:3.6m/s -> t=37:(1.9,8.4)|V:3.6m/s -> t=38:(2.3,8.3)|V:3.5m/s -> t=39:(2.6,8.2)|V:3.5m/s -> t=40:(3.0,8.1)|V:3.5m/s -> t=41:(3.4,8.0)|V:3.6m/s -> t=42:(3.7,7.9)|V:3.6m/s -> t=43:(4.1,7.8)|V:3.6m/s -> t=44:(4.4,7.8)|V:3.6m/s -> t=45:(4.7,7.7)|V:3.6m/s -> t=46:(5.1,7.6)|V:3.6m/s -> t=47:(5.4,7.5)|V:3.6m/s -> t=48:(5.7,7.4)|V:3.5m/s -> t=49:(6.0,7.4)|V:3.4m/s
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=4, 1358 tokens)

```text
Scenario 'a184c073-2b99-4a6b-81f4-5a10d1f95fdd' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection, Stop-and-Go Platooning

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Allowed-Left

[AGENT HISTORY]
- Vehicle (Impact:15) [pos(t=49): far-left-front | motion: faster than ego | interaction: low influence]: t=0:(-20.0,17.1)|V:6.7m/s -> t=1:(-19.6,17.1)|V:6.7m/s -> t=2:(-19.2,17.0)|V:6.8m/s -> t=3:(-18.7,16.8)|V:6.8m/s -> t=4:(-18.1,16.7)|V:6.9m/s -> t=5:(-17.5,16.6)|V:6.9m/s -> t=6:(-16.8,16.4)|V:7.0m/s -> t=7:(-16.0,16.2)|V:7.0m/s -> t=8:(-15.3,16.1)|V:7.1m/s -> t=9:(-14.5,15.9)|V:7.1m/s -> t=10:(-13.8,15.7)|V:7.1m/s -> t=11:(-13.1,15.6)|V:7.1m/s -> t=12:(-12.5,15.4)|V:7.1m/s -> t=13:(-11.8,15.3)|V:7.1m/s -> t=14:(-11.1,15.1)|V:7.1m/s -> t=15:(-10.4,14.9)|V:7.1m/s -> t=16:(-9.7,14.8)|V:7.1m/s -> t=17:(-9.1,14.6)|V:7.1m/s -> t=18:(-8.4,14.5)|V:7.0m/s -> t=19:(-7.7,14.3)|V:7.0m/s -> t=20:(-7.1,14.2)|V:6.9m/s -> t=21:(-6.4,14.0)|V:6.8m/s -> t=22:(-5.8,13.9)|V:6.7m/s -> t=23:(-5.1,13.7)|V:6.7m/s -> t=24:(-4.4,13.5)|V:6.6m/s -> t=25:(-3.8,13.4)|V:6.6m/s -> t=26:(-3.1,13.2)|V:6.5m/s -> t=27:(-2.5,13.1)|V:6.4m/s -> t=28:(-1.8,12.9)|V:6.4m/s -> t=29:(-1.2,12.8)|V:6.3m/s -> t=30:(-0.5,12.6)|V:6.3m/s -> t=31:(0.1,12.5)|V:6.4m/s -> t=32:(0.7,12.3)|V:6.4m/s -> t=33:(1.4,12.2)|V:6.4m/s -> t=34:(2.0,12.0)|V:6.4m/s -> t=35:(2.6,11.9)|V:6.4m/s -> t=36:(3.2,11.7)|V:6.4m/s -> t=37:(3.8,11.6)|V:6.4m/s -> t=38:(4.4,11.5)|V:6.3m/s -> t=39:(5.0,11.3)|V:6.2m/s -> t=40:(5.6,11.2)|V:6.1m/s -> t=41:(6.2,11.1)|V:6.1m/s -> t=42:(6.8,10.9)|V:6.0m/s -> t=43:(7.3,10.8)|V:6.0m/s -> t=44:(7.9,10.7)|V:5.9m/s -> t=45:(8.4,10.6)|V:5.8m/s -> t=46:(9.0,10.5)|V:5.7m/s -> t=47:(9.5,10.3)|V:5.6m/s -> t=48:(10.1,10.2)|V:5.5m/s -> t=49:(10.6,10.1)|V:5.4m/s
[PREDICT]:
```

### Agent 5 — NBR5 (traj_id=13, 1401 tokens)

```text
Scenario 'a184c073-2b99-4a6b-81f4-5a10d1f95fdd' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection, Stop-and-Go Platooning

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply, Lane-Change-Allowed-Left

[AGENT HISTORY]
- Vehicle (Impact:15) [pos(t=49): far-left-rear | motion: faster than ego | interaction: low influence]: t=0:(-48.0,17.8)|V:6.9m/s -> t=1:(-47.9,17.7)|V:6.9m/s -> t=2:(-47.6,17.6)|V:7.1m/s -> t=3:(-47.3,17.5)|V:7.3m/s -> t=4:(-47.0,17.3)|V:7.3m/s -> t=5:(-46.6,17.1)|V:7.4m/s -> t=6:(-46.1,16.9)|V:7.5m/s -> t=7:(-45.7,16.6)|V:7.5m/s -> t=8:(-45.0,16.4)|V:7.6m/s -> t=9:(-44.3,16.1)|V:7.7m/s -> t=10:(-43.5,15.9)|V:7.8m/s -> t=11:(-42.7,15.7)|V:7.9m/s -> t=12:(-41.8,15.5)|V:8.0m/s -> t=13:(-40.9,15.3)|V:8.2m/s -> t=14:(-40.0,15.0)|V:8.3m/s -> t=15:(-39.1,14.8)|V:8.3m/s -> t=16:(-38.2,14.6)|V:8.4m/s -> t=17:(-37.4,14.4)|V:8.5m/s -> t=18:(-36.6,14.2)|V:8.6m/s -> t=19:(-35.8,14.0)|V:8.6m/s -> t=20:(-35.0,13.9)|V:8.5m/s -> t=21:(-34.2,13.7)|V:8.5m/s -> t=22:(-33.5,13.5)|V:8.5m/s -> t=23:(-32.8,13.3)|V:8.5m/s -> t=24:(-32.0,13.1)|V:8.5m/s -> t=25:(-31.2,12.9)|V:8.5m/s -> t=26:(-30.3,12.7)|V:8.5m/s -> t=27:(-29.4,12.5)|V:8.5m/s -> t=28:(-28.5,12.4)|V:8.6m/s -> t=29:(-27.7,12.2)|V:8.6m/s -> t=30:(-26.8,12.0)|V:8.6m/s -> t=31:(-26.0,11.8)|V:8.6m/s -> t=32:(-25.1,11.5)|V:8.6m/s -> t=33:(-24.3,11.3)|V:8.6m/s -> t=34:(-23.5,11.1)|V:8.6m/s -> t=35:(-22.7,11.0)|V:8.6m/s -> t=36:(-22.0,10.8)|V:8.5m/s -> t=37:(-21.2,10.6)|V:8.5m/s -> t=38:(-20.5,10.4)|V:8.4m/s -> t=39:(-19.7,10.2)|V:8.3m/s -> t=40:(-19.0,10.0)|V:8.2m/s -> t=41:(-18.3,9.8)|V:8.1m/s -> t=42:(-17.5,9.7)|V:8.0m/s -> t=43:(-16.8,9.5)|V:7.9m/s -> t=44:(-16.0,9.3)|V:7.8m/s -> t=45:(-15.3,9.1)|V:7.7m/s -> t=46:(-14.6,9.0)|V:7.6m/s -> t=47:(-13.8,8.8)|V:7.5m/s -> t=48:(-13.1,8.6)|V:7.4m/s -> t=49:(-12.3,8.4)|V:7.3m/s
[PREDICT]:
```

# Split: val  (`data_av2/features/val`, 20 scenes total)

---

## Scene 1/5 — `1f6e2e71-f13e-4547-9b57-dd46256f936f` (dataset index 4)

**Agents written:** 6  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1263/2000 |
| 1 | NBR1 | 31 | True | 181/2000 |
| 2 | NBR2 | 8 | True | 1290/2000 |
| 3 | NBR3 | 25 | True | 1276/2000 |
| 4 | NBR4 | 20 | True | 1291/2000 |
| 5 | NBR5 | 15 | True | 181/2000 |

### Agent 0 — FOCAL (traj_id=0, 1263 tokens)

```text
Scenario '1f6e2e71-f13e-4547-9b57-dd46256f936f' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-3.0,0.6)|V:1.1m/s -> t=1:(-2.9,0.6)|V:1.1m/s -> t=2:(-2.9,0.6)|V:1.0m/s -> t=3:(-2.8,0.6)|V:1.0m/s -> t=4:(-2.8,0.6)|V:0.9m/s -> t=5:(-2.7,0.5)|V:0.9m/s -> t=6:(-2.6,0.5)|V:0.9m/s -> t=7:(-2.5,0.4)|V:0.9m/s -> t=8:(-2.5,0.4)|V:1.0m/s -> t=9:(-2.4,0.3)|V:1.0m/s -> t=10:(-2.3,0.2)|V:1.0m/s -> t=11:(-2.2,0.2)|V:1.1m/s -> t=12:(-2.1,0.1)|V:1.1m/s -> t=13:(-2.0,0.1)|V:1.1m/s -> t=14:(-1.9,0.0)|V:1.0m/s -> t=15:(-1.9,-0.0)|V:1.0m/s -> t=16:(-1.8,-0.1)|V:1.0m/s -> t=17:(-1.7,-0.1)|V:1.0m/s -> t=18:(-1.6,-0.2)|V:0.9m/s -> t=19:(-1.6,-0.2)|V:0.9m/s -> t=20:(-1.5,-0.2)|V:0.8m/s -> t=21:(-1.4,-0.3)|V:0.8m/s -> t=22:(-1.4,-0.3)|V:0.8m/s -> t=23:(-1.3,-0.3)|V:0.7m/s -> t=24:(-1.2,-0.3)|V:0.8m/s -> t=25:(-1.1,-0.3)|V:0.8m/s -> t=26:(-1.0,-0.3)|V:0.8m/s -> t=27:(-1.0,-0.3)|V:0.8m/s -> t=28:(-0.9,-0.3)|V:0.8m/s -> t=29:(-0.9,-0.3)|V:0.7m/s -> t=30:(-0.8,-0.3)|V:0.7m/s -> t=31:(-0.8,-0.3)|V:0.6m/s -> t=32:(-0.7,-0.3)|V:0.6m/s -> t=33:(-0.7,-0.3)|V:0.4m/s -> t=34:(-0.7,-0.3)|V:0.3m/s -> t=35:(-0.7,-0.3)|V:0.2m/s -> t=36:(-0.7,-0.3)|V:0.1m/s -> t=37:(-0.7,-0.3)|V:0.1m/s -> t=38:(-0.7,-0.3)|V:0.1m/s -> t=39:(-0.7,-0.3)|V:0.1m/s -> t=40:(-0.7,-0.3)|V:0.1m/s -> t=41:(-0.7,-0.3)|V:0.1m/s -> t=42:(-0.6,-0.2)|V:0.1m/s -> t=43:(-0.6,-0.2)|V:0.1m/s -> t=44:(-0.5,-0.2)|V:0.1m/s -> t=45:(-0.4,-0.2)|V:0.0m/s -> t=46:(-0.4,-0.1)|V:0.2m/s -> t=47:(-0.3,-0.1)|V:0.5m/s -> t=48:(-0.1,-0.0)|V:0.8m/s -> t=49:(0.0,0.0)|V:1.2m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=31, 181 tokens)

```text
Scenario '1f6e2e71-f13e-4547-9b57-dd46256f936f' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Vehicle (Impact:73) (Entered scene at t=15) [pos(t=49): lead | motion: stationary | interaction: high influence (direct path blockage)]: t=15~49 Stationary at (14.6, 1.0)
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=8, 1290 tokens)

```text
Scenario '1f6e2e71-f13e-4547-9b57-dd46256f936f' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Vehicle (Impact:17) [pos(t=49): far-right-front | motion: faster than ego | interaction: low influence]: t=0:(9.3,-21.5)|V:4.5m/s -> t=1:(9.2,-21.4)|V:4.5m/s -> t=2:(9.2,-21.2)|V:4.3m/s -> t=3:(9.2,-20.9)|V:4.2m/s -> t=4:(9.1,-20.6)|V:4.1m/s -> t=5:(9.1,-20.3)|V:4.1m/s -> t=6:(9.0,-19.8)|V:4.0m/s -> t=7:(9.0,-19.4)|V:3.9m/s -> t=8:(8.9,-19.0)|V:3.8m/s -> t=9:(8.9,-18.5)|V:3.8m/s -> t=10:(8.8,-18.1)|V:3.8m/s -> t=11:(8.8,-17.7)|V:3.8m/s -> t=12:(8.8,-17.3)|V:3.8m/s -> t=13:(8.7,-16.9)|V:3.8m/s -> t=14:(8.7,-16.6)|V:3.7m/s -> t=15:(8.7,-16.2)|V:3.6m/s -> t=16:(8.6,-15.9)|V:3.6m/s -> t=17:(8.6,-15.6)|V:3.5m/s -> t=18:(8.6,-15.3)|V:3.5m/s -> t=19:(8.5,-14.9)|V:3.3m/s -> t=20:(8.5,-14.6)|V:3.2m/s -> t=21:(8.5,-14.3)|V:3.1m/s -> t=22:(8.5,-14.0)|V:3.0m/s -> t=23:(8.4,-13.7)|V:2.9m/s -> t=24:(8.4,-13.4)|V:2.8m/s -> t=25:(8.4,-13.1)|V:2.7m/s -> t=26:(8.4,-12.9)|V:2.6m/s -> t=27:(8.4,-12.7)|V:2.4m/s -> t=28:(8.4,-12.4)|V:2.3m/s -> t=29:(8.4,-12.3)|V:2.1m/s -> t=30:(8.4,-12.1)|V:1.9m/s -> t=31:(8.4,-11.9)|V:1.8m/s -> t=32:(8.4,-11.7)|V:1.6m/s -> t=33:(8.4,-11.6)|V:1.5m/s -> t=34:(8.3,-11.4)|V:1.4m/s -> t=35:(8.3,-11.3)|V:1.3m/s -> t=36:(8.3,-11.2)|V:1.3m/s -> t=37:(8.3,-11.0)|V:1.3m/s -> t=38:(8.3,-10.9)|V:1.2m/s -> t=39:(8.3,-10.8)|V:1.2m/s -> t=40:(8.3,-10.7)|V:1.0m/s -> t=41:(8.2,-10.6)|V:0.9m/s -> t=42:(8.2,-10.5)|V:0.8m/s -> t=43:(8.2,-10.4)|V:0.8m/s -> t=44:(8.2,-10.4)|V:0.8m/s -> t=45:(8.2,-10.3)|V:0.8m/s -> t=46:(8.2,-10.3)|V:0.8m/s -> t=47:(8.2,-10.2)|V:0.7m/s -> t=48:(8.2,-10.2)|V:0.6m/s -> t=49:(8.2,-10.1)|V:0.5m/s
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=25, 1276 tokens)

```text
Scenario '1f6e2e71-f13e-4547-9b57-dd46256f936f' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Vehicle (Impact:16) (Entered scene at t=1) [pos(t=49): left-front | motion: similar speed to ego | interaction: low influence]: t=1:(16.0,3.6)|V:1.0m/s -> t=2:(16.0,3.6)|V:1.0m/s -> t=3:(15.9,3.5)|V:1.0m/s -> t=4:(15.8,3.5)|V:1.0m/s -> t=5:(15.8,3.5)|V:1.0m/s -> t=6:(15.7,3.5)|V:1.0m/s -> t=7:(15.6,3.5)|V:1.0m/s -> t=8:(15.5,3.5)|V:1.0m/s -> t=9:(15.4,3.5)|V:1.0m/s -> t=10:(15.3,3.5)|V:0.9m/s -> t=11:(15.2,3.5)|V:0.8m/s -> t=12:(15.1,3.5)|V:0.8m/s -> t=13:(15.0,3.4)|V:0.8m/s -> t=14:(14.9,3.4)|V:0.8m/s -> t=15:(14.8,3.4)|V:0.9m/s -> t=16:(14.7,3.4)|V:1.0m/s -> t=17:(14.6,3.4)|V:1.0m/s -> t=18:(14.4,3.4)|V:1.0m/s -> t=19:(14.3,3.4)|V:1.0m/s -> t=20:(14.2,3.4)|V:1.0m/s -> t=21:(14.1,3.4)|V:1.0m/s -> t=22:(14.1,3.3)|V:0.9m/s -> t=23:(14.0,3.3)|V:0.9m/s -> t=24:(13.9,3.3)|V:0.8m/s -> t=25:(13.9,3.3)|V:0.7m/s -> t=26:(13.9,3.3)|V:0.6m/s -> t=27:(13.8,3.3)|V:0.5m/s -> t=28:(13.8,3.3)|V:0.3m/s -> t=29:(13.8,3.3)|V:0.2m/s -> t=30:(13.8,3.3)|V:0.1m/s -> t=31:(13.8,3.3)|V:0.1m/s -> t=32:(13.8,3.3)|V:0.1m/s -> t=33:(13.8,3.3)|V:0.1m/s -> t=34:(13.8,3.4)|V:0.2m/s -> t=35:(13.8,3.4)|V:0.2m/s -> t=36:(13.8,3.4)|V:0.2m/s -> t=37:(13.8,3.4)|V:0.2m/s -> t=38:(13.8,3.4)|V:0.2m/s -> t=39:(13.8,3.4)|V:0.2m/s -> t=40:(13.8,3.4)|V:0.2m/s -> t=41:(13.8,3.4)|V:0.2m/s -> t=42:(13.8,3.4)|V:0.2m/s -> t=43:(13.8,3.4)|V:0.2m/s -> t=44:(13.8,3.4)|V:0.2m/s -> t=45:(13.8,3.4)|V:0.2m/s -> t=46:(13.8,3.4)|V:0.1m/s -> t=47:(13.8,3.4)|V:0.1m/s -> t=48:(13.8,3.4)|V:0.0m/s -> t=49:(13.9,3.4)|V:0.0m/s
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=20, 1291 tokens)

```text
Scenario '1f6e2e71-f13e-4547-9b57-dd46256f936f' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Vehicle (Impact:8) (Entered scene at t=1) [pos(t=49): far-right-front | motion: faster than ego | interaction: low influence]: t=1:(10.5,-35.1)|V:4.4m/s -> t=2:(10.5,-34.9)|V:4.5m/s -> t=3:(10.5,-34.6)|V:4.6m/s -> t=4:(10.4,-34.2)|V:4.6m/s -> t=5:(10.4,-33.9)|V:4.5m/s -> t=6:(10.4,-33.4)|V:4.4m/s -> t=7:(10.3,-33.0)|V:4.3m/s -> t=8:(10.3,-32.5)|V:4.2m/s -> t=9:(10.2,-32.1)|V:4.2m/s -> t=10:(10.2,-31.6)|V:4.1m/s -> t=11:(10.1,-31.2)|V:4.0m/s -> t=12:(10.1,-30.8)|V:4.0m/s -> t=13:(10.1,-30.4)|V:4.0m/s -> t=14:(10.0,-30.0)|V:3.9m/s -> t=15:(10.0,-29.6)|V:3.9m/s -> t=16:(9.9,-29.3)|V:3.8m/s -> t=17:(9.9,-28.9)|V:3.8m/s -> t=18:(9.9,-28.5)|V:3.7m/s -> t=19:(9.8,-28.1)|V:3.6m/s -> t=20:(9.8,-27.8)|V:3.6m/s -> t=21:(9.7,-27.4)|V:3.5m/s -> t=22:(9.7,-27.1)|V:3.5m/s -> t=23:(9.7,-26.7)|V:3.5m/s -> t=24:(9.6,-26.4)|V:3.5m/s -> t=25:(9.6,-26.1)|V:3.4m/s -> t=26:(9.6,-25.7)|V:3.4m/s -> t=27:(9.5,-25.4)|V:3.4m/s -> t=28:(9.5,-25.1)|V:3.4m/s -> t=29:(9.4,-24.8)|V:3.3m/s -> t=30:(9.4,-24.5)|V:3.3m/s -> t=31:(9.4,-24.2)|V:3.2m/s -> t=32:(9.3,-23.9)|V:3.2m/s -> t=33:(9.3,-23.6)|V:3.1m/s -> t=34:(9.2,-23.3)|V:3.0m/s -> t=35:(9.2,-23.0)|V:3.0m/s -> t=36:(9.2,-22.7)|V:2.9m/s -> t=37:(9.1,-22.4)|V:2.8m/s -> t=38:(9.1,-22.2)|V:2.8m/s -> t=39:(9.1,-21.9)|V:2.8m/s -> t=40:(9.1,-21.7)|V:2.8m/s -> t=41:(9.0,-21.4)|V:2.7m/s -> t=42:(9.0,-21.2)|V:2.7m/s -> t=43:(9.0,-21.0)|V:2.6m/s -> t=44:(9.0,-20.7)|V:2.5m/s -> t=45:(8.9,-20.5)|V:2.3m/s -> t=46:(8.9,-20.3)|V:2.3m/s -> t=47:(8.9,-20.1)|V:2.2m/s -> t=48:(8.9,-19.9)|V:2.0m/s -> t=49:(8.9,-19.7)|V:1.9m/s
[PREDICT]:
```

### Agent 5 — NBR5 (traj_id=15, 181 tokens)

```text
Scenario '1f6e2e71-f13e-4547-9b57-dd46256f936f' in pittsburgh. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Approaching Intersection

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Intersection-Rules-Apply

[AGENT HISTORY]
- Vehicle (Impact:8) (Entered scene at t=1) [pos(t=49): far-left-rear | motion: stationary | interaction: low influence (parked)]: t=1~49 Stationary at (-5.6, 9.2)
[PREDICT]:
```

---

## Scene 2/5 — `1879fe1a-40a9-4f88-8bc2-f4f439261df1` (dataset index 3)

**Agents written:** 6  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1284/2000 |
| 1 | NBR1 | 32 | True | 194/2000 |
| 2 | NBR2 | 28 | True | 194/2000 |
| 3 | NBR3 | 27 | True | 195/2000 |
| 4 | NBR4 | 10 | True | 186/2000 |
| 5 | NBR5 | 30 | True | 194/2000 |

### Agent 0 — FOCAL (traj_id=0, 1284 tokens)

```text
Scenario '1879fe1a-40a9-4f88-8bc2-f4f439261df1' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line), Lane-Change-Allowed-Right

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-11.2,-0.3)|V:0.4m/s -> t=1:(-11.1,-0.3)|V:1.3m/s -> t=2:(-10.9,-0.3)|V:2.6m/s -> t=3:(-10.7,-0.3)|V:3.6m/s -> t=4:(-10.5,-0.3)|V:4.4m/s -> t=5:(-10.3,-0.3)|V:4.9m/s -> t=6:(-10.1,-0.3)|V:5.0m/s -> t=7:(-9.9,-0.3)|V:4.9m/s -> t=8:(-9.7,-0.2)|V:4.5m/s -> t=9:(-9.5,-0.2)|V:4.0m/s -> t=10:(-9.3,-0.2)|V:3.7m/s -> t=11:(-9.2,-0.2)|V:3.3m/s -> t=12:(-9.0,-0.2)|V:2.5m/s -> t=13:(-8.8,-0.2)|V:2.1m/s -> t=14:(-8.6,-0.2)|V:1.8m/s -> t=15:(-8.4,-0.2)|V:1.7m/s -> t=16:(-8.2,-0.2)|V:1.6m/s -> t=17:(-8.0,-0.2)|V:1.5m/s -> t=18:(-7.8,-0.2)|V:1.4m/s -> t=19:(-7.6,-0.2)|V:1.4m/s -> t=20:(-7.4,-0.2)|V:1.5m/s -> t=21:(-7.1,-0.2)|V:1.5m/s -> t=22:(-7.0,-0.2)|V:1.6m/s -> t=23:(-6.8,-0.2)|V:1.6m/s -> t=24:(-6.6,-0.2)|V:1.7m/s -> t=25:(-6.4,-0.2)|V:1.8m/s -> t=26:(-6.2,-0.2)|V:1.8m/s -> t=27:(-6.0,-0.2)|V:1.9m/s -> t=28:(-5.8,-0.2)|V:1.9m/s -> t=29:(-5.6,-0.2)|V:1.8m/s -> t=30:(-5.5,-0.2)|V:1.8m/s -> t=31:(-5.3,-0.2)|V:1.8m/s -> t=32:(-5.1,-0.2)|V:1.8m/s -> t=33:(-4.9,-0.2)|V:1.9m/s -> t=34:(-4.6,-0.2)|V:1.9m/s -> t=35:(-4.4,-0.2)|V:2.0m/s -> t=36:(-4.2,-0.2)|V:2.1m/s -> t=37:(-3.9,-0.2)|V:2.2m/s -> t=38:(-3.6,-0.2)|V:2.3m/s -> t=39:(-3.4,-0.2)|V:2.6m/s -> t=40:(-3.1,-0.2)|V:2.7m/s -> t=41:(-2.8,-0.2)|V:2.8m/s -> t=42:(-2.4,-0.1)|V:2.8m/s -> t=43:(-2.1,-0.1)|V:2.9m/s -> t=44:(-1.8,-0.1)|V:3.0m/s -> t=45:(-1.4,-0.1)|V:3.1m/s -> t=46:(-1.1,-0.1)|V:3.2m/s -> t=47:(-0.7,-0.1)|V:3.3m/s -> t=48:(-0.4,-0.0)|V:3.4m/s -> t=49:(0.0,0.0)|V:3.5m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=32, 194 tokens)

```text
Scenario '1879fe1a-40a9-4f88-8bc2-f4f439261df1' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line), Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:9) (Entered scene at t=45) [pos(t=49): left-front | motion: stationary | interaction: low influence (parked)]: t=45~49 Stationary at (4.2, 5.2)
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=28, 194 tokens)

```text
Scenario '1879fe1a-40a9-4f88-8bc2-f4f439261df1' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line), Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:8) (Entered scene at t=35) [pos(t=49): right-front | motion: stationary | interaction: low influence (parked)]: t=35~49 Stationary at (6.3, -4.6)
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=27, 195 tokens)

```text
Scenario '1879fe1a-40a9-4f88-8bc2-f4f439261df1' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line), Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:8) (Entered scene at t=34) [pos(t=49): left-rear | motion: stationary | interaction: low influence (parked)]: t=34~49 Stationary at (-5.9, 5.2)
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=10, 186 tokens)

```text
Scenario '1879fe1a-40a9-4f88-8bc2-f4f439261df1' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line), Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:8) [pos(t=49): far-left-rear | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (-2.3, 8.0)
[PREDICT]:
```

### Agent 5 — NBR5 (traj_id=30, 194 tokens)

```text
Scenario '1879fe1a-40a9-4f88-8bc2-f4f439261df1' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line), Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:8) (Entered scene at t=42) [pos(t=49): left-front | motion: stationary | interaction: low influence (parked)]: t=42~49 Stationary at (7.7, 3.5)
[PREDICT]:
```

---

## Scene 3/5 — `c79f1fd5-7ed4-4a3c-bff5-6a6e95028205` (dataset index 17)

**Agents written:** 3  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 141/2000 |
| 1 | NBR1 | 3 | True | 178/2000 |
| 2 | NBR2 | 5 | True | 178/2000 |
| 3 | EMPTY | — | False | 1/2000 |
| 4 | EMPTY | — | False | 1/2000 |
| 5 | EMPTY | — | False | 1/2000 |

### Agent 0 — FOCAL (traj_id=0, 141 tokens)

```text
Scenario 'c79f1fd5-7ed4-4a3c-bff5-6a6e95028205' in dearborn. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Standard Driving

[AFFORDANCE & INTENTION]
Focal Agent Intention: Stationary
Map Affordance: Lane-Change-Allowed-Right

[AGENT HISTORY]
* Focal Agent [Host]: t=0~49 Stationary at (0.0, 0.0)
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=3, 178 tokens)

```text
Scenario 'c79f1fd5-7ed4-4a3c-bff5-6a6e95028205' in dearborn. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Standard Driving

[AFFORDANCE & INTENTION]
Focal Agent Intention: Stationary
Map Affordance: Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:68) (Entered scene at t=13) [pos(t=49): lead | motion: stationary | interaction: high influence (direct path blockage)]: t=13~49 Stationary at (24.7, -0.3)
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=5, 178 tokens)

```text
Scenario 'c79f1fd5-7ed4-4a3c-bff5-6a6e95028205' in dearborn. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Standard Driving

[AFFORDANCE & INTENTION]
Focal Agent Intention: Stationary
Map Affordance: Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:7) (Entered scene at t=22) [pos(t=49): left-front | motion: stationary | interaction: low influence (parked)]: t=22~49 Stationary at (15.2, 2.8)
[PREDICT]:
```

---

## Scene 4/5 — `1224ba18-0080-4194-ac6d-dc1a0ba6717d` (dataset index 2)

**Agents written:** 6  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1306/2000 |
| 1 | NBR1 | 19 | True | 181/2000 |
| 2 | NBR2 | 20 | True | 184/2000 |
| 3 | NBR3 | 12 | True | 174/2000 |
| 4 | NBR4 | 11 | True | 174/2000 |
| 5 | NBR5 | 7 | True | 1384/2000 |

### Agent 0 — FOCAL (traj_id=0, 1306 tokens)

```text
Scenario '1224ba18-0080-4194-ac6d-dc1a0ba6717d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-45.4,0.3)|V:10.0m/s -> t=1:(-44.8,0.3)|V:10.0m/s -> t=2:(-44.1,0.4)|V:9.9m/s -> t=3:(-43.5,0.4)|V:9.9m/s -> t=4:(-42.8,0.5)|V:9.9m/s -> t=5:(-41.9,0.5)|V:9.9m/s -> t=6:(-41.0,0.5)|V:9.9m/s -> t=7:(-40.0,0.5)|V:9.8m/s -> t=8:(-38.9,0.5)|V:9.8m/s -> t=9:(-37.9,0.4)|V:9.8m/s -> t=10:(-36.8,0.4)|V:9.9m/s -> t=11:(-35.8,0.3)|V:9.9m/s -> t=12:(-34.7,0.2)|V:9.9m/s -> t=13:(-33.7,0.2)|V:9.8m/s -> t=14:(-32.6,0.2)|V:9.8m/s -> t=15:(-31.5,0.1)|V:9.8m/s -> t=16:(-30.4,0.1)|V:9.8m/s -> t=17:(-29.3,0.0)|V:9.8m/s -> t=18:(-28.2,0.0)|V:9.7m/s -> t=19:(-27.2,-0.0)|V:9.7m/s -> t=20:(-26.3,-0.0)|V:9.7m/s -> t=21:(-25.4,-0.0)|V:9.6m/s -> t=22:(-24.6,-0.0)|V:9.5m/s -> t=23:(-23.6,-0.1)|V:9.4m/s -> t=24:(-22.7,-0.1)|V:9.3m/s -> t=25:(-21.8,-0.1)|V:9.3m/s -> t=26:(-20.8,-0.1)|V:9.2m/s -> t=27:(-19.9,-0.1)|V:9.2m/s -> t=28:(-19.0,-0.1)|V:9.2m/s -> t=29:(-18.0,-0.1)|V:9.3m/s -> t=30:(-17.1,-0.1)|V:9.3m/s -> t=31:(-16.1,-0.1)|V:9.4m/s -> t=32:(-15.2,-0.1)|V:9.4m/s -> t=33:(-14.2,-0.1)|V:9.4m/s -> t=34:(-13.3,-0.1)|V:9.4m/s -> t=35:(-12.4,-0.1)|V:9.3m/s -> t=36:(-11.5,-0.1)|V:9.3m/s -> t=37:(-10.6,-0.0)|V:9.3m/s -> t=38:(-9.6,-0.0)|V:9.3m/s -> t=39:(-8.7,-0.0)|V:9.3m/s -> t=40:(-7.8,-0.0)|V:9.3m/s -> t=41:(-6.9,-0.0)|V:9.3m/s -> t=42:(-5.9,-0.0)|V:9.3m/s -> t=43:(-5.0,-0.0)|V:9.3m/s -> t=44:(-4.2,-0.0)|V:9.2m/s -> t=45:(-3.3,-0.0)|V:9.2m/s -> t=46:(-2.4,-0.0)|V:9.1m/s -> t=47:(-1.6,-0.0)|V:8.9m/s -> t=48:(-0.8,-0.0)|V:8.8m/s -> t=49:(0.0,0.0)|V:8.6m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=19, 181 tokens)

```text
Scenario '1224ba18-0080-4194-ac6d-dc1a0ba6717d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:9) (Entered scene at t=9) [pos(t=49): left-front | motion: stationary | interaction: low influence (parked)]: t=9~49 Stationary at (0.4, 4.3)
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=20, 184 tokens)

```text
Scenario '1224ba18-0080-4194-ac6d-dc1a0ba6717d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:7) (Entered scene at t=20) [pos(t=49): right-front | motion: stationary | interaction: low influence (parked)]: t=20~49 Stationary at (14.4, -4.1)
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=12, 174 tokens)

```text
Scenario '1224ba18-0080-4194-ac6d-dc1a0ba6717d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:7) [pos(t=49): right-rear | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (-16.2, -4.1)
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=11, 174 tokens)

```text
Scenario '1224ba18-0080-4194-ac6d-dc1a0ba6717d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:6) [pos(t=49): left-rear | motion: stationary | interaction: low influence (parked)]: t=0~49 Stationary at (-18.5, 3.3)
[PREDICT]:
```

### Agent 5 — NBR5 (traj_id=7, 1384 tokens)

```text
Scenario '1224ba18-0080-4194-ac6d-dc1a0ba6717d' in austin. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Restricted-Left (Solid Line)

[AGENT HISTORY]
- Vehicle (Impact:6) [pos(t=49): follower | motion: similar speed to ego | interaction: low influence]: t=0:(-72.9,0.2)|V:10.2m/s -> t=1:(-72.4,0.2)|V:10.2m/s -> t=2:(-71.8,0.2)|V:10.2m/s -> t=3:(-71.0,0.2)|V:10.1m/s -> t=4:(-70.2,0.2)|V:10.1m/s -> t=5:(-69.3,0.2)|V:10.0m/s -> t=6:(-68.3,0.2)|V:10.0m/s -> t=7:(-67.2,0.2)|V:10.0m/s -> t=8:(-66.1,0.2)|V:10.0m/s -> t=9:(-65.0,0.2)|V:10.0m/s -> t=10:(-64.0,0.2)|V:10.0m/s -> t=11:(-62.9,0.2)|V:10.1m/s -> t=12:(-61.9,0.2)|V:10.1m/s -> t=13:(-60.9,0.2)|V:10.1m/s -> t=14:(-59.9,0.2)|V:10.2m/s -> t=15:(-58.8,0.1)|V:10.2m/s -> t=16:(-57.8,0.1)|V:10.2m/s -> t=17:(-56.8,0.1)|V:10.2m/s -> t=18:(-55.7,0.1)|V:10.2m/s -> t=19:(-54.7,0.1)|V:10.1m/s -> t=20:(-53.7,0.1)|V:10.1m/s -> t=21:(-52.7,0.1)|V:10.0m/s -> t=22:(-51.7,0.1)|V:10.0m/s -> t=23:(-50.6,0.1)|V:9.9m/s -> t=24:(-49.6,0.1)|V:10.0m/s -> t=25:(-48.6,0.1)|V:10.0m/s -> t=26:(-47.6,0.1)|V:10.1m/s -> t=27:(-46.6,0.0)|V:10.2m/s -> t=28:(-45.5,0.0)|V:10.2m/s -> t=29:(-44.5,0.0)|V:10.3m/s -> t=30:(-43.5,-0.0)|V:10.3m/s -> t=31:(-42.5,-0.0)|V:10.4m/s -> t=32:(-41.5,-0.1)|V:10.4m/s -> t=33:(-40.5,-0.1)|V:10.3m/s -> t=34:(-39.5,-0.1)|V:10.3m/s -> t=35:(-38.5,-0.2)|V:10.3m/s -> t=36:(-37.5,-0.2)|V:10.2m/s -> t=37:(-36.5,-0.2)|V:10.2m/s -> t=38:(-35.5,-0.3)|V:10.1m/s -> t=39:(-34.5,-0.3)|V:10.0m/s -> t=40:(-33.5,-0.3)|V:9.9m/s -> t=41:(-32.5,-0.4)|V:9.9m/s -> t=42:(-31.5,-0.4)|V:9.9m/s -> t=43:(-30.5,-0.4)|V:9.9m/s -> t=44:(-29.5,-0.4)|V:10.0m/s -> t=45:(-28.4,-0.5)|V:10.0m/s -> t=46:(-27.4,-0.5)|V:10.0m/s -> t=47:(-26.4,-0.5)|V:10.1m/s -> t=48:(-25.4,-0.5)|V:10.1m/s -> t=49:(-24.4,-0.5)|V:10.1m/s
[PREDICT]:
```

---

## Scene 5/5 — `94c0f43a-407c-4289-8f1e-71384fe4564b` (dataset index 13)

**Agents written:** 5  |  **max_len_per_agent:** 2000

| Agent | Role | traj_id | valid | tokens used |
|-------|------|---------|-------|-------------|
| 0 | FOCAL | 0 | True | 1368/2000 |
| 1 | NBR1 | 1 | True | 1393/2000 |
| 2 | NBR2 | 5 | True | 195/2000 |
| 3 | NBR3 | 13 | True | 197/2000 |
| 4 | NBR4 | 18 | True | 197/2000 |
| 5 | EMPTY | — | False | 1/2000 |

### Agent 0 — FOCAL (traj_id=0, 1368 tokens)

```text
Scenario '94c0f43a-407c-4289-8f1e-71384fe4564b' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Allowed-Left, Lane-Change-Allowed-Right

[AGENT HISTORY]
* Focal Agent [Host]: t=0:(-68.7,3.3)|V:15.1m/s -> t=1:(-67.9,3.2)|V:15.1m/s -> t=2:(-67.0,3.2)|V:15.0m/s -> t=3:(-65.9,3.1)|V:15.0m/s -> t=4:(-64.7,3.0)|V:15.0m/s -> t=5:(-63.3,2.9)|V:15.0m/s -> t=6:(-61.8,2.8)|V:15.0m/s -> t=7:(-60.3,2.6)|V:15.0m/s -> t=8:(-58.7,2.5)|V:15.0m/s -> t=9:(-57.1,2.4)|V:15.0m/s -> t=10:(-55.5,2.2)|V:14.9m/s -> t=11:(-54.0,2.1)|V:14.9m/s -> t=12:(-52.5,2.0)|V:14.9m/s -> t=13:(-51.0,1.8)|V:14.8m/s -> t=14:(-49.5,1.7)|V:14.8m/s -> t=15:(-48.0,1.6)|V:14.8m/s -> t=16:(-46.5,1.5)|V:14.9m/s -> t=17:(-45.1,1.3)|V:14.9m/s -> t=18:(-43.6,1.2)|V:15.0m/s -> t=19:(-42.1,1.1)|V:15.0m/s -> t=20:(-40.6,1.0)|V:14.9m/s -> t=21:(-39.2,0.9)|V:14.9m/s -> t=22:(-37.7,0.8)|V:14.9m/s -> t=23:(-36.3,0.7)|V:14.8m/s -> t=24:(-34.8,0.6)|V:14.8m/s -> t=25:(-33.3,0.5)|V:14.7m/s -> t=26:(-31.9,0.4)|V:14.7m/s -> t=27:(-30.4,0.3)|V:14.6m/s -> t=28:(-29.0,0.2)|V:14.5m/s -> t=29:(-27.5,0.2)|V:14.5m/s -> t=30:(-26.1,0.1)|V:14.4m/s -> t=31:(-24.7,0.0)|V:14.3m/s -> t=32:(-23.3,-0.0)|V:14.3m/s -> t=33:(-21.8,-0.1)|V:14.2m/s -> t=34:(-20.4,-0.1)|V:14.1m/s -> t=35:(-19.0,-0.1)|V:14.0m/s -> t=36:(-17.6,-0.2)|V:14.0m/s -> t=37:(-16.2,-0.2)|V:13.9m/s -> t=38:(-14.8,-0.2)|V:13.8m/s -> t=39:(-13.4,-0.2)|V:13.8m/s -> t=40:(-12.1,-0.2)|V:13.7m/s -> t=41:(-10.7,-0.2)|V:13.7m/s -> t=42:(-9.3,-0.2)|V:13.7m/s -> t=43:(-8.0,-0.2)|V:13.6m/s -> t=44:(-6.6,-0.2)|V:13.5m/s -> t=45:(-5.3,-0.2)|V:13.4m/s -> t=46:(-3.9,-0.1)|V:13.3m/s -> t=47:(-2.6,-0.1)|V:13.2m/s -> t=48:(-1.3,-0.0)|V:13.1m/s -> t=49:(0.0,0.0)|V:13.0m/s
[PREDICT]:
```

### Agent 1 — NBR1 (traj_id=1, 1393 tokens)

```text
Scenario '94c0f43a-407c-4289-8f1e-71384fe4564b' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Allowed-Left, Lane-Change-Allowed-Right

[AGENT HISTORY]
- Autonomous Vehicle (Impact:85) [pos(t=49): right-front | motion: slower than ego | interaction: high conflict risk (TTC ~3.0s)]: t=0:(-60.0,-1.0)|V:6.9m/s -> t=1:(-59.3,-1.1)|V:6.9m/s -> t=2:(-58.5,-1.2)|V:13.9m/s -> t=3:(-57.5,-1.3)|V:14.0m/s -> t=4:(-56.3,-1.4)|V:14.0m/s -> t=5:(-55.0,-1.5)|V:14.0m/s -> t=6:(-53.6,-1.6)|V:14.1m/s -> t=7:(-52.1,-1.8)|V:14.2m/s -> t=8:(-50.6,-1.9)|V:14.1m/s -> t=9:(-49.1,-2.0)|V:14.1m/s -> t=10:(-47.7,-2.2)|V:14.1m/s -> t=11:(-46.2,-2.3)|V:14.2m/s -> t=12:(-44.8,-2.4)|V:14.1m/s -> t=13:(-43.4,-2.5)|V:14.2m/s -> t=14:(-42.0,-2.6)|V:14.3m/s -> t=15:(-40.6,-2.8)|V:14.4m/s -> t=16:(-39.1,-2.9)|V:14.3m/s -> t=17:(-37.7,-3.0)|V:14.1m/s -> t=18:(-36.3,-3.1)|V:14.3m/s -> t=19:(-34.9,-3.2)|V:14.2m/s -> t=20:(-33.5,-3.3)|V:14.1m/s -> t=21:(-32.1,-3.4)|V:14.0m/s -> t=22:(-30.7,-3.5)|V:14.1m/s -> t=23:(-29.3,-3.5)|V:14.1m/s -> t=24:(-27.9,-3.6)|V:14.0m/s -> t=25:(-26.5,-3.7)|V:13.9m/s -> t=26:(-25.1,-3.8)|V:13.9m/s -> t=27:(-23.7,-3.8)|V:13.8m/s -> t=28:(-22.3,-3.9)|V:13.8m/s -> t=29:(-21.0,-3.9)|V:13.7m/s -> t=30:(-19.6,-3.9)|V:13.5m/s -> t=31:(-18.2,-4.0)|V:13.5m/s -> t=32:(-16.9,-4.0)|V:13.5m/s -> t=33:(-15.5,-4.0)|V:13.5m/s -> t=34:(-14.2,-4.0)|V:13.4m/s -> t=35:(-12.9,-4.0)|V:13.4m/s -> t=36:(-11.5,-4.0)|V:13.3m/s -> t=37:(-10.2,-4.0)|V:13.3m/s -> t=38:(-8.9,-4.0)|V:13.3m/s -> t=39:(-7.5,-4.0)|V:13.3m/s -> t=40:(-6.2,-4.0)|V:13.4m/s -> t=41:(-4.9,-3.9)|V:13.4m/s -> t=42:(-3.5,-3.9)|V:13.3m/s -> t=43:(-2.2,-3.9)|V:13.4m/s -> t=44:(-0.9,-3.8)|V:13.4m/s -> t=45:(0.5,-3.8)|V:13.2m/s -> t=46:(1.8,-3.7)|V:13.2m/s -> t=47:(3.1,-3.6)|V:13.3m/s -> t=48:(4.5,-3.6)|V:13.4m/s -> t=49:(5.8,-3.5)|V:13.3m/s
[PREDICT]:
```

### Agent 2 — NBR2 (traj_id=5, 195 tokens)

```text
Scenario '94c0f43a-407c-4289-8f1e-71384fe4564b' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Allowed-Left, Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:7) (Entered scene at t=9) [pos(t=49): far-right-rear | motion: stationary | interaction: low influence (parked)]: t=9~49 Stationary at (-8.5, -10.8)
[PREDICT]:
```

### Agent 3 — NBR3 (traj_id=13, 197 tokens)

```text
Scenario '94c0f43a-407c-4289-8f1e-71384fe4564b' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Allowed-Left, Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:5) (Entered scene at t=33) [pos(t=49): far-left-front | motion: stationary | interaction: low influence (parked)]: t=33~49 Stationary at (13.9, 18.8)
[PREDICT]:
```

### Agent 4 — NBR4 (traj_id=18, 197 tokens)

```text
Scenario '94c0f43a-407c-4289-8f1e-71384fe4564b' in washington-dc. Task: High-Fidelity Motion Prediction.
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Lane-Change-Allowed-Left, Lane-Change-Allowed-Right

[AGENT HISTORY]
- Vehicle (Impact:5) (Entered scene at t=46) [pos(t=49): far-left-front | motion: stationary | interaction: low influence (parked)]: t=46~49 Stationary at (13.0, 19.8)
[PREDICT]:
```


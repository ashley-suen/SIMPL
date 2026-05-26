# Walkthrough: Advanced Semantic Prompt Engine Implementation

We have successfully overhauled the data preprocessing and prompt generation mechanism in `av2_llm_dataset.py`, transitioning it from a pure numerical parser to a highly cognitive, **Instance-Centric Semantic Engine**.

## 1. What was Implemented

Driven by your rigorous mathematical formulations and AV dataset expertise, the following core features are now actively running in our PyTorch Dataset:

### 🌟 Multi-Label Scenario Identification
The code extracts Focal Agent's past trajectories (`mean_heading_change`, `lateral_std`) and Lane Graph arrays (`intersect`, `lane_ctrs`) to apply dynamic Multi-Label thresholds.
- **Example Outputs observed in testing**: `[Straight Roadway, Stop-and-Go Platooning]`, `[Inside Intersection, Turning Scenario]`.

### 🌟 Focal Intention & Affordance
- Evaluates Map Constraints via `$t=49$` nearest-lane boundaries (`cross_left`, `cross_right`).
- Projects Affordance tags like `Left-Allow`, `Right-Allow` into the prompt.
- Estimates the implicit Focal Agent action (`Keep-Straight`, `Lane-Change`, `Stationary`).

### 🌟 Autonomous Spatial Relationships (Instance-Centric)
The "Polar Sector" approach was fully replaced.
- We constructed the logic to classify every valid Neighbor based on its relative transformed coordinates `(x, y)` and focal closure rates `dV`.
- Behaviors are strictly typed: `Lead Vehicle (closing)`, `Left Neighbor (moving away to left)`, `On-Collision Path Risk` (TTC calculated for intersections!).

## 2. Evidence of Success
When executing a test over one of the generated `.pkl` scenarios (`5d70...` in Pittsburgh), the new Prompt block successfully prints:

```text
[SCENARIO IDENTIFICATION]
Labels: Straight Roadway, Stop-and-Go Platooning

[AFFORDANCE & INTENTION]
Focal Agent Intention: Keep-Straight
Map Affordance: Slow-Allow, Deceleration-Permitted, Accelerate-Allow

[SPATIAL RELATIONS & HISTORY]
* Focal Agent [Host]: t=0:(-20.7,-0.3)|V:7.2m/s -> ...
- Neighbor 1 (score) [Left Neighbor at 3.9m lateral, 7.3m long.]: t=0:(6.7,3.8)|V:0.0m/s ...
- Neighbor 2 (score) [Lead Vehicle (9.1m ahead, closing)]: ...
```

## 3. Next Steps
> [!TIP]
> The Prompt Engine and the `llm_motion_model.py` (SmolLM Architecture) are now fully operational. The immediate next logical step is to fuse them into an end-to-end `train.py` script to verify gradients and loss backpropagation.

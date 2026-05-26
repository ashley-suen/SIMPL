# Semantic LLM Prompt Engine for Argoverse 2

## Goal
Implement a highly semantic, "human-like" prompt generation logic based on the user's advanced Instance-Centric mapping methodology. This will extract abstract concepts from raw trajectory matrices, transforming them into a semantic prompt that gives the LLM (SmolLM) a true "World Model" understanding before processing coordinates.

## Proposed Changes

### `av2_llm_dataset.py`

This file will undergo a major augmentation to include three powerful semantic extraction modules:

#### 1. Scenario Identification (Multi-label)
Instead of just location strings, output an array of macro-level scene labels.
*   **Logic additions**:
    *   Intersection Approaching/Inside: Computed by distance to nearest `intersect==1`.
    *   Straight Roadway: Threshold-based filtering (`lateral_std < 1.2`, `head_change < 0.1`, `v > 3`).
    *   Merging/Diverging: Comparing local `left` / `right` lane neighbor counts at $t=0$ vs $t=49$.
    *   Roadside / Cut-in: Identifying static-starts and neighbor dynamic lateral shifts.
    *   Platooning: Neighbor vehicle clusters in the `-1.8m < Y < 1.8m` corridor with low speed.

#### 2. Spatial Relations (Instance-centric)
Instead of compass-directions, assign intelligent roles to neighbors.
*   **Logic additions**:
    *   Convert neighbors to focal-centric coordinates $(x,y)$.
    *   `Lead/Following Vehicle`: Assign based on same-lane corridor (`|Y| < 1.8m`) and sign of $X$.
    *   `Left/Right Neighbor`: Assign based on adjacent corridors. Check if $V_y$ is dragging them heavily towards `Y=0` for the `Merging` suffix.
    *   `On-Collision Path`: Simplified Time-To-Collision/Geometric crossing check for intersection scenarios.

#### 3. Intention & Affordance
Add explicit labels to the Focal Agent stating what it is doing and what the map allows.
*   **Logic additions**:
    *   **Intention**: Extracted strictly from past 5 seconds of the focal trajectory (Stationary, Straight, Left/Right Turn, U-Turn, Lane Changes).
    *   **Affordance**: Evaluating map rules based on the immediate Host Lane (`cross_left / cross_right`) to inject tags like `Left-Allow`, `Right-Allow`.

#### Prompt Output Restructuring
The generated prompt will change its structure substantially:
```text
[SCENARIO IDENTIFICATION]
Location: Miami | Labels: [Inside Intersection, Platooning]

[AFFORDANCE & INTENTION]
Focal Intention: Straight
Map Affordance: [Accelerate-Allow, Left-Allow]

[SPATIAL RELATIONS & HISTORY]
- Lead Vehicle (10m ahead): t=... -> t=...
- Right Neighbor (merging in): t=... -> t=...
- On-Collision Path Hazard!: t=... -> t=...

[PREDICTION]
...
```

## User Review Required
> [!IMPORTANT]
> The thresholds for these derivations (e.g., Lane width = 3.6m so `1.8m` radius for "Same lane"; Turning threshold = 0.2 rad) are tuned based on empirical defaults. Please confirm if you accept the thresholds proposed in your design discussions or if you want specific custom bounds.

## Verification Plan
1. Override `generate_prompt()` in a scratch test script.
2. Run against the 20 preprocessed training scenes.
3. Print out the prompt text to verify the multi-label scenario tags and role classifications.
4. Verify edge cases (e.g. is U-turn correctly not triggered on a straight road).

"""
AV2 Hybrid Dataset — returns compact semantic text tokens + numerical embeddings.

Text portion  : scene labels + focal intention + [MAP CONTEXT]
                (non-redundant map semantics) + focal dynamics +
                surrounding agents context (all semantic, NO coordinate steps)
Numerical portion:
  agent_features [N, T_obs, 13]  = (x,y,vx,vy,cos_θ,sin_θ) + type_one_hot(7)
  lane_features  [L, 10,    8]   = per-point (x,y,dx,dy,is_intersect,type_0,type_1,type_2)
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AV2HybridDataset(Dataset):
    def __init__(self,
                 data_dir,
                 tokenizer_name="Qwen/Qwen3-0.6B-Base",
                 obs_len=50,
                 max_agents=6,
                 max_lanes=20,
                 max_text_len=256):

        self.data_dir     = data_dir
        self.obs_len      = obs_len
        self.max_agents   = max_agents
        self.max_lanes    = max_lanes
        self.max_text_len = max_text_len

        self.file_list = (
            glob.glob(os.path.join(data_dir, "*.pkl")) or
            glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
        )
        if not self.file_list:
            raise FileNotFoundError(f"No .pkl files found under {data_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── helpers (reused from av2_llm_dataset) ────────────────────────────────

    def _extract_focal_kinematics(self, trajs_pos, trajs_vel, has_flags):
        obs_mask = has_flags[0, :self.obs_len].astype(bool)
        if obs_mask.sum() < 5:
            return {"speed_mean": 0, "lateral_std": 0, "heading_change": 0,
                    "lateral_displacement": 0, "last_speed": 0, "first_speed": 0,
                    "valid_frames": 0}
        f_pos = trajs_pos[0, :self.obs_len][obs_mask]
        f_vel = trajs_vel[0, :self.obs_len][obs_mask]
        all_speeds = np.linalg.norm(f_vel, axis=1)
        valid_speeds = all_speeds[all_speeds > 0.1]
        speed_mean   = valid_speeds.mean() if len(valid_speeds) > 0 else 0
        first_speed  = valid_speeds[0]     if len(valid_speeds) > 0 else 0
        last_speed   = all_speeds[-1]
        lateral_std  = np.std(f_pos[:, 1])
        lateral_disp = f_pos[-1, 1] - f_pos[0, 1]
        valid_vel_mask = all_speeds > 0.1
        if valid_vel_mask.sum() >= 2:
            va = f_vel[valid_vel_mask]
            angles = np.unwrap(np.arctan2(va[:, 1], va[:, 0]))
            heading_change = angles[-1] - angles[0]
        else:
            heading_change = 0.0
        return {"speed_mean": speed_mean, "lateral_std": lateral_std,
                "heading_change": heading_change, "lateral_displacement": lateral_disp,
                "last_speed": last_speed, "first_speed": first_speed,
                "valid_frames": obs_mask.sum()}

    def _identify_scenarios_and_intention(self, fk, lane_graph, neighbors_info):
        labels = []
        is_intersect = lane_graph["intersect"][:, 0] == 1
        int_lanes = lane_graph["lane_ctrs"][is_intersect]
        dists_to_int = np.linalg.norm(int_lanes, axis=1) if len(int_lanes) > 0 else [999]
        min_d = np.min(dists_to_int)
        at_intersection = False
        if min_d < 2.0:
            labels.append("Inside Intersection"); at_intersection = True
        elif min_d <= 15.0:
            labels.append("Approaching Intersection"); at_intersection = True
        if (not at_intersection and fk["lateral_std"] < 1.2 and
                abs(fk["heading_change"]) < 0.1 and fk["speed_mean"] > 1.0 and
                abs(fk["lateral_displacement"]) < 1.0 and fk["valid_frames"] > 20):
            labels.append("Straight Roadway")
        if abs(fk["heading_change"]) > 0.2:
            labels.append("Turning Scenario")
        if abs(fk["lateral_displacement"]) > 1.5 and abs(fk["heading_change"]) < 0.15:
            labels.append("Lane Change Behavior")
        same_lane = [n for n in neighbors_info
                     if abs(n["curr_pos"][1]) < 1.8 and n["curr_pos"][0] > 0]
        if len(same_lane) >= 2 and np.mean([n["curr_speed"] for n in same_lane]) < 3.0:
            labels.append("Stop-and-Go Platooning")

        intention = "Keep-Straight"
        if fk["last_speed"] < 0.5:
            intention = "Stopping" if fk["first_speed"] > 1.0 else "Stationary"
        elif abs(fk["lateral_displacement"]) > 1.5 and abs(fk["heading_change"]) < 0.15:
            intention = "Lane-Change-Left" if fk["lateral_displacement"] > 0 else "Lane-Change-Right"
        elif abs(fk["heading_change"]) > 0.2:
            intention = "Left-Turn" if fk["heading_change"] > 0 else "Right-Turn"
            if abs(fk["heading_change"]) > np.pi * 0.7:
                intention = "U-Turn"

        dists    = np.linalg.norm(lane_graph["lane_ctrs"], axis=1)
        host_idx = np.argmin(dists)
        c_left   = np.argmax(lane_graph["cross_left"][host_idx, 0, :])
        c_right  = np.argmax(lane_graph["cross_right"][host_idx, 0, :])
        affordances = []
        if at_intersection:
            affordances.append("Intersection-Rules-Apply")
        if c_left == 0 and lane_graph["left"][host_idx, 0] == 1:
            affordances.append("Lane-Change-Allowed-Left")
        elif lane_graph["left"][host_idx, 0] == 1:
            affordances.append("Lane-Change-Restricted-Left (Solid Line)")
        if c_right == 0 and lane_graph["right"][host_idx, 0] == 1:
            affordances.append("Lane-Change-Allowed-Right")
        elif lane_graph["right"][host_idx, 0] == 1:
            affordances.append("Lane-Change-Restricted-Right (Solid Line)")
        if not affordances:
            affordances.append("Map Constraints: None")
        if not labels:
            labels.append("Standard Driving")
        return labels, intention, affordances

    def _extract_spatial_relations(self, n, focal_speed):
        x, y   = n["curr_pos"]
        vx, vy = n["curr_vel"]
        n_speed = n["curr_speed"]
        dist    = np.hypot(x, y)
        if abs(y) < 1.8:   lat_pos = "same-lane"
        elif 1.8<=y<=5.4:  lat_pos = "left"
        elif -5.4<=y<=-1.8:lat_pos = "right"
        elif y > 5.4:      lat_pos = "far-left"
        else:              lat_pos = "far-right"
        lon_pos = "front" if x > 0 else "rear"
        if lat_pos == "same-lane":
            position = "lead" if x > 0 else "follower"
        else:
            position = f"{lat_pos}-{lon_pos}"
        vels = np.linalg.norm(n["vel_hist"][n["flags"]], axis=1) if n["flags"].any() else [0]
        max_speed  = np.max(vels) if len(vels) > 0 else 0
        last_speed = vels[-1]    if len(vels) > 0 else 0
        first_speed= ([v for v in vels if v > 0.1] or [0])[0]
        rel_v = last_speed - focal_speed
        is_stationary = max_speed < 0.5
        if is_stationary:
            motion = "stationary"
        elif last_speed < 0.5 and first_speed > 1.0:
            motion = "stopping"
        elif last_speed > 1.0 and first_speed < 0.5:
            motion = "accelerating from stop"
        else:
            motion = ("faster than ego" if rel_v > 1.0 else
                      "slower than ego" if rel_v < -1.0 else "similar speed to ego")
        if not is_stationary and last_speed > 0.5:
            valid_pos = n["pos_hist"][n["flags"]]
            lat_disp  = valid_pos[-1][1] - valid_pos[0][1] if len(valid_pos) > 1 else 0
            CUT = 7.0
            if "left" in lat_pos and lat_disp < -1.5 and abs(y) < CUT:
                motion += ", cutting in towards ego"
            elif "right" in lat_pos and lat_disp > 1.5 and abs(y) < CUT:
                motion += ", cutting in towards ego"
            elif abs(y) >= 1.8 and abs(y) < CUT and abs(lat_disp) > 1.5 and (y*lat_disp > 0):
                motion += ", moving laterally away"
        min_cpa = 999.0; ttc = 999.0
        if not is_stationary:
            for t_sim in np.arange(0, 5.0, 0.5):
                ego_x = focal_speed * t_sim
                ag_x  = x + vx * t_sim; ag_y = y + vy * t_sim
                d = np.hypot(ego_x - ag_x, ag_y)
                if d < min_cpa: min_cpa = d
                if d < 1.5: ttc = min(ttc, t_sim)
        else:
            if x > 0:
                min_cpa = abs(y)
                ttc = x / focal_speed if focal_speed > 0.5 else 999
            else:
                min_cpa = dist
        conflict_risk = (ttc < 4.0) and (min_cpa < 4.0)
        influence_score = 0
        if conflict_risk:
            interaction = f"high conflict risk (TTC ~{ttc:.1f}s)"
            influence_score = 100 - ttc * 5
        elif position == "lead" and dist < 25.0:
            interaction = "high influence (direct path blockage)"
            influence_score = 80 - dist / 2.0
        elif "cutting in" in motion and dist < 25.0:
            interaction = "high influence (cut-in hazard)"
            influence_score = 75 - dist / 2.0
        elif position in ["left-front","right-front"] and dist < 20.0 and min_cpa < 8.0 and not is_stationary:
            interaction = "moderate influence (close proximity)"
            influence_score = 50 - dist
        elif position == "follower" and dist < 15.0 and rel_v > 1.0:
            interaction = "moderate influence (approaching fast from rear)"
            influence_score = 45
        elif is_stationary:
            interaction = "low influence (parked)"
            influence_score = max(0, 10 - dist/5.0)
        else:
            interaction = "low influence"
            influence_score = max(0, 30 - dist)
        semantic_role = f"pos(t=49): {position} | motion: {motion} | interaction: {interaction}"
        return semantic_role, influence_score, is_stationary

    @staticmethod
    def _cat_name(cat_id):
        if isinstance(cat_id, str):
            if cat_id == 'av':    return "Autonomous Vehicle"
            if cat_id == 'focal': return "Focal Vehicle"
        return "Vehicle"

    # ── Scene text (all semantic, no coordinate steps) ───────────────────────

    def compose_scene_text(self, df_row):
        """
        Compose the semantic scene text (no tokenization):
          scene header + labels + [FOCAL INTENTION] + [MAP CONTEXT]
          (non-redundant map semantics) + focal dynamics + surrounding agents.
        Map Affordance is dropped (redundant with [MAP CONTEXT]); intention kept.
        Stops at [AGENT TRAJECTORIES] header. NO coordinate step tokens.
        Returns (text, selected_neighbor_ids).
        """
        seq_id    = df_row["SEQ_ID"]
        city_name = df_row["CITY_NAME"]
        trajs     = df_row["TRAJS"]
        lane_graph = df_row["LANE_GRAPH"]

        trajs_pos = trajs["trajs_pos"]
        trajs_vel = trajs["trajs_vel"]
        has_flags = trajs["has_flags"]
        trajs_cat = trajs["trajs_cat"]

        fk = self._extract_focal_kinematics(trajs_pos, trajs_vel, has_flags)

        # Collect valid neighbors (same logic as existing dataset)
        neighbors_info = []
        trajs_ctrs = trajs["trajs_ctrs"]
        trajs_vecs = trajs["trajs_vecs"]
        for i in range(1, len(trajs_cat)):
            if not has_flags[i, self.obs_len - 1]:
                continue
            ctr, vec  = trajs_ctrs[i], trajs_vecs[i]
            theta     = np.arctan2(vec[1], vec[0])
            rot_mat   = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]])
            pos_focal  = trajs_pos[i, :self.obs_len, :].dot(rot_mat.T) + ctr
            vel_focal  = trajs_vel[i, :self.obs_len, :].dot(rot_mat.T)
            curr_pos   = pos_focal[-1]
            curr_vel   = vel_focal[-1]
            curr_speed = np.linalg.norm(curr_vel)
            raw_n = {"curr_pos": curr_pos, "curr_vel": curr_vel,
                     "curr_speed": curr_speed, "vel_hist": vel_focal,
                     "pos_hist": pos_focal, "flags": has_flags[i, :self.obs_len].astype(bool)}
            semantic_role, influence_score, is_stationary = \
                self._extract_spatial_relations(raw_n, fk["last_speed"])
            if influence_score <= 0 and "low influence" in semantic_role:
                continue
            neighbors_info.append({
                "id": i, "cat": trajs_cat[i],
                "semantic_role": semantic_role,
                "influence_score": influence_score,
                "curr_pos": curr_pos,
                "curr_speed": curr_speed,
            })
        neighbors_info.sort(key=lambda x: x["influence_score"], reverse=True)

        labels, intention, _affordances = \
            self._identify_scenarios_and_intention(fk, lane_graph, neighbors_info)

        # Focal dynamics
        sd = fk["last_speed"] - fk["first_speed"]
        speed_trend = ("accelerating" if sd > 0.5 else
                       "decelerating" if sd < -0.5 else "constant speed")
        hc = fk["heading_change"]
        heading_desc = ("turning left"  if hc > 0.3 else
                        "turning right" if hc < -0.3 else
                        "slight curve"  if abs(hc) > 0.1 else "straight ahead")

        text  = f"Scenario '{seq_id}' in {city_name}. Task: High-Fidelity Motion Prediction.\n"
        text += f"[SCENARIO IDENTIFICATION]\nLabels: {', '.join(labels)}\n\n"
        # Map Affordance dropped: now redundant with [MAP CONTEXT] (lane boundaries,
        # intersection, crossability). Intention (from focal kinematics) is kept.
        text += f"[FOCAL INTENTION]\nFocal Agent Intention: {intention}\n\n"
        # [MAP CONTEXT]: non-redundant map semantics extracted at preprocess time.
        # Old pkls without this field degrade gracefully (block omitted).
        map_ctx = df_row.get("MAP_CONTEXT", "") if hasattr(df_row, "get") else ""
        if isinstance(map_ctx, str) and map_ctx.strip():
            text += f"[MAP CONTEXT]\n{map_ctx}\n\n"
        text += (f"[FOCAL DYNAMICS]\n"
                 f"Speed: {fk['first_speed']:.1f}→{fk['last_speed']:.1f}m/s "
                 f"({speed_trend}) | Heading: {heading_desc}\n\n")

        # Surrounding agents context (up to max_agents-1 neighbors)
        selected = neighbors_info[:self.max_agents - 1]
        if selected:
            text += "[SURROUNDING AGENTS CONTEXT]\n"
            for k, n in enumerate(selected, 1):
                sr_parts     = n["semantic_role"].split(" | ")
                position_str = sr_parts[0].replace("pos(t=49): ", "") if sr_parts else "unknown"
                interact_str = sr_parts[2].replace("interaction: ", "") if len(sr_parts) > 2 else "unknown"
                dist_m       = np.hypot(n["curr_pos"][0], n["curr_pos"][1])
                rel_v        = n["curr_speed"] - fk["last_speed"]
                if rel_v > 0.5:
                    rel_v_desc = f"+{rel_v:.1f}m/s (faster than focal, approaching if ahead)"
                elif rel_v < -0.5:
                    rel_v_desc = f"{rel_v:.1f}m/s (slower than focal)"
                else:
                    rel_v_desc = f"similar speed ({n['curr_speed']:.1f}m/s)"
                agent_type = self._cat_name(n["cat"])
                text += (f"  Agent{k} ({agent_type}): "
                         f"Distance {dist_m:.1f}m | {position_str} | "
                         f"Speed {rel_v_desc} | {interact_str}\n")
            text += "\n"

        text += "[AGENT TRAJECTORIES]\n"

        return text, [n["id"] for n in selected]  # neighbor ids for ordering

    def _build_scene_text(self, df_row):
        """Tokenize the composed scene text (thin wrapper over compose_scene_text)."""
        text, selected_ids = self.compose_scene_text(df_row)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_text_len,
                             padding="max_length", return_tensors="pt")
        return (enc["input_ids"].squeeze(0),
                enc["attention_mask"].squeeze(0),
                selected_ids)

    # ── Numerical features ────────────────────────────────────────────────────

    def _get_agent_features(self, df_row, neighbor_ids):
        """
        Returns:
          features [max_agents, obs_len, 13]  float32
            = (x, y, vx, vy, cos_θ, sin_θ, type_0..6)
          valid    [max_agents]                bool
          train_mask [max_agents]              bool
        All coordinates in focal-agent frame (trajs_pos is already normalized).
        """
        trajs     = df_row["TRAJS"]
        trajs_pos = trajs["trajs_pos"]     # [N, 110, 2] already in focal frame
        trajs_vel = trajs["trajs_vel"]     # [N, 110, 2]
        trajs_ang = trajs["trajs_ang"]     # [N, 110]
        trajs_type = trajs["trajs_type"]   # [N, 110, 7]  one-hot
        has_flags  = trajs["has_flags"]    # [N, 110]
        trajs_cat  = trajs["trajs_cat"]

        # Agent order: [focal(0), neighbor_ids...]
        agent_ids = [0] + list(neighbor_ids)
        # Pad to max_agents
        while len(agent_ids) < self.max_agents:
            agent_ids.append(-1)

        features   = np.zeros((self.max_agents, self.obs_len, 13), dtype=np.float32)
        valid      = np.zeros(self.max_agents, dtype=bool)
        train_mask = np.zeros(self.max_agents, dtype=bool)

        for slot, ag_id in enumerate(agent_ids):
            if ag_id < 0 or ag_id >= len(trajs_cat):
                continue
            pos  = trajs_pos[ag_id, :self.obs_len]       # [T, 2]
            vel  = trajs_vel[ag_id, :self.obs_len]        # [T, 2]
            ang  = trajs_ang[ag_id, :self.obs_len]        # [T]
            typ  = trajs_type[ag_id, :self.obs_len].astype(np.float32)  # [T, 7]
            flag = has_flags[ag_id, :self.obs_len].astype(bool)         # [T]

            feat = np.zeros((self.obs_len, 13), dtype=np.float32)
            feat[:, 0:2] = pos
            feat[:, 2:4] = vel
            feat[:, 4]   = np.cos(ang)
            feat[:, 5]   = np.sin(ang)
            feat[:, 6:13] = typ
            # Zero out invalid frames
            feat[~flag]  = 0.0

            features[slot]   = feat
            valid[slot]      = True
            cat = str(trajs_cat[ag_id]).lower()
            train_mask[slot] = cat in ('focal', 'score')

        return (torch.from_numpy(features),
                torch.from_numpy(valid),
                torch.from_numpy(train_mask))

    def _get_lane_features(self, df_row):
        """
        Returns:
          features [max_lanes, 10, 8]  float32
            = (x, y, dx, dy, is_intersect, lane_type_0, lane_type_1, lane_type_2) per point
          valid    [max_lanes]          bool
        Selects top-max_lanes closest lanes by lane center distance.
        node_ctrs already in focal-agent frame.
        """
        lg         = df_row["LANE_GRAPH"]
        node_ctrs  = lg["node_ctrs"]    # [L, 10, 2]
        node_vecs  = lg["node_vecs"]    # [L, 10, 2]
        lane_ctrs  = lg["lane_ctrs"]    # [L, 2]
        intersect  = lg["intersect"]    # [L, 10]
        lane_type  = lg["lane_type"]    # [L, 10, 3]

        L = node_ctrs.shape[0]
        dists = np.linalg.norm(lane_ctrs, axis=1)   # [L]
        top_idx = np.argsort(dists)[:self.max_lanes] # take closest

        features = np.zeros((self.max_lanes, 10, 8), dtype=np.float32)
        valid    = np.zeros(self.max_lanes, dtype=bool)

        for slot, idx in enumerate(top_idx):
            if idx >= L:
                break
            pts     = node_ctrs[idx]                              # [10, 2]
            dirs    = node_vecs[idx]                              # [10, 2]
            intsect = intersect[idx].astype(np.float32)          # [10]
            lt      = lane_type[idx].astype(np.float32)          # [10, 3]

            f = np.zeros((10, 8), dtype=np.float32)
            f[:, 0:2] = pts
            f[:, 2:4] = dirs
            f[:, 4]   = intsect
            f[:, 5:8] = lt
            features[slot] = f
            valid[slot]    = True

        return torch.from_numpy(features), torch.from_numpy(valid)

    # ── GT trajectories ───────────────────────────────────────────────────────

    def _get_gt(self, df_row, neighbor_ids):
        trajs      = df_row["TRAJS"]
        trajs_pos  = trajs["trajs_pos"]
        has_flags  = trajs["has_flags"]
        trajs_cat  = trajs["trajs_cat"]

        agent_ids = [0] + list(neighbor_ids)
        while len(agent_ids) < self.max_agents:
            agent_ids.append(-1)

        gt_abs    = torch.zeros(self.max_agents, 60, 2)
        gt_anchor = torch.zeros(self.max_agents, 2)
        gt_masks  = torch.zeros(self.max_agents, 60, dtype=torch.bool)

        for slot, ag_id in enumerate(agent_ids):
            if ag_id < 0 or ag_id >= len(trajs_cat):
                continue
            abs_pos = torch.tensor(trajs_pos[ag_id, self.obs_len:], dtype=torch.float32)
            flags   = torch.tensor(has_flags[ag_id, self.obs_len:], dtype=torch.bool)

            anchor_idx = self.obs_len - 1
            while anchor_idx > 0 and not has_flags[ag_id, anchor_idx]:
                anchor_idx -= 1
            anchor = torch.tensor(trajs_pos[ag_id, anchor_idx], dtype=torch.float32)

            gt_abs[slot]    = abs_pos
            gt_anchor[slot] = anchor
            gt_masks[slot]  = flags

        return gt_abs, gt_anchor, gt_masks

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        row = pd.read_pickle(self.file_list[idx]).iloc[0]

        text_ids, text_mask, neighbor_ids = self._build_scene_text(row)
        agent_features, agent_valid, train_mask = self._get_agent_features(row, neighbor_ids)
        lane_features,  lane_valid              = self._get_lane_features(row)
        gt_abs, gt_anchor, gt_masks             = self._get_gt(row, neighbor_ids)

        return {
            # Text (semantic)
            "text_input_ids":      text_ids,        # [L_text]
            "text_attention_mask": text_mask,        # [L_text]
            # Numerical agent features
            "agent_features":      agent_features,   # [N, T_obs, 13]
            "agent_valid":         agent_valid,       # [N]
            "train_mask":          train_mask,        # [N]
            # Numerical lane features
            "lane_features":       lane_features,     # [max_lanes, 10, 8]
            "lane_valid":          lane_valid,         # [max_lanes]
            # GT
            "gt_abs_trajectories": gt_abs,            # [N, 60, 2]
            "gt_anchor":           gt_anchor,          # [N, 2]
            "gt_masks":            gt_masks,            # [N, 60]
        }

    def __len__(self):
        return len(self.file_list)

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class AV2PromptDataset(Dataset):
    """
    A PyTorch Dataset that loads Argoverse 2 .pkl features, 
    and converts the numerical trajectory and lane graph data into 
    a highly semantic, instance-centric Language Model Prompt.
    """
    def __init__(self, data_dir, tokenizer_name="Qwen/Qwen3-0.6B-Base", obs_len=50, max_neighbors=5, max_lanes=10, traj_step=1, max_len_per_agent=8192):
        super().__init__()
        self.data_dir = data_dir
        # Search recursively to support both flat and subdirectory layouts
        self.file_list = glob.glob(os.path.join(data_dir, "*.pkl")) or \
                         glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.obs_len = obs_len
        self.max_neighbors = max_neighbors
        self.max_lanes = max_lanes
        self.traj_step = traj_step
        self.max_len_per_agent = max_len_per_agent

        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .pkl files found under {data_dir}. "
                                    f"Check the path and directory structure.")

    def __len__(self):
        return len(self.file_list)
        
    def _extract_focal_kinematics(self, trajs_pos, trajs_vel, has_flags, lane_graph):
        """Extracts history kinematics of the Focal agent to establish intentions and dynamics."""
        obs_mask = has_flags[0, :self.obs_len]
        if obs_mask.sum() < 5:
            return {"speed_mean": 0, "lateral_std": 0, "heading_change": 0, "lateral_displacement": 0, "last_speed": 0, "first_speed": 0}
            
        f_pos = trajs_pos[0, :self.obs_len][obs_mask]
        f_vel = trajs_vel[0, :self.obs_len][obs_mask]
        
        all_speeds = np.linalg.norm(f_vel, axis=1)
        valid_speeds = all_speeds[all_speeds > 0.1]  # Filter AV2 t=0 zero-padding
        speed_mean = valid_speeds.mean() if len(valid_speeds) > 0 else 0
        first_speed = valid_speeds[0] if len(valid_speeds) > 0 else 0
        last_speed = all_speeds[-1]  # Last observed speed (unfiltered is correct here)

        lateral_std = np.std(f_pos[:, 1])
        lateral_displacement = f_pos[-1, 1] - f_pos[0, 1]

        # Compute heading change only on frames with valid motion (avoid arctan2(0,0) noise)
        valid_vel_mask = all_speeds > 0.1
        if valid_vel_mask.sum() >= 2:
            valid_f_vel = f_vel[valid_vel_mask]
            angles = np.unwrap(np.arctan2(valid_f_vel[:, 1], valid_f_vel[:, 0]))
            heading_change = angles[-1] - angles[0]
        else:
            heading_change = 0.0

        return {
            "speed_mean": speed_mean,
            "lateral_std": lateral_std,
            "heading_change": heading_change,
            "lateral_displacement": lateral_displacement,
            "first_speed": first_speed,
            "last_speed": last_speed,
            "valid_frames": obs_mask.sum()
        }

    def _identify_scenarios_and_intention(self, fk, lane_graph, neighbors_info):
        """Phase 1 & 3: Identifies Global Scenarios and Focal Intention / Affordance."""
        labels = []
        is_intersect = lane_graph["intersect"][:, 0] == 1
        int_lanes = lane_graph["lane_ctrs"][is_intersect]
        dists_to_int = np.linalg.norm(int_lanes, axis=1) if len(int_lanes) > 0 else [999]
        min_d = np.min(dists_to_int)
        
        at_intersection = False
        if min_d < 2.0:
            labels.append("Inside Intersection")
            at_intersection = True
        elif min_d <= 15.0:
            labels.append("Approaching Intersection")
            at_intersection = True

        # Defect 6 & Bug 3: Fix intention thresholds & missing frames
        if (not at_intersection and fk["lateral_std"] < 1.2
            and abs(fk["heading_change"]) < 0.1 and fk["speed_mean"] > 1.0 
            and abs(fk["lateral_displacement"]) < 1.0 and fk["valid_frames"] > 20):
            labels.append("Straight Roadway")
            
        if abs(fk["heading_change"]) > 0.2:
            labels.append("Turning Scenario")
            
        if abs(fk["lateral_displacement"]) > 1.5 and abs(fk["heading_change"]) < 0.15:
            labels.append("Lane Change Behavior")
            
        same_lane_neighbors = [n for n in neighbors_info if abs(n["curr_pos"][1]) < 1.8 and n["curr_pos"][0] > 0]
        if len(same_lane_neighbors) >= 2:
            speeds = [n["curr_speed"] for n in same_lane_neighbors]
            if np.mean(speeds) < 3.0:
                labels.append("Stop-and-Go Platooning")
                
        # --- Focal Intention ---
        intention = "Keep-Straight"
        # Defect 6: Add 'Stopping' intention
        if fk["last_speed"] < 0.5: 
            if fk["first_speed"] > 1.0:
                intention = "Stopping"
            else:
                intention = "Stationary"
        elif abs(fk["lateral_displacement"]) > 1.5 and abs(fk["heading_change"]) < 0.15:
            intention = "Lane-Change-Left" if fk["lateral_displacement"] > 0 else "Lane-Change-Right"
        elif abs(fk["heading_change"]) > 0.2:
            intention = "Left-Turn" if fk["heading_change"] > 0 else "Right-Turn"
            if abs(fk["heading_change"]) > np.pi * 0.7: intention = "U-Turn"
            
        # --- Host Affordance --- (Defect 3: Removed hallucinated 'Accelerate-Allow' filler)
        dists = np.linalg.norm(lane_graph["lane_ctrs"], axis=1)
        host_idx = np.argmin(dists)
        c_left = np.argmax(lane_graph["cross_left"][host_idx, 0, :])
        c_right = np.argmax(lane_graph["cross_right"][host_idx, 0, :])
        
        affordances = []
        if at_intersection: affordances.append("Intersection-Rules-Apply")
        
        if c_left == 0 and lane_graph["left"][host_idx, 0] == 1: affordances.append("Lane-Change-Allowed-Left")
        elif lane_graph["left"][host_idx, 0] == 1: affordances.append("Lane-Change-Restricted-Left (Solid Line)")
        
        if c_right == 0 and lane_graph["right"][host_idx, 0] == 1: affordances.append("Lane-Change-Allowed-Right")
        elif lane_graph["right"][host_idx, 0] == 1: affordances.append("Lane-Change-Restricted-Right (Solid Line)")
        
        if not affordances: affordances.append("Map Constraints: None")
        if not labels: labels.append("Standard Driving")
        
        return labels, intention, affordances

    def _extract_spatial_relations(self, n, focal_speed):
        """Phase 2: Extracts instance-centric semantic roles returning dict with score."""
        x, y = n["curr_pos"]
        vx, vy = n["curr_vel"]
        n_speed = n["curr_speed"]
        dist = np.hypot(x, y)
        
        # --- 1. Position Abstraction (Defect 2: Note this is t=49, +Y is Left) ---
        if abs(y) < 1.8: lat_pos = "same-lane"
        elif 1.8 <= y <= 5.4: lat_pos = "left"
        elif -5.4 <= y <= -1.8: lat_pos = "right"
        elif y > 5.4: lat_pos = "far-left"
        else: lat_pos = "far-right"
        
        lon_pos = "front" if x > 0 else "rear"
        
        if lat_pos == "same-lane":
            position = "lead" if x > 0 else "follower"
        else:
            position = f"{lat_pos}-{lon_pos}"
            
        # --- 2. Motion Abstraction ---
        vels = np.linalg.norm(n["vel_hist"][n["flags"]], axis=1) if n["flags"].any() else [0]
        valid_vels = vels[vels > 0.1]
        max_speed = np.max(vels) if len(vels) > 0 else 0
        first_speed = valid_vels[0] if len(valid_vels) > 0 else 0
        last_speed = vels[-1] if len(vels) > 0 else 0
        
        rel_v = last_speed - focal_speed
        
        motion = ""
        is_stationary = False
        if max_speed < 0.5:
            motion = "stationary"
            is_stationary = True
        elif last_speed < 0.5 and first_speed > 1.0:
            motion = "stopping"
        elif last_speed > 1.0 and first_speed < 0.5:
            motion = "accelerating from stop"
        else:
            if rel_v > 1.0:
                motion = "faster than ego"
            elif rel_v < -1.0:
                motion = "slower than ego"
            else:
                motion = "similar speed to ego"

        # Merging behavior (Bug 2: Must actually be moving laterally)
        if not is_stationary and last_speed > 0.5:
            valid_pos = n["pos_hist"][n["flags"]]
            lat_disp = valid_pos[-1][1] - valid_pos[0][1] if len(valid_pos) > 1 else 0
            
            # Only flag cut-in / lateral-away within ~2 lane widths (7m)
            # to avoid mislabeling parallel traffic on separate roads
            CUT_IN_MAX_LAT = 7.0
            if "left" in lat_pos and lat_disp < -1.5 and abs(y) < CUT_IN_MAX_LAT:
                motion += ", cutting in towards ego"
            elif "right" in lat_pos and lat_disp > 1.5 and abs(y) < CUT_IN_MAX_LAT:
                motion += ", cutting in towards ego"
            elif abs(y) >= 1.8 and abs(y) < CUT_IN_MAX_LAT and abs(lat_disp) > 1.5 and (y * lat_disp > 0):
                motion += ", moving laterally away"
                
        # --- 3. Interaction / Influence Score (Defect 4 & Bug 1: CPA) ---
        min_cpa = 999.0
        ttc = 999.0
        if not is_stationary:
            # Simulate 5 seconds ahead for CPA (Closest Point of Approach)
            for t_sim in np.arange(0, 5.0, 0.5):
                ego_x = focal_speed * t_sim
                ag_x = x + vx * t_sim
                ag_y = y + vy * t_sim
                d = np.hypot(ego_x - ag_x, ag_y)
                if d < min_cpa: min_cpa = d
                if d < 1.5: ttc = min(ttc, t_sim)
        else:
            if x > 0:
                min_cpa = abs(y)
                if min_cpa < 1.5: ttc = x / focal_speed if focal_speed > 0.5 else 999
            else:
                min_cpa = dist # already passed
                
        conflict_risk = (ttc < 4.0) and (min_cpa < 4.0)
        
        influence_score = 0
        interaction = "low influence"
        
        if conflict_risk:
            interaction = f"high conflict risk (TTC ~{ttc:.1f}s)"
            influence_score = 100 - (ttc * 5)
        elif position == "lead" and dist < 25.0:
            interaction = "high influence (direct path blockage)"
            influence_score = 80 - (dist / 2.0)
        elif "cutting in" in motion and dist < 25.0:
            interaction = "high influence (cut-in hazard)"
            influence_score = 75 - (dist / 2.0)
        elif position in ["left-front", "right-front"] and dist < 20.0 and min_cpa < 8.0 and not is_stationary:
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
        # In AV2, trajs_cat contains strings like 'focal', 'av', 'score', 'unscore'
        if isinstance(cat_id, str):
            if cat_id == 'av': return "Autonomous Vehicle"
            if cat_id == 'focal': return "Focal Vehicle"
            return "Vehicle" # 'score' or 'unscore' are typically vehicles/agents
        
        # Fallback if it ever becomes int
        try:
            return {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}.get(int(cat_id), "Agent")
        except:
            return "Agent"

    def generate_prompt(self, df_row):
        seq_id = df_row["SEQ_ID"]
        city_name = df_row["CITY_NAME"]
        trajs = df_row["TRAJS"]
        lane_graph = df_row["LANE_GRAPH"]
        
        trajs_pos = trajs["trajs_pos"]
        trajs_vel = trajs["trajs_vel"]
        has_flags = trajs["has_flags"]
        trajs_cat = trajs["trajs_cat"]
        
        # 1. Base Kinematics
        fk = self._extract_focal_kinematics(trajs_pos, trajs_vel, has_flags, lane_graph)
        
        # 2. Extract Neighbors in Focal Frame
        neighbors_info = []
        trajs_ctrs = trajs["trajs_ctrs"]
        trajs_vecs = trajs["trajs_vecs"]
        
        for i in range(1, len(trajs_cat)):
            if has_flags[i, self.obs_len - 1]:
                ctr, vec = trajs_ctrs[i], trajs_vecs[i]
                theta = np.arctan2(vec[1], vec[0])
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                
                pos_focal = trajs_pos[i, :self.obs_len, :].dot(rot_mat.T) + ctr
                vel_focal = trajs_vel[i, :self.obs_len, :].dot(rot_mat.T)
                
                # Compute semantics & influence score
                curr_pos = pos_focal[-1]
                curr_vel = vel_focal[-1]
                curr_speed = np.linalg.norm(curr_vel)
                
                raw_n_dict = {"curr_pos": curr_pos, "curr_vel": curr_vel, "curr_speed": curr_speed, 
                              "vel_hist": vel_focal, "pos_hist": pos_focal, "flags": has_flags[i, :self.obs_len]}
                semantic_role, influence_score, is_stationary = self._extract_spatial_relations(raw_n_dict, fk["last_speed"])
                
                # Discard essentially irrelevant neighbors
                if influence_score <= 0 and "low influence" in semantic_role:
                    continue
                    
                neighbors_info.append({
                    "id": i, "cat": trajs_cat[i],
                    "pos_hist": pos_focal, "vel_hist": vel_focal,
                    "flags": has_flags[i, :self.obs_len],
                    "semantic_role": semantic_role,
                    "influence_score": influence_score,
                    "is_stationary": is_stationary,
                    "curr_pos": curr_pos,
                    "curr_speed": curr_speed
                })
                
        # 3. Build Semantic Labels
        labels, intention, affordances = self._identify_scenarios_and_intention(fk, lane_graph, neighbors_info)
        
        # --- PROMPT CONSTRUCTION ---
        prompt = f"Scenario '{seq_id}' in {city_name}. Task: High-Fidelity Motion Prediction.\n"
        prompt += f"[SCENARIO IDENTIFICATION]\nLabels: {', '.join(labels)}\n\n"
        prompt += f"[AFFORDANCE & INTENTION]\nFocal Agent Intention: {intention}\nMap Affordance: {', '.join(affordances)}\n\n"
        prompt += "[SPATIAL RELATIONS & HISTORY]\n"

        agent_char_ranges = []   # (char_start, char_end) for each agent written to prompt
        written_agent_ids = []   # corresponding index into trajs_pos

        # Focal Agent
        focal_valid_vels = np.linalg.norm(trajs_vel[0, :self.obs_len][has_flags[0, :self.obs_len]], axis=1)
        focal_is_stationary = focal_valid_vels.max() < 0.5 if len(focal_valid_vels) > 0 else True

        seg_start = len(prompt)
        if focal_is_stationary:
            pos = trajs_pos[0, 49] if has_flags[0, 49] else trajs_pos[0, 0]
            prompt += f"* Focal Agent [Host]: t=0~49 Stationary at ({pos[0]:.1f}, {pos[1]:.1f})\n"
        else:
            f_steps = []
            for t in range(0, self.obs_len, self.traj_step):
                if has_flags[0, t]:
                    pos = trajs_pos[0, t]
                    spd = np.linalg.norm(trajs_vel[0, t])
                    f_steps.append(f"t={t}:({pos[0]:.1f},{pos[1]:.1f})|V:{spd:.1f}m/s")
            prompt += f"* Focal Agent [Host]: {' -> '.join(f_steps)}\n"
        agent_char_ranges.append((seg_start, len(prompt)))
        written_agent_ids.append(0)

        # Sort neighbors by Influence Score descending
        neighbors_info.sort(key=lambda x: x["influence_score"], reverse=True)

        n_count = 0
        for n in neighbors_info:
            if n_count >= self.max_neighbors: break

            if n["influence_score"] < 5 and "low influence" in n["semantic_role"] and n_count >= 2:
                continue

            first_idx = np.where(n["flags"])[0][0]
            start_note = f" (Entered scene at t={first_idx})" if first_idx > 0 else ""
            agent_type = self._cat_name(n["cat"])

            seg_start = len(prompt)
            if n["is_stationary"]:
                t_range = f"t={first_idx}~49" if first_idx > 0 else "t=0~49"
                prompt += f"- {agent_type} (Impact:{n['influence_score']:.0f}){start_note} [{n['semantic_role']}]: {t_range} Stationary at ({n['curr_pos'][0]:.1f}, {n['curr_pos'][1]:.1f})\n"
            else:
                n_steps = []
                for t in range(0, self.obs_len, self.traj_step):
                    if n["flags"][t]:
                        pos = n["pos_hist"][t]
                        spd = np.linalg.norm(n["vel_hist"][t])
                        n_steps.append(f"t={t}:({pos[0]:.1f},{pos[1]:.1f})|V:{spd:.1f}m/s")
                prompt += f"- {agent_type} (Impact:{n['influence_score']:.0f}){start_note} [{n['semantic_role']}]: {' -> '.join(n_steps)}\n"
            agent_char_ranges.append((seg_start, len(prompt)))
            written_agent_ids.append(n["id"])
            n_count += 1

        prompt += "\n[60-STEP PREDICTION]:"

        return prompt, agent_char_ranges, written_agent_ids

    def generate_per_agent_prompts(self, df_row):
        """
        Build one independent prompt per agent (focal + up to max_neighbors).
        Each prompt = shared scene header + that agent's own trajectory segment.
        This avoids the single-sequence token-budget problem: every agent gets
        its own full context window in the LLM.

        Returns:
            agent_prompts:     list of (prompt_str, traj_index) — one per written agent
            written_agent_ids: list of int — indices into trajs_pos / has_flags
        """
        seq_id    = df_row["SEQ_ID"]
        city_name = df_row["CITY_NAME"]
        trajs     = df_row["TRAJS"]
        lane_graph = df_row["LANE_GRAPH"]

        trajs_pos = trajs["trajs_pos"]
        trajs_vel = trajs["trajs_vel"]
        has_flags = trajs["has_flags"]
        trajs_cat = trajs["trajs_cat"]

        fk = self._extract_focal_kinematics(trajs_pos, trajs_vel, has_flags, lane_graph)

        # Collect valid neighbors (same logic as generate_prompt)
        neighbors_info = []
        trajs_ctrs = trajs["trajs_ctrs"]
        trajs_vecs = trajs["trajs_vecs"]
        for i in range(1, len(trajs_cat)):
            if has_flags[i, self.obs_len - 1]:
                ctr, vec = trajs_ctrs[i], trajs_vecs[i]
                theta   = np.arctan2(vec[1], vec[0])
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]])
                pos_focal = trajs_pos[i, :self.obs_len, :].dot(rot_mat.T) + ctr
                vel_focal = trajs_vel[i, :self.obs_len, :].dot(rot_mat.T)
                curr_pos   = pos_focal[-1]
                curr_vel   = vel_focal[-1]
                curr_speed = np.linalg.norm(curr_vel)
                raw_n_dict = {"curr_pos": curr_pos, "curr_vel": curr_vel,
                              "curr_speed": curr_speed, "vel_hist": vel_focal,
                              "pos_hist": pos_focal, "flags": has_flags[i, :self.obs_len]}
                semantic_role, influence_score, is_stationary = \
                    self._extract_spatial_relations(raw_n_dict, fk["last_speed"])
                if influence_score <= 0 and "low influence" in semantic_role:
                    continue
                neighbors_info.append({
                    "id": i, "cat": trajs_cat[i],
                    "pos_hist": pos_focal, "vel_hist": vel_focal,
                    "flags": has_flags[i, :self.obs_len],
                    "semantic_role": semantic_role,
                    "influence_score": influence_score,
                    "is_stationary": is_stationary,
                    "curr_pos": curr_pos,
                    "curr_speed": curr_speed
                })

        labels, intention, affordances = \
            self._identify_scenarios_and_intention(fk, lane_graph, neighbors_info)

        # Shared scene header (identical for all agent prompts)
        scene_header  = f"Scenario '{seq_id}' in {city_name}. Task: High-Fidelity Motion Prediction.\n"
        scene_header += f"[SCENARIO IDENTIFICATION]\nLabels: {', '.join(labels)}\n\n"
        scene_header += (f"[AFFORDANCE & INTENTION]\nFocal Agent Intention: {intention}\n"
                         f"Map Affordance: {', '.join(affordances)}\n\n")
        scene_header += "[AGENT HISTORY]\n"

        # Focal agent segment
        focal_valid_vels = np.linalg.norm(
            trajs_vel[0, :self.obs_len][has_flags[0, :self.obs_len]], axis=1)
        focal_is_stationary = focal_valid_vels.max() < 0.5 if len(focal_valid_vels) > 0 else True

        # Focal dynamics summary (semantic, language-friendly)
        speed_delta = fk["last_speed"] - fk["first_speed"]
        if speed_delta > 0.5:
            speed_trend = "accelerating"
        elif speed_delta < -0.5:
            speed_trend = "decelerating"
        else:
            speed_trend = "constant speed"
        if abs(fk["heading_change"]) > 0.3:
            heading_desc = "turning left" if fk["heading_change"] > 0 else "turning right"
        elif abs(fk["heading_change"]) > 0.1:
            heading_desc = "slight curve"
        else:
            heading_desc = "straight ahead"
        focal_dynamics = (f"[FOCAL DYNAMICS]\n"
                          f"Speed: {fk['first_speed']:.1f}→{fk['last_speed']:.1f}m/s "
                          f"({speed_trend}) | Heading: {heading_desc}\n\n")

        if focal_is_stationary:
            pos = trajs_pos[0, 49] if has_flags[0, 49] else trajs_pos[0, 0]
            focal_seg = (focal_dynamics +
                         f"* Focal Agent [Host]: t=0~49 Stationary at "
                         f"({pos[0]:.1f}, {pos[1]:.1f})\n[PREDICT]:")
        else:
            f_steps = []
            for t in range(0, self.obs_len, self.traj_step):
                if has_flags[0, t]:
                    pos = trajs_pos[0, t]
                    spd = np.linalg.norm(trajs_vel[0, t])
                    f_steps.append(f"t={t}:({pos[0]:.1f},{pos[1]:.1f})|V:{spd:.1f}m/s")
            focal_seg = (focal_dynamics +
                         f"* Focal Agent [Host]: {' -> '.join(f_steps)}\n[PREDICT]:")

        agent_prompts     = [(scene_header + focal_seg, 0)]
        written_agent_ids = [0]

        # Neighbor segments (sorted by influence score, same filter as generate_prompt)
        neighbors_info.sort(key=lambda x: x["influence_score"], reverse=True)
        n_count = 0
        for n in neighbors_info:
            if n_count >= self.max_neighbors:
                break
            if n["influence_score"] < 5 and "low influence" in n["semantic_role"] and n_count >= 2:
                continue

            first_idx  = np.where(n["flags"])[0][0]
            start_note = f" (Entered scene at t={first_idx})" if first_idx > 0 else ""
            agent_type = self._cat_name(n["cat"])

            # Relative context block (semantic, language-friendly)
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
                rel_v_desc = f"similar speed to focal ({n['curr_speed']:.1f}m/s)"
            rel_block = (f"[RELATIVE CONTEXT vs FOCAL AGENT]\n"
                         f"Distance: {dist_m:.1f}m | Position: {position_str} | "
                         f"Relative speed: {rel_v_desc} | Risk: {interact_str}\n\n")

            if n["is_stationary"]:
                t_range = f"t={first_idx}~49" if first_idx > 0 else "t=0~49"
                nbr_seg = (rel_block +
                           f"- {agent_type}{start_note}: "
                           f"{t_range} Stationary at "
                           f"({n['curr_pos'][0]:.1f}, {n['curr_pos'][1]:.1f})\n[PREDICT]:")
            else:
                n_steps = []
                for t in range(0, self.obs_len, self.traj_step):
                    if n["flags"][t]:
                        pos = n["pos_hist"][t]
                        spd = np.linalg.norm(n["vel_hist"][t])
                        n_steps.append(f"t={t}:({pos[0]:.1f},{pos[1]:.1f})|V:{spd:.1f}m/s")
                nbr_seg = (rel_block +
                           f"- {agent_type}{start_note}: {' -> '.join(n_steps)}\n[PREDICT]:")

            agent_prompts.append((scene_header + nbr_seg, n["id"]))
            written_agent_ids.append(n["id"])
            n_count += 1

        return agent_prompts, written_agent_ids

    def generate_unified_prompt(self, df_row):
        """
        Build one unified prompt for the entire scene containing all agents.
        Each agent's section ends with [PREDICT]: — we extract the token index
        of that marker to use as the agent's representation in the LLM hidden states.

        Returns:
            input_ids:            LongTensor [L]
            attention_mask:       LongTensor [L]
            agent_token_positions: list[int]  — token index of [PREDICT]: per agent
            written_agent_ids:    list[int]   — indices into trajs_pos
        """
        seq_id     = df_row["SEQ_ID"]
        city_name  = df_row["CITY_NAME"]
        trajs      = df_row["TRAJS"]
        lane_graph = df_row["LANE_GRAPH"]

        trajs_pos  = trajs["trajs_pos"]
        trajs_vel  = trajs["trajs_vel"]
        has_flags  = trajs["has_flags"]
        trajs_cat  = trajs["trajs_cat"]

        fk = self._extract_focal_kinematics(trajs_pos, trajs_vel, has_flags, lane_graph)

        # Collect valid neighbors
        neighbors_info = []
        trajs_ctrs = trajs["trajs_ctrs"]
        trajs_vecs = trajs["trajs_vecs"]
        for i in range(1, len(trajs_cat)):
            if has_flags[i, self.obs_len - 1]:
                ctr, vec  = trajs_ctrs[i], trajs_vecs[i]
                theta     = np.arctan2(vec[1], vec[0])
                rot_mat   = np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta),  np.cos(theta)]])
                pos_focal  = trajs_pos[i, :self.obs_len, :].dot(rot_mat.T) + ctr
                vel_focal  = trajs_vel[i, :self.obs_len, :].dot(rot_mat.T)
                curr_pos   = pos_focal[-1]
                curr_vel   = vel_focal[-1]
                curr_speed = np.linalg.norm(curr_vel)
                raw_n      = {"curr_pos": curr_pos, "curr_vel": curr_vel,
                              "curr_speed": curr_speed, "vel_hist": vel_focal,
                              "pos_hist": pos_focal, "flags": has_flags[i, :self.obs_len]}
                semantic_role, influence_score, is_stationary = \
                    self._extract_spatial_relations(raw_n, fk["last_speed"])
                if influence_score <= 0 and "low influence" in semantic_role:
                    continue
                neighbors_info.append({
                    "id": i, "cat": trajs_cat[i],
                    "pos_hist": pos_focal, "vel_hist": vel_focal,
                    "flags": has_flags[i, :self.obs_len],
                    "semantic_role": semantic_role,
                    "influence_score": influence_score,
                    "is_stationary": is_stationary,
                    "curr_pos": curr_pos,
                    "curr_speed": curr_speed,
                })
        neighbors_info.sort(key=lambda x: x["influence_score"], reverse=True)

        labels, intention, affordances = \
            self._identify_scenarios_and_intention(fk, lane_graph, neighbors_info)

        # ── Scene header ──────────────────────────────────────────────────────
        prompt  = f"Scenario '{seq_id}' in {city_name}. Task: High-Fidelity Motion Prediction.\n"
        prompt += f"[SCENARIO IDENTIFICATION]\nLabels: {', '.join(labels)}\n\n"
        prompt += (f"[AFFORDANCE & INTENTION]\nFocal Agent Intention: {intention}\n"
                   f"Map Affordance: {', '.join(affordances)}\n\n")

        # ── Focal dynamics ────────────────────────────────────────────────────
        speed_delta = fk["last_speed"] - fk["first_speed"]
        speed_trend = ("accelerating" if speed_delta > 0.5 else
                       "decelerating" if speed_delta < -0.5 else "constant speed")
        heading_desc = ("turning left"  if fk["heading_change"] > 0.3 else
                        "turning right" if fk["heading_change"] < -0.3 else
                        "slight curve"  if abs(fk["heading_change"]) > 0.1 else
                        "straight ahead")
        prompt += (f"[FOCAL DYNAMICS]\n"
                   f"Speed: {fk['first_speed']:.1f}→{fk['last_speed']:.1f}m/s "
                   f"({speed_trend}) | Heading: {heading_desc}\n\n")

        # ── Collect neighbors to write (apply filters first) ─────────────────
        PREDICT_MARKER = "[PREDICT]:"
        predict_char_ends = []
        written_agent_ids = []

        selected_neighbors = []
        n_count = 0
        for n in neighbors_info:
            if n_count >= self.max_neighbors:
                break
            if n["influence_score"] < 5 and "low influence" in n["semantic_role"] and n_count >= 2:
                continue
            selected_neighbors.append(n)
            n_count += 1

        # ── BLOCK 1: All relative context summaries (before any trajectory) ──
        # Placing all context here ensures the focal [PREDICT]: token can attend
        # to every neighbor's relative information (causal LM ordering).
        if selected_neighbors:
            prompt += "[SURROUNDING AGENTS CONTEXT]\n"
            for k, n in enumerate(selected_neighbors, 1):
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
                prompt += (f"  Agent{k} ({agent_type}): "
                           f"Distance {dist_m:.1f}m | {position_str} | "
                           f"Speed {rel_v_desc} | {interact_str}\n")
            prompt += "\n"

        # ── BLOCK 2: All agent trajectories + [PREDICT]: markers ─────────────
        # Each [PREDICT]: immediately follows its own trajectory.
        # Because all relative context is above, focal [PREDICT]: sees everything.
        prompt += "[AGENT TRAJECTORIES]\n"

        # Focal agent trajectory
        focal_valid_vels    = np.linalg.norm(
            trajs_vel[0, :self.obs_len][has_flags[0, :self.obs_len]], axis=1)
        focal_is_stationary = focal_valid_vels.max() < 0.5 if len(focal_valid_vels) > 0 else True

        if focal_is_stationary:
            pos = trajs_pos[0, 49] if has_flags[0, 49] else trajs_pos[0, 0]
            prompt += f"* Focal Agent [Host]: t=0~49 Stationary at ({pos[0]:.1f}, {pos[1]:.1f})\n"
        else:
            f_steps = []
            for t in range(0, self.obs_len, self.traj_step):
                if has_flags[0, t]:
                    pos = trajs_pos[0, t]
                    spd = np.linalg.norm(trajs_vel[0, t])
                    f_steps.append(f"t={t}:({pos[0]:.1f},{pos[1]:.1f})|V:{spd:.1f}m/s")
            prompt += f"* Focal Agent [Host]: {' -> '.join(f_steps)}\n"
        prompt += PREDICT_MARKER + "\n"
        predict_char_ends.append(len(prompt) - len("\n") - 1)
        written_agent_ids.append(0)

        # Neighbor trajectories — use same AgentK label as in context summary
        for k, n in enumerate(selected_neighbors, 1):
            first_idx  = np.where(n["flags"])[0][0]
            start_note = f" (Entered at t={first_idx})" if first_idx > 0 else ""
            agent_type = self._cat_name(n["cat"])
            label      = f"Agent{k} ({agent_type})"

            if n["is_stationary"]:
                t_range = f"t={first_idx}~49" if first_idx > 0 else "t=0~49"
                prompt += (f"- {label}{start_note}: "
                           f"{t_range} Stationary at ({n['curr_pos'][0]:.1f}, {n['curr_pos'][1]:.1f})\n")
            else:
                n_steps = []
                for t in range(0, self.obs_len, self.traj_step):
                    if n["flags"][t]:
                        pos = n["pos_hist"][t]
                        spd = np.linalg.norm(n["vel_hist"][t])
                        n_steps.append(f"t={t}:({pos[0]:.1f},{pos[1]:.1f})|V:{spd:.1f}m/s")
                prompt += f"- {label}{start_note}: {' -> '.join(n_steps)}\n"
            prompt += PREDICT_MARKER + "\n"
            predict_char_ends.append(len(prompt) - len("\n") - 1)
            written_agent_ids.append(n["id"])

        # ── Tokenize with offset mapping ──────────────────────────────────────
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len_per_agent,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids      = enc["input_ids"].squeeze(0)        # [L]
        attention_mask = enc["attention_mask"].squeeze(0)   # [L]
        offsets        = enc["offset_mapping"].squeeze(0)   # [L, 2]
        L_tok          = input_ids.shape[0]

        # Map character positions to token indices
        # For each [PREDICT]: end position, find the last token whose start <= char_pos
        offsets_list = offsets.tolist()
        agent_token_positions = []
        for char_end in predict_char_ends:
            token_idx = 0
            for i in range(L_tok - 1, -1, -1):
                s, e = offsets_list[i]
                if s == 0 and e == 0:
                    continue    # padding token — skip
                if s <= char_end:
                    token_idx = i
                    break
            agent_token_positions.append(token_idx)

        return input_ids, attention_mask, agent_token_positions, written_agent_ids

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        df = pd.read_pickle(file_path)
        row = df.iloc[0]

        # Unified prompt: one sequence per scene, all agents included
        input_ids, attention_mask, agent_token_positions, written_agent_ids = \
            self.generate_unified_prompt(row)

        max_total  = self.max_neighbors + 1   # focal + up to max_neighbors
        L          = input_ids.shape[0]

        # Pad agent_token_positions to max_total slots (extras map to position 0)
        while len(agent_token_positions) < max_total:
            agent_token_positions.append(0)
        agent_positions = torch.tensor(agent_token_positions[:max_total], dtype=torch.long)  # [N]

        # GT trajectories
        trajs_pos = row["TRAJS"]["trajs_pos"]
        has_flags = row["TRAJS"]["has_flags"]
        trajs_cat = row["TRAJS"]["trajs_cat"]

        gt_trajectories     = torch.zeros(max_total, 60, 2, dtype=torch.float32)
        gt_abs_trajectories = torch.zeros(max_total, 60, 2, dtype=torch.float32)
        gt_anchor           = torch.zeros(max_total, 2,  dtype=torch.float32)
        gt_masks            = torch.zeros(max_total, 60, dtype=torch.bool)
        agent_valid         = torch.zeros(max_total,     dtype=torch.bool)
        train_mask          = torch.zeros(max_total,     dtype=torch.bool)

        for a_idx, ag_id in enumerate(written_agent_ids):
            abs_pos = torch.tensor(trajs_pos[ag_id, self.obs_len:, :], dtype=torch.float32)
            flags   = torch.tensor(has_flags[ag_id, self.obs_len:],    dtype=torch.bool)

            anchor_idx = self.obs_len - 1
            while anchor_idx > 0 and not has_flags[ag_id, anchor_idx]:
                anchor_idx -= 1
            anchor = torch.tensor(trajs_pos[ag_id, anchor_idx], dtype=torch.float32)

            disp = torch.zeros(60, 2, dtype=torch.float32)
            prev = anchor.clone()
            for t in range(60):
                if flags[t]:
                    disp[t] = abs_pos[t] - prev
                    prev    = abs_pos[t]

            gt_trajectories[a_idx]     = disp
            gt_abs_trajectories[a_idx] = abs_pos
            gt_anchor[a_idx]           = anchor
            gt_masks[a_idx]            = flags
            agent_valid[a_idx]         = True
            cat = str(trajs_cat[ag_id])
            train_mask[a_idx]          = (cat in ('focal', 'score'))

        return {
            "input_ids":           input_ids,              # [L]   unified scene sequence
            "attention_mask":      attention_mask,          # [L]
            "agent_positions":     agent_positions,         # [N]   token index per agent
            "gt_trajectories":     gt_trajectories,         # [N, 60, 2] displacement GT
            "gt_abs_trajectories": gt_abs_trajectories,     # [N, 60, 2] absolute GT
            "gt_anchor":           gt_anchor,               # [N, 2]
            "gt_masks":            gt_masks,                # [N, 60]
            "agent_valid":         agent_valid,             # [N]
            "train_mask":          train_mask,              # [N] loss mask (focal+score only)
        }

"""
Hybrid LLM Motion Predictor
============================
Architecture:
  1. Text tokens  (semantic scene description) → LLM embedding table
  2. Agent tokens (numerical trajectory + type) → AgentEncoder → projection
  3. Lane tokens  (numerical polyline geometry) → LaneEncoder  → projection
  4. Concatenate all as inputs_embeds → Qwen3-0.6B (LoRA, single forward)
  5. Extract [AGT] summary token hidden states per agent
  6. Level-k Interaction Decoder (GameFormer-style cross-attention refinement)
  7. MLP Decoder → multi-modal trajectories [B, N, K, T, 2]

Sequence layout:
  [text (L_t)] [agent_embs × N] [lane_embs × L] [AGT_tokens × N]
  ↑ causal attention ─────────────────────────────────────────────────→
  Each AGT_i sees: full text + all agent embeddings + all lane embeddings
                   + AGT_0 .. AGT_{i-1}

Gradient design note:
  ALL trajectory modules are feedforward (MLP + MaxPool / Attention), NO RNNs.
  Recurrent decoders (GRU/LSTM) backprop through T timesteps, amplifying the
  gradient ~T× (BPTT) — the primary cause of gradient explosion. GameFormer's
  GMMPredictor is a pure MLP for exactly this reason; we follow the same design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType


# ── Agent Encoder ─────────────────────────────────────────────────────────────

class AgentEncoder(nn.Module):
    """
    MLP + MaxPool trajectory history encoder (consistent with FutureEncoder).

    Replaces the previous GRU implementation to eliminate 50-step BPTT,
    which was the other major source of gradient explosion alongside GRUDecoder.

    Architecture: per-frame MLP → MaxPool over T → projection
    This matches the GameFormer philosophy: use feedforward + pooling,
    not RNNs, for trajectory feature extraction.

    Input:  [B*N, T_obs, 13]  (x,y,vx,vy,cos_θ,sin_θ, type_one_hot×7)
    Output: [B*N, out_dim]
    """
    def __init__(self, input_dim=13, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        # x: [B*N, T, 13]
        h = self.mlp(x)                 # [B*N, T, hidden_dim]  — no BPTT
        h = h.max(dim=1).values         # [B*N, hidden_dim]     — MaxPool over T
        return self.out_proj(h)         # [B*N, out_dim]


# ── Lane Encoder ──────────────────────────────────────────────────────────────

class LaneEncoder(nn.Module):
    """
    PointNet-lite: shared MLP per point + max-pool over 10 points per polyline.
    Input:  [B*L, 10, 8]  (x,y,dx,dy,is_intersect,type_0,type_1,type_2)
    Output: [B*L, out_dim]
    """
    def __init__(self, input_dim=8, hidden_dim=64, out_dim=1024):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        # x: [B*L, 10, 8]
        h = self.point_mlp(x)           # [B*L, 10, hidden_dim]
        h = h.max(dim=1).values         # [B*L, hidden_dim]  ← PointNet max-pool
        return self.out_proj(h)         # [B*L, out_dim]


# ── Future Encoder ────────────────────────────────────────────────────────────

class FutureEncoder(nn.Module):
    """
    GameFormer-style MLP + MaxPool trajectory encoder.

    Replaces GRU to eliminate 60-step BPTT gradient amplification.
    Features: (x, y, dx, dy) per step → MLP → MaxPool over T.
    Input is detached both by the caller AND internally (GameFormer pattern).

    Input:  [M, T, 2]   absolute trajectory (detached by caller)
    Output: [M, out_dim]
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, traj):
        # traj: [M, T, 2] — caller must detach; we also detach inside (GameFormer)
        vel  = torch.diff(traj, dim=1)                      # [M, T-1, 2]
        feat = torch.cat([traj[:, 1:], vel], dim=-1)        # [M, T-1, 4]
        out  = self.mlp(feat.detach())                      # detach inside too
        return out.max(dim=1).values                        # [M, out_dim]


# ── MLP Trajectory Decoder ────────────────────────────────────────────────────

class MLPDecoder(nn.Module):
    """
    Feedforward multi-modal trajectory decoder — aligned with GameFormer's
    GMMPredictor (Linear→ELU→Linear, predicting the WHOLE trajectory at once).

    Replaces the previous GRUDecoder. A GRU decoder backprops through T=60
    timesteps (BPTT), causing the gradient on its recurrent weight W_hh to
    accumulate ~60×, which was the dominant source of gradient explosion.
    This MLP predicts all T×2 displacements in a single forward pass, so the
    gradient w.r.t. every parameter is O(1) — no temporal amplification.

    Per-mode prediction: agent_repr + mode_embedding → MLP → [T, 2].

    Input:  [BN, H]
    Output: [BN, K, T, 2]  per-step displacements
    """
    def __init__(self, hidden_dim, num_modes=6, num_future_steps=60,
                 mlp_hidden=512, dropout=0.1):
        super().__init__()
        self.num_modes        = num_modes
        self.num_future_steps = num_future_steps

        # Learned per-mode query, added to the agent representation
        self.mode_embeds = nn.Embedding(num_modes, hidden_dim)
        nn.init.normal_(self.mode_embeds.weight, std=0.02)

        # GameFormer-style MLP head: predict entire trajectory in one shot
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.ELU(),
            nn.Linear(mlp_hidden, num_future_steps * 2),
        )

    def forward(self, agent_repr):
        # agent_repr: [BN, H]
        BN, H = agent_repr.shape
        K, T  = self.num_modes, self.num_future_steps
        mode_e = self.mode_embeds(
            torch.arange(K, device=agent_repr.device))           # [K, H]
        x    = agent_repr.unsqueeze(1) + mode_e.unsqueeze(0)     # [BN, K, H]
        disp = self.mlp(x)                                       # [BN, K, T*2]
        return disp.view(BN, K, T, 2)


# ── Level-k Interaction Decoder ───────────────────────────────────────────────

class InteractionDecoder(nn.Module):
    """
    Level-k Interaction Decoder — aligned with GameFormer's original design.

    Key GameFormer principles adopted:
      1. SelfTransformer first: self-attention across ALL agents captures global
         interaction context before individual refinement (GameFormer core).
      2. Per-level independent modules (GameFormer uses separate InteractionDecoder
         per level): eliminates cross-level gradient accumulation on shared weights.
      3. cross_attn_dim=128 (GameFormer=256; 128 safer given our H=1024 inputs).
      4. FutureEncoder: MLP+MaxPool, replaces GRU (eliminates 60-step BPTT).
      5. Detach query in cross-attention: gradient reaches h_agents ONLY through
         the residual, not through the attention computation.
      6. Detach h_cur between levels: stops multi-level gradient cascade.
      7. Zero-init out_proj: interaction starts as identity (activated gradually).
      8. FFN after cross-attention: GameFormer's full transformer block structure.
    """
    def __init__(self, traj_decoder, hidden_dim=1024, cross_attn_dim=128,
                 nhead=4, n_levels=2, dropout=0.1):
        super().__init__()
        self.traj_decoder = traj_decoder
        self.n_levels     = n_levels

        # Shared FutureEncoder across levels (GameFormer shares future_encoder)
        self.future_enc  = FutureEncoder(out_dim=cross_attn_dim)

        # ── Per-level independent modules (GameFormer: separate decoder per level) ──
        # SelfTransformer: global interaction context across ALL agents
        self.self_attns  = nn.ModuleList([
            nn.MultiheadAttention(cross_attn_dim, nhead, dropout=dropout, batch_first=True)
            for _ in range(n_levels)])
        self.sa_norms    = nn.ModuleList([nn.LayerNorm(cross_attn_dim) for _ in range(n_levels)])

        # CrossTransformer: per-agent refinement with interaction context
        self.q_projs     = nn.ModuleList([nn.Linear(hidden_dim, cross_attn_dim) for _ in range(n_levels)])
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(cross_attn_dim, nhead, dropout=dropout, batch_first=True)
            for _ in range(n_levels)])
        self.ca_norms    = nn.ModuleList([nn.LayerNorm(cross_attn_dim) for _ in range(n_levels)])

        # FFN (GameFormer: full transformer block with dim→dim*4→dim GELU)
        self.ffns        = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cross_attn_dim, cross_attn_dim * 4), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(cross_attn_dim * 4, cross_attn_dim), nn.Dropout(dropout),
            ) for _ in range(n_levels)])
        self.ffn_norms   = nn.ModuleList([nn.LayerNorm(cross_attn_dim) for _ in range(n_levels)])

        # Project correction back to H; zero-init → no interaction at training start
        self.out_projs   = nn.ModuleList([nn.Linear(cross_attn_dim, hidden_dim) for _ in range(n_levels)])
        self.out_norms   = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_levels)])
        for op in self.out_projs:
            nn.init.zeros_(op.weight)
            nn.init.zeros_(op.bias)

    def _refine(self, h_agents, prev_trajs, level_idx, agent_valid=None):
        """
        h_agents:    [B, N, H]
        prev_trajs:  [B, N, K, T, 2]  — detached by caller
        agent_valid: [B, N] bool       — mask padding agents from self-attention KV
        Returns:     [B, N, H]
        """
        B, N, H = h_agents.shape

        # Step 1: Encode ALL agents' mean futures (MLP+MaxPool, detached)
        K_mean  = prev_trajs.mean(dim=2)                         # [B, N, T, 2]
        T       = K_mean.shape[2]
        fut_emb = self.future_enc(
            K_mean.reshape(B * N, T, 2)
        ).reshape(B, N, -1)                                       # [B, N, D]

        # Step 2: SelfTransformer — global interaction context (GameFormer key step)
        # key_padding_mask: True = ignore (padding agents should not attend as KV)
        pad_mask = None
        if agent_valid is not None:
            pad_mask = ~agent_valid                               # [B, N], True=padding
        sa_out, _ = self.self_attns[level_idx](fut_emb, fut_emb, fut_emb,
                                               key_padding_mask=pad_mask)
        interaction = self.sa_norms[level_idx](fut_emb + sa_out) # [B, N, D]

        # Step 3: CrossTransformer — per-agent refinement
        # GameFormer: query = last_content + multi_futures (combines prev repr + future context)
        # Detach query: gradient to h_agents flows through residual only (not attn path)
        q       = self.q_projs[level_idx](h_agents.detach())     # [B, N, D]
        query   = q + interaction                                 # [B, N, D]
        ca_out, _ = self.cross_attns[level_idx](query, interaction, interaction)
        ca_out  = self.ca_norms[level_idx](query + ca_out)       # [B, N, D]

        # Step 4: FFN (GameFormer's full transformer block)
        ffn_out = self.ffns[level_idx](ca_out)
        ca_out  = self.ffn_norms[level_idx](ca_out + ffn_out)    # [B, N, D]

        # Step 5: ResNet residual — project to H, zero-initialized
        correction = self.out_projs[level_idx](ca_out)           # [B, N, H]
        return self.out_norms[level_idx](h_agents + correction)

    def forward(self, h_agents, gt_anchor, agent_valid=None):
        """
        h_agents:    [B, N, H]
        gt_anchor:   [B, N, 2]
        agent_valid: [B, N] bool  — propagated to _refine for padding mask
        Returns:     list of [B, N, K, T, 2], length = n_levels + 1
        """
        B, N, H = h_agents.shape

        # Level-0: pure encoder-decoder, no interaction (GameFormer: InitialDecoder)
        traj_disp = self.traj_decoder(h_agents.reshape(B * N, H))
        K, T      = traj_disp.shape[1], traj_disp.shape[2]
        traj_disp = traj_disp.reshape(B, N, K, T, 2)

        anchor   = gt_anchor.unsqueeze(2).unsqueeze(2)           # [B, N, 1, 1, 2]
        traj_abs = anchor + traj_disp.cumsum(dim=-2)

        all_preds_disp = [traj_disp]

        h_cur = h_agents
        for level_idx in range(self.n_levels):
            h_cur = self._refine(h_cur, traj_abs.detach(), level_idx, agent_valid)
            traj_disp_k = self.traj_decoder(
                h_cur.reshape(B * N, H)).reshape(B, N, K, T, 2)
            traj_abs = anchor + traj_disp_k.cumsum(dim=-2)
            all_preds_disp.append(traj_disp_k)
            h_cur = h_cur.detach()

        return all_preds_disp   # list of [B, N, K, T, 2], length = n_levels+1


# ── Main Model ────────────────────────────────────────────────────────────────

class HybridLLMPredictor(nn.Module):
    """
    Hybrid Token LLM Predictor with Level-k Interaction Decoding.

    Sequence (inputs_embeds):
      [text_emb (L_t)] [agent_emb × N] [lane_emb × L] [AGT_token × N]
      └── text via embedding table ──┘  └── numerical projections ──┘  └── learned ─┘

    AGT position of agent i = L_t + N + L + i
    """

    def __init__(self,
                 model_name="Qwen/Qwen3-0.6B-Base",
                 lora_r=16,
                 lora_alpha=32,
                 lora_target_modules=None,
                 lora_dropout=0.05,
                 n_levels=2,
                 max_agents=6,
                 max_lanes=20,
                 num_modes=6,
                 num_future_steps=60,
                 gru_hidden=256,
                 gru_layers=2,
                 use_flash_attn=True,
                 dtype=torch.bfloat16,
                 device=None):
        super().__init__()

        self.max_agents      = max_agents
        self.max_lanes       = max_lanes
        self.num_modes       = num_modes
        self.num_future_steps = num_future_steps

        # ── LLM ──────────────────────────────────────────────────────────────
        self.llm_config = AutoConfig.from_pretrained(model_name)
        H = self.llm_config.hidden_size

        attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
        self.llm = AutoModel.from_pretrained(
            model_name, attn_implementation=attn_impl,
            torch_dtype=dtype, low_cpu_mem_usage=False)

        for param in self.llm.parameters():
            param.requires_grad = False

        if lora_target_modules is None:
            lora_target_modules = "all-linear"
        # "all-linear" is a PEFT special value that auto-discovers every nn.Linear
        # in the LLM — more comprehensive than listing modules by name.
        # A plain list of names is also accepted for selective targeting.
        if isinstance(lora_target_modules, list) and lora_target_modules == ["all-linear"]:
            lora_target_modules = "all-linear"
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none", task_type=TaskType.FEATURE_EXTRACTION)
        self.llm = get_peft_model(self.llm, lora_config)
        print(f"Loaded {model_name} with LoRA (r={lora_r}, α={lora_alpha}, "
              f"targets={lora_target_modules})")
        self.llm.print_trainable_parameters()

        # ── Encoders ─────────────────────────────────────────────────────────
        self.agent_encoder = AgentEncoder(input_dim=13, out_dim=H)
        self.lane_encoder  = LaneEncoder(input_dim=8,  out_dim=H)

        # Learnable [AGT] summary tokens — one shared embedding expanded per agent
        self.agent_summary_token = nn.Parameter(torch.randn(1, 1, H) * 0.02)

        # ResNet-inspired correction projection.
        # agent_emb is fed to the LLM as .detach() (prevents gradient explosion through
        # 28 transformer layers).  The LLM output is treated as a learned "correction"
        # added to the encoder's identity path:
        #   h_agents = agent_emb  +  llm_correction_proj(h_llm)
        #              ↑ identity         ↑ residual correction
        # Zero-init ensures the LLM contributes nothing at the start, letting the
        # encoder converge first before LLM refinement gradually activates.
        self.llm_correction_proj = nn.Linear(H, H)
        nn.init.zeros_(self.llm_correction_proj.weight)
        nn.init.zeros_(self.llm_correction_proj.bias)

        # ── Decoder (feedforward MLP — no BPTT, see MLPDecoder docstring) ──────
        self.traj_decoder = MLPDecoder(
            hidden_dim=H, num_modes=num_modes,
            num_future_steps=num_future_steps, mlp_hidden=gru_hidden * 2)
        self.traj_decoder = self.traj_decoder.to(dtype)

        # Normalize combined hidden states before decoding
        self.pre_decoder_norm = nn.LayerNorm(H)

        self.interaction = InteractionDecoder(
            traj_decoder=self.traj_decoder, hidden_dim=H,
            cross_attn_dim=128, nhead=4, n_levels=n_levels)

        if device is not None:
            self.to(device)

    # ── Forward ───────────────────────────────────────────────────────────────

    def _build_inputs_embeds(self, text_input_ids, text_attention_mask,
                              agent_emb, lane_emb, N, L):
        """
        Build inputs_embeds from pre-computed (and already DETACHED) encoder embeddings.

        agent_emb / lane_emb must be detached before calling to prevent gradient
        from flowing back through the 28 LLM layers to the encoders.

        Returns:
          inputs_embeds: [B, L_total, H]
          attn_mask:     [B, L_total]
          agt_positions: list[int]
        """
        B = text_input_ids.shape[0]

        # Text embeddings via LLM embedding table
        text_emb = self.llm.get_input_embeddings()(text_input_ids)   # [B, L_t, H]
        L_t = text_emb.shape[1]

        # Learnable [AGT] summary tokens
        agt_tokens = self.agent_summary_token.expand(B, N, -1)        # [B, N, H]

        # Sequence: [text | agent_embs | lane_embs | AGT_tokens]
        inputs_embeds = torch.cat([text_emb, agent_emb, lane_emb, agt_tokens], dim=1)

        L_total   = inputs_embeds.shape[1]
        attn_mask = torch.ones(B, L_total, device=inputs_embeds.device, dtype=torch.long)
        attn_mask[:, :L_t] = text_attention_mask

        agt_positions = [L_t + N + L + i for i in range(N)]
        return inputs_embeds, attn_mask, agt_positions

    def forward(self, text_input_ids, text_attention_mask,
                agent_features, agent_valid,
                lane_features, lane_valid,
                gt_anchor):
        """
        Args:
            text_input_ids:      [B, L_t]
            text_attention_mask: [B, L_t]
            agent_features:      [B, N, T_obs, 13]
            agent_valid:         [B, N]  bool
            lane_features:       [B, L, 10, 8]
            lane_valid:          [B, L]  bool
            gt_anchor:           [B, N, 2]
        Returns:
            all_preds: list of [B, N, K, T, 2], length = n_levels + 1
        """
        B = text_input_ids.shape[0]
        N = agent_features.shape[1]
        L = lane_features.shape[1]
        H = self.llm_config.hidden_size

        # ── Encoder forward (with gradient) ──────────────────────────────────
        agent_emb = self.agent_encoder(
            agent_features.reshape(B * N, agent_features.shape[2],
                                   agent_features.shape[3])
        ).reshape(B, N, H)                                             # [B, N, H]

        lane_emb = self.lane_encoder(
            lane_features.reshape(B * L, lane_features.shape[2],
                                  lane_features.shape[3])
        ).reshape(B, L, H)                                             # [B, L, H]

        # ── LLM forward with DETACHED encoder outputs ─────────────────────────
        # Detaching prevents gradient from flowing back through 28 transformer
        # layers to the encoders (the primary cause of gradient explosion).
        # The bypass projection below provides encoders with a direct gradient path.
        inputs_embeds, attn_mask, agt_pos = self._build_inputs_embeds(
            text_input_ids, text_attention_mask,
            agent_emb.detach(), lane_emb.detach(), N, L)

        outputs = self.llm(inputs_embeds=inputs_embeds,
                           attention_mask=attn_mask,
                           output_hidden_states=False)
        hidden = outputs.last_hidden_state                             # [B, L_total, H]

        # ── ResNet-style residual: identity(encoder) + correction(LLM) ─────────
        h_llm = torch.stack([hidden[:, pos] for pos in agt_pos], dim=1)  # [B, N, H]

        # h_agents = agent_emb (identity)  +  llm_correction_proj(h_llm) (residual correction)
        # Gradient path to AgentEncoder: loss → decoder → agent_emb  (SHORT, no LLM)
        # Gradient path to LLM LoRA:    loss → decoder → h_correction → h_llm → LoRA (long)
        # llm_correction_proj is zero-initialized → LLM contributes 0 at training start,
        # letting the encoder converge first before LLM refinement activates.
        h_correction = self.llm_correction_proj(h_llm)                  # [B, N, H]
        # Norm first, then mask: LayerNorm of a zero vector yields bias (non-zero
        # after training), so masking must happen AFTER norm to keep padding agents at 0.
        h_agents = self.pre_decoder_norm(agent_emb + h_correction)
        h_agents = h_agents * agent_valid.unsqueeze(-1).float()

        # ── Level-k interaction decoding ──────────────────────────────────────
        all_preds = self.interaction(h_agents, gt_anchor, agent_valid)
        return all_preds   # list of [B, N, K, T, 2]


# Alias for backward compat
HybridMotionPredictor = HybridLLMPredictor

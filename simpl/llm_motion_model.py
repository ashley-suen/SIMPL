import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType


class GRUDecoder(nn.Module):
    """
    Autoregressive-style GRU decoder for multi-modal trajectory prediction.

    For each of K modes, the agent representation (+ a learned mode embedding)
    is projected to a GRU initial hidden state. A shared GRU then processes T
    learned step-query vectors sequentially, producing one displacement vector
    per step.

    Architecture:
        agent_repr [BN, H]
            + mode_embed [K, H]  →  combined [BN*K, H]
            → h_proj             →  h0 [num_layers, BN*K, gru_hidden]
            + step_queries [T, H]  (learned, shared across modes/agents)
            → GRU(queries, h0)   →  gru_out [BN*K, T, gru_hidden]
            → out_proj           →  disp [BN*K, T, 2]
            → reshape            →  [BN, K, T, 2]
    """
    def __init__(self, hidden_dim, gru_hidden=256, num_layers=2,
                 num_modes=6, num_future_steps=60, dropout=0.1):
        super().__init__()
        self.num_modes        = num_modes
        self.num_future_steps = num_future_steps
        self.gru_hidden       = gru_hidden
        self.num_layers       = num_layers

        self.mode_embeds  = nn.Embedding(num_modes, hidden_dim)
        self.h_proj       = nn.Linear(hidden_dim, num_layers * gru_hidden)
        self.gru          = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj     = nn.Linear(gru_hidden, 2)

        self.step_queries = nn.Parameter(torch.zeros(num_future_steps, hidden_dim))
        nn.init.normal_(self.step_queries, std=0.02)

    def forward(self, agent_repr):
        """
        agent_repr: [BN, H]
        Returns:    [BN, K, T, 2]
        """
        BN, H = agent_repr.shape
        K     = self.num_modes
        T     = self.num_future_steps

        mode_ids  = torch.arange(K, device=agent_repr.device)
        mode_e    = self.mode_embeds(mode_ids)                             # [K, H]

        agent_exp = agent_repr.unsqueeze(1).expand(BN, K, H).reshape(BN * K, H)
        mode_exp  = mode_e.unsqueeze(0).expand(BN, K, H).reshape(BN * K, H)
        combined  = agent_exp + mode_exp                                   # [BN*K, H]

        h0 = self.h_proj(combined)                                         # [BN*K, num_layers*gru_hidden]
        h0 = h0.view(BN * K, self.num_layers, self.gru_hidden)
        h0 = h0.permute(1, 0, 2).contiguous()                             # [num_layers, BN*K, gru_hidden]

        queries = self.step_queries.unsqueeze(0).expand(BN * K, T, H)

        gru_out, _ = self.gru(queries, h0)                                 # [BN*K, T, gru_hidden]
        disp       = self.out_proj(gru_out)                                # [BN*K, T, 2]

        return disp.view(BN, K, T, 2)


class UnifiedLLMMotionPredictor(nn.Module):
    """
    Unified-prompt architecture:
      - One LLM forward pass per scene (all agents in a single sequence)
      - Per-agent representations extracted at [PREDICT]: token positions
      - No Social Attention needed: LLM self-attention already handles cross-agent context
      - GRU Decoder produces multi-modal trajectories for all agents

    Pipeline:
        input_ids [B, L]  (unified scene prompt)
            ↓  LLM
        hidden [B, L, H]
            ↓  gather at agent_positions [B, N]
        agent_repr [B, N, H]
            ↓  GRU Decoder
        predicted_trajs [B, N, K, T, 2]
    """
    def __init__(self,
                 model_name="Qwen/Qwen3-0.6B-Base",
                 num_future_steps=60,
                 num_modes=6,
                 lora_r=16,
                 lora_alpha=32,
                 lora_target_modules=None,
                 lora_dropout=0.05,
                 gru_hidden=256,
                 gru_layers=2,
                 device=None,
                 use_flash_attn=False,
                 dtype=torch.float32):
        super().__init__()

        self.num_future_steps = num_future_steps
        self.num_modes        = num_modes

        self.config = AutoConfig.from_pretrained(model_name)

        attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
        self.llm = AutoModel.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        # Freeze all LLM weights; LoRA adapters will be the only trainable parts
        for param in self.llm.parameters():
            param.requires_grad = False

        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        print(f"Loaded {model_name} with LoRA (r={lora_r}, alpha={lora_alpha}, "
              f"targets={lora_target_modules}).")
        self.llm.print_trainable_parameters()

        hidden_dim = self.config.hidden_size

        self.gru_decoder = GRUDecoder(
            hidden_dim=hidden_dim,
            gru_hidden=gru_hidden,
            num_layers=gru_layers,
            num_modes=num_modes,
            num_future_steps=num_future_steps,
        )
        self.gru_decoder = self.gru_decoder.to(dtype)

        if device is not None:
            self.to(device)

    def forward(self, input_ids, attention_mask, agent_positions, agent_valid):
        """
        Args:
            input_ids:       [B, L]   unified scene prompt (all agents in one sequence)
            attention_mask:  [B, L]
            agent_positions: [B, N]   token index of each agent's [PREDICT]: marker
            agent_valid:     [B, N]   bool — True = real agent, False = padding slot
        Returns:
            predicted_trajs: [B, N, K, T, 2]  per-step displacement predictions
        """
        B, L = input_ids.shape
        N    = agent_positions.shape[1]

        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=False)
        hidden = outputs.last_hidden_state                                  # [B, L, H]
        H      = hidden.shape[-1]

        # Extract hidden state at each agent's [PREDICT]: token position
        pos     = agent_positions.clamp(0, L - 1)                          # [B, N]
        pos_exp = pos.unsqueeze(-1).expand(B, N, H)                        # [B, N, H]
        agent_repr = hidden.gather(1, pos_exp)                             # [B, N, H]

        # Zero out padding slots
        agent_repr = agent_repr * agent_valid.unsqueeze(-1).float()

        # GRU decoder: flatten to [B*N, H], decode, reshape back
        predicted_trajs = self.gru_decoder(agent_repr.view(B * N, H))      # [B*N, K, T, 2]
        predicted_trajs = predicted_trajs.view(B, N, self.num_modes,
                                               self.num_future_steps, 2)   # [B, N, K, T, 2]
        return predicted_trajs


# Keep old name as alias for backward compatibility with saved checkpoints
SmolLMMotionPredictor = UnifiedLLMMotionPredictor


# --- Quick Local Test ---
if __name__ == "__main__":
    N_agents = 6
    B, L     = 2, 512
    K        = 6

    dummy_input_ids   = torch.randint(0, 49000, (B, L))
    dummy_attn_mask   = torch.ones((B, L), dtype=torch.long)
    dummy_agent_pos   = torch.randint(0, L, (B, N_agents))
    dummy_agent_valid = torch.ones(B, N_agents, dtype=torch.bool)
    dummy_agent_valid[0, -1] = False

    print("Initializing UnifiedLLMMotionPredictor (Qwen3-0.6B, GRU decoder, K=6 modes)...")
    model = UnifiedLLMMotionPredictor(model_name="Qwen/Qwen3-0.6B-Base",
                                      num_modes=K, lora_r=16, lora_alpha=32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    print("Running forward pass...")
    out = model(dummy_input_ids, dummy_attn_mask, dummy_agent_pos, dummy_agent_valid)
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, N_agents, K, 60, 2), f"Unexpected shape: {out.shape}"
    print("Forward pass successful!")

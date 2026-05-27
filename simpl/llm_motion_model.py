import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SocialAttention(nn.Module):
    """
    Single-layer multi-head self-attention over agent representations.
    Lets each agent attend to all other (valid) agents, providing social context
    after each agent has been independently encoded by the LLM.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, agent_valid):
        """
        x:           [B, N, H]
        agent_valid: [B, N] bool — True = agent present, False = padding slot
        Returns:     [B, N, H]
        """
        key_padding_mask = ~agent_valid          # True means IGNORE that key
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + out)


class GRUDecoder(nn.Module):
    """
    Autoregressive-style GRU decoder for multi-modal trajectory prediction.

    For each of K modes, the agent representation (+ a learned mode embedding)
    is projected to a GRU initial hidden state. A shared GRU then processes T
    learned step-query vectors sequentially, producing one displacement vector
    per step. Because the GRU hidden state evolves across T steps, each output
    displacement is conditioned on all previous ones — ensuring temporal
    coherence without the O(T) gradient amplification of the cumsum approach.

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

        # Learnable per-step query vectors fed as GRU input sequence
        self.step_queries = nn.Parameter(torch.zeros(num_future_steps, hidden_dim))
        nn.init.normal_(self.step_queries, std=0.02)

    def forward(self, agent_repr):
        """
        agent_repr: [BN, H]
        Returns:    [BN, K, T, 2]  per-step displacement predictions
        """
        BN, H = agent_repr.shape
        K     = self.num_modes
        T     = self.num_future_steps

        # Mode-conditioned initial hidden state
        mode_ids  = torch.arange(K, device=agent_repr.device)
        mode_e    = self.mode_embeds(mode_ids)                             # [K, H]

        agent_exp = agent_repr.unsqueeze(1).expand(BN, K, H).reshape(BN * K, H)
        mode_exp  = mode_e.unsqueeze(0).expand(BN, K, H).reshape(BN * K, H)
        combined  = agent_exp + mode_exp                                   # [BN*K, H]

        h0 = self.h_proj(combined)                                         # [BN*K, num_layers*gru_hidden]
        h0 = h0.view(BN * K, self.num_layers, self.gru_hidden)
        h0 = h0.permute(1, 0, 2).contiguous()                             # [num_layers, BN*K, gru_hidden]

        # Step queries: [T, H] → [BN*K, T, H]
        queries = self.step_queries.unsqueeze(0).expand(BN * K, T, H)

        gru_out, _ = self.gru(queries, h0)                                 # [BN*K, T, gru_hidden]
        disp       = self.out_proj(gru_out)                                # [BN*K, T, 2]

        return disp.view(BN, K, T, 2)


class SmolLMMotionPredictor(nn.Module):
    def __init__(self,
                 model_name="HuggingFaceTB/SmolLM-135M",
                 num_future_steps=60,
                 num_modes=6,
                 unfreeze_last_n_layers=2,
                 social_num_heads=4,
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

        # Freeze strategy: embeddings frozen, last N transformer layers trainable
        for param in self.llm.embed_tokens.parameters():
            param.requires_grad = False

        num_layers    = len(self.llm.layers)
        frozen_layers = num_layers - unfreeze_last_n_layers
        for i, layer in enumerate(self.llm.layers):
            req = (i >= frozen_layers)
            for param in layer.parameters():
                param.requires_grad = req

        for param in self.llm.norm.parameters():
            param.requires_grad = True

        print(f"Loaded {model_name}.")
        print(f"Total layers: {num_layers}. Frozen: {frozen_layers}, Trainable: {unfreeze_last_n_layers}.")

        hidden_dim = self.config.hidden_size  # 576 for SmolLM-135M

        # Social cross-attention over agents
        self.social_attn = SocialAttention(hidden_dim, num_heads=social_num_heads)

        # GRU decoder replaces flat MLP head
        self.gru_decoder = GRUDecoder(
            hidden_dim=hidden_dim,
            gru_hidden=gru_hidden,
            num_layers=gru_layers,
            num_modes=num_modes,
            num_future_steps=num_future_steps,
        )

        # Match dtype of trainable modules to LLM
        self.social_attn = self.social_attn.to(dtype)
        self.gru_decoder = self.gru_decoder.to(dtype)

        if device is not None:
            self.to(device)

    def forward(self, input_ids, attention_mask, agent_valid):
        """
        Args:
            input_ids:      [B, N, L]
            attention_mask: [B, N, L]
            agent_valid:    [B, N]
        Returns:
            predicted_trajs: [B, N, K, T, 2]  per-step displacement predictions
        """
        B, N, L = input_ids.shape

        ids_flat  = input_ids.view(B * N, L)
        mask_flat = attention_mask.view(B * N, L)

        outputs = self.llm(input_ids=ids_flat, attention_mask=mask_flat,
                           output_hidden_states=False)
        hidden = outputs.last_hidden_state                                  # [B*N, L, H]
        H      = hidden.shape[-1]

        # Masked mean-pool over valid tokens
        mask_exp = mask_flat.unsqueeze(-1).float()
        pooled   = (hidden * mask_exp).sum(dim=1) / \
                   mask_exp.sum(dim=1).clamp(min=1e-9)                      # [B*N, H]

        agent_repr = pooled.view(B, N, H)
        agent_repr = agent_repr * agent_valid.unsqueeze(-1).float()
        agent_repr = self.social_attn(agent_repr, agent_valid)              # [B, N, H]

        # GRU decoder: [B*N, H] → [B*N, K, T, 2]
        predicted_trajs = self.gru_decoder(agent_repr.view(B * N, H))      # [B*N, K, T, 2]
        predicted_trajs = predicted_trajs.view(B, N, self.num_modes,
                                               self.num_future_steps, 2)   # [B, N, K, T, 2]
        return predicted_trajs


# --- Quick Local Test ---
if __name__ == "__main__":
    N_agents = 6
    B, L     = 2, 128
    K        = 6

    dummy_input_ids   = torch.randint(0, 49000, (B, N_agents, L))
    dummy_attn_mask   = torch.ones((B, N_agents, L), dtype=torch.long)
    dummy_attn_mask[0, -1, 10:] = 0
    dummy_agent_valid = torch.ones(B, N_agents, dtype=torch.bool)
    dummy_agent_valid[0, -1] = False

    print("Initializing SmolLMMotionPredictor (GRU decoder, K=6 modes)...")
    model = SmolLMMotionPredictor(model_name="HuggingFaceTB/SmolLM-135M",
                                   num_modes=K, unfreeze_last_n_layers=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    print("Running forward pass...")
    out = model(dummy_input_ids, dummy_attn_mask, dummy_agent_valid)
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, N_agents, K, 60, 2), f"Unexpected shape: {out.shape}"
    print("Forward pass successful!")

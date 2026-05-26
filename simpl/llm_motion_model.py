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
        # key_padding_mask: True means IGNORE that key position
        key_padding_mask = ~agent_valid          # [B, N]
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + out)                # residual + LayerNorm


class SmolLMMotionPredictor(nn.Module):
    def __init__(self,
                 model_name="HuggingFaceTB/SmolLM-135M",
                 num_future_steps=60,
                 num_modes=6,
                 unfreeze_last_n_layers=2,
                 social_num_heads=4,
                 device=None,
                 use_flash_attn=False,
                 dtype=torch.float32):
        super().__init__()

        self.num_future_steps = num_future_steps
        self.num_modes = num_modes

        self.config = AutoConfig.from_pretrained(model_name)

        # Load LLM with specified dtype.
        # Flash Attention 2 has compatibility issues with DDP in some environments,
        # so we default to "sdpa" (PyTorch's native scaled_dot_product_attention),
        # which is also optimized and more stable.
        # low_cpu_mem_usage=False disables Transformers' memory optimization that can
        # conflict with DDP's parameter broadcast.
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

        # Social cross-attention: agents attend to each other after LLM encoding
        self.social_attn = SocialAttention(hidden_dim, num_heads=social_num_heads)

        # MLP prediction head
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_modes * num_future_steps * 2),
        )

        # Always convert social_attn and mlp_head to match LLM dtype.
        # LLM is loaded in `dtype` by AutoModel; these two default to fp32.
        # Without this, the model is mixed-precision (fp32 head + bf16 LLM).
        self.social_attn = self.social_attn.to(dtype)
        self.mlp_head = self.mlp_head.to(dtype)

        if device is not None:
            self.to(device)

    def forward(self, input_ids, attention_mask, agent_valid):
        """
        Args:
            input_ids:      [B, N, L]
            attention_mask: [B, N, L]
            agent_valid:    [B, N]
        Returns:
            predicted_trajs: [B, N, K, T, 2]  K=num_modes, T=num_future_steps
        """
        B, N, L = input_ids.shape

        ids_flat  = input_ids.view(B * N, L)
        mask_flat = attention_mask.view(B * N, L)

        outputs = self.llm(input_ids=ids_flat, attention_mask=mask_flat,
                           output_hidden_states=False)
        hidden = outputs.last_hidden_state           # [B*N, L, H]
        H = hidden.shape[-1]

        mask_exp = mask_flat.unsqueeze(-1).float()
        pooled   = (hidden * mask_exp).sum(dim=1) / \
                   mask_exp.sum(dim=1).clamp(min=1e-9)                      # [B*N, H]

        agent_repr = pooled.view(B, N, H)
        agent_repr = agent_repr * agent_valid.unsqueeze(-1).float()
        agent_repr = self.social_attn(agent_repr, agent_valid)              # [B, N, H]

        K, T = self.num_modes, self.num_future_steps
        pred_flat       = self.mlp_head(agent_repr.view(B * N, H))         # [B*N, K*T*2]
        predicted_trajs = pred_flat.view(B, N, K, T, 2)                    # [B, N, K, T, 2]

        return predicted_trajs


# --- Quick Local Test ---
if __name__ == "__main__":
    N_agents = 6
    B, L     = 2, 128

    K = 6
    dummy_input_ids = torch.randint(0, 49000, (B, N_agents, L))
    dummy_attn_mask = torch.ones((B, N_agents, L), dtype=torch.long)
    dummy_attn_mask[0, -1, 10:] = 0
    dummy_agent_valid = torch.ones(B, N_agents, dtype=torch.bool)
    dummy_agent_valid[0, -1] = False

    print("Initializing Model (Plan A — separate agent encoding, K=6 modes)...")
    model = SmolLMMotionPredictor(model_name="HuggingFaceTB/SmolLM-135M",
                                   num_modes=K, unfreeze_last_n_layers=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")

    print("Running forward pass...")
    out = model(dummy_input_ids, dummy_attn_mask, dummy_agent_valid)
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, N_agents, K, 60, 2), "Output geometry is incorrect."
    print("Forward pass successful!")

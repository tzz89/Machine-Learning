import torch
import torch.nn as nn

# Stop 1.15


class SiglipVisonConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens=None,
        **kwargs,
    ):

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipVisonConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),  # (1, position)
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # [batch, emb_dim, num_patches_h, num_patches_w]
        embeddings = patch_embeds.flatten(2)  # [batch, emb_dim, num_patches]
        embeddings = embeddings.transpose(1, 2)  # [batch, num_patches, emb_dim]
        embeddings = embeddings + self.position_embedding(self.position.ids)
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisonConfig):
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisonConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        # [Batch, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.shape
        query_state = self.q_proj(hidden_states)
        key_state = self.k_proj(hidden_states)
        value_state = self.v_proj(hidden_states)

        query_state = query_state.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_state = key_state.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_state = value_state.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weight = torch.matmul(query_state, key_state.transpose(2, 3)) * self.scale

        if attn_weight.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but are of size {attn_weight.size()}"
            )

        attn_weight = nn.functional.softmax(
            attn_weight, dim=-1, dtype=torch.float32
        ).to(query_state.dtype)
        attn_weight = nn.functional.dropout(
            attn_weight, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weight, value_state)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is of size {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weight


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisonConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds):
        for layer in self.layers:
            input_embeds = layer(input_embeds)
        return input_embeds


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisonConfig):
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm_1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        hidden_states, _ = self.self_atn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisonConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embedding = SiglipVisionEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.Linear(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        hidden_state = self.embedding(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_state)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisonConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        # [Batch, Channels, Height, Width] -> [Batch, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values)

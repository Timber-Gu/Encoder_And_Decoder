import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize embedding weights using normal distribution
        # LLaMA uses a normal distribution with mean 0 and std 0.02, let's keep it that way
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        # LLaMA does not scale embeddings by sqrt(d_model) like vanilla transformers
        return self.embedding(x)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)

        # make sure the dimension is even
        assert head_dim % 2 == 0, "Dimension must be even"

        # create the dimension indices [0, 2, 4, ..., d_model-2]
        # every 2 dimensions are one sub-dimension
        dim_indices = torch.arange(0, head_dim, 2)
        # create the position indices [0, 1, 2, ..., max_len-1]
        pos_indices = torch.arange(max_len)

        # create the sin and cos values for each dimension (for RoPE)
        inv_freq = 1.0 / (10000 ** (dim_indices / head_dim))
        # Use outer product for pos_indices and inv_freq
        pos_enc_a = pos_indices[:, None] * inv_freq[None, :]  # [max_len, d_model/2]
        self.register_buffer("cos", torch.cos(pos_enc_a))  # [max_len, d_model/2]
        self.register_buffer("sin", torch.sin(pos_enc_a))  # [max_len, d_model/2]

    def forward(self, x):
        # This module is used only to hold RoPE buffers (cos/sin).
        # We do not apply RoPE here. RoPE is applied inside attention to q and k.
        return self.dropout(x)
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        output = self.weight * output
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(nn.functional.gelu(self.w1(x)) * self.w3(x)))
    
class RMSNormResidual(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        residual = x.clone()
        x = self.norm(x)
        x = sublayer(x)
        x = self.dropout(x)
        return x + residual

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, rope: RotaryPositionEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, d_model = q.shape
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to q and k after RMSNorm (done by caller) and before attention
        if self.rope is not None:
            # q,k: [B, H, T, d_k] -> pairwise rotate last dim
            sin = self.rope.sin[:seq_len]  # [T, d_k/2]
            cos = self.rope.cos[:seq_len]  # [T, d_k/2]
            # reshape to broadcast over batch and heads
            sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,T,d_k/2]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,d_k/2]

            def apply_rope(t):
                t_reshaped = t.view(batch_size, self.num_heads, seq_len, -1, 2)
                t1 = t_reshaped[..., 0]
                t2 = t_reshaped[..., 1]
                rot1 = t1 * cos - t2 * sin
                rot2 = t1 * sin + t2 * cos
                return torch.stack([rot1, rot2], dim=-1).view(batch_size, self.num_heads, seq_len, -1)

            q = apply_rope(q)
            k = apply_rope(k)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)   
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o(output)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, d_model: int, dropout: float, use_rope: bool = True):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.attention_residual = RMSNormResidual(d_model, dropout)
        self.ffn_residual = RMSNormResidual(d_model, dropout)

    def forward(self, x, mask=True):
        # Apply self-attention with residual connection (pre-norm inside residual)
        x = self.attention_residual(x, lambda x: self.self_attention_block(x, x, x, mask))
        # Apply feed-forward with residual connection
        x = self.ffn_residual(x, lambda x: self.feed_forward_block(x))
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.decoder_blocks = layers

    def forward(self, x, mask=True):
        for layer in self.decoder_blocks:
            x = layer(x, mask)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class LlamaModel(nn.Module):
    def __init__(self, decoder: Decoder, embed: InputEmbedding, projection: ProjectionLayer):  
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.projection_layer = projection

    def forward(self, x, mask=None):
        # Input embedding
        x = self.embed(x)
        # Pass through decoder
        x = self.decoder(x, mask)
        # Project to vocabulary
        return self.projection_layer(x)

def build_llama_model(vocab_size: int, seq_length: int, d_model: int = 4096, N: int = 32, h: int = 32, dropout: float = 0.1, d_ff: int = 11008):
    """
    Build a decoder-only transformer model similar to Llama 3 architecture
    
    Args:
        vocab_size: Size of the vocabulary
        seq_length: Maximum sequence length
        d_model: Model dimension
        N: Number of decoder layers
        h: Number of attention heads
        dropout: Dropout rate
        d_ff: Dimension of feed-forward layer
    """
    # Create embedding layers
    embed = InputEmbedding(vocab_size=vocab_size, d_model=d_model)
    rope = RotaryPositionEmbedding(d_model // h, seq_length, dropout)
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttention(d_model, h, dropout, rope=rope)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention, feed_forward, d_model, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create decoder
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create projection layer
    projection = ProjectionLayer(d_model, vocab_size)
    
    # Create model
    model = LlamaModel(decoder, embed, projection)
    
    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=0.02)
    
    return model

def create_causal_mask(seq_len):
    """Create a causal mask for self-attention"""
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0)  # Add batch dimension [1, seq_len, seq_len]

def main():
    # Model parameters - using a smaller model for demonstration
    vocab_size = 32000  # Llama 3 uses a tokenizer with ~128K tokens
    seq_length = 4096   # Llama 3 supports context lengths of 8K-128K
    d_model = 4096      
    N = 32               
    h = 32              
    dropout = 0.1
    d_ff = 11008        
    
    # Build model
    model = build_llama_model(vocab_size, seq_length, d_model, N, h, dropout, d_ff)
    
    # Create a sample input
    batch_size = 1
    curr_seq_len = 16
    x = torch.randint(0, vocab_size, (batch_size, curr_seq_len))
    
    # Create causal mask
    mask = create_causal_mask(curr_seq_len)
    
    # Forward pass
    output = model(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main()
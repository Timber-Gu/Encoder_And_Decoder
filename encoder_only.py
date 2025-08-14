import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] (only seq_len is used)
        batch_size, seq_len = x.size(0), x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        return self.position_embeddings(position_ids)
    
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=1e-12):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.zeros(1))
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        variance=x.var(dim=-1,keepdim=True,unbiased=False)
        return self.alpha*(x-mean)/(torch.sqrt(variance+self.eps))+self.beta
    
class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.dropout=dropout
        self.w1=nn.Linear(d_model,d_ff)
        self.w2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        return self.w2(self.dropout(nn.functional.gelu(self.w1(x))))

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        # Pre-LN residual: x + Dropout(Sublayer(Norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=num_heads
        self.d_k=d_model//self.h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,q,k,v):
        # q: query tensor [batch_size, seq_len_q, d_model]
        query=self.w_q(q)
        # k: key tensor [batch_size, seq_len_k, d_model]
        key=self.w_k(k)
        # v: value tensor [batch_size, seq_len_v, d_model]
        value=self.w_v(v)
        
        # split the query,key,value into self.h heads
        # [batch_size, seq_len_q, d_model] -> [batch_size, num_heads, seq_len_q, d_k]
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        # calculate the attention scores
        attention_scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(self.d_k)
        # no mask needed for encoder only model
        # apply the softmax function
        attention_probs=nn.functional.softmax(attention_scores,dim=-1)
        # apply the dropout
        attention_probs=self.dropout(attention_probs)
        # calculate the weighted sum of the value
        weighted_sum=torch.matmul(attention_probs,value)
        # concatenate the heads
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size, seq_len_q, d_model]
        weighted_sum = weighted_sum.transpose(1, 2).contiguous()
        weighted_sum = weighted_sum.view(q.shape[0], q.shape[1], self.d_model)
        # apply the linear transformation
        return self.w_o(weighted_sum) # [batch_size, seq_len_q, d_model]

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # Use separate residual connections (with their own LayerNorm) for attention and FFN
        self.attn_residual = ResidualConnection(dropout)
        self.ffn_residual = ResidualConnection(dropout)

    def forward(self, x):
        x = self.attn_residual(x, lambda t: self.self_attention(t, t, t))
        x = self.ffn_residual(x, self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class BERT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, max_len: int, dropout: float, d_ff: int):
        super().__init__()
        self.token_embedding = InputEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len)
        self.encoder = Encoder(d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, dropout=dropout)
        self.output_layer = ProjectionLayer(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [batch_size, seq_len]
        token_embeddings = self.token_embedding(input_ids)            # [batch, seq, d_model]
        position_embeddings = self.position_embedding(token_embeddings)  # [batch, seq, d_model]
        hidden_states = token_embeddings + position_embeddings         # [batch, seq, d_model]
        encoded = self.encoder(hidden_states)                          # [batch, seq, d_model]
        return self.output_layer(encoded)                              # [batch, seq, vocab]

def print_model_parameters(model: nn.Module) -> None:
    total_params = 0
    trainable_params = 0
    print("Parameter details:")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"- {name:60s} shape={tuple(param.shape)} params={num_params:,}")
    non_trainable_params = total_params - trainable_params
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print("\nSummary:")
    print(f"- Total params:        {total_params:,}")
    print(f"- Trainable params:    {trainable_params:,}")
    print(f"- Non-trainable params:{non_trainable_params:,}")
    print(f"- Approx size:         {total_bytes / (1024**2):.2f} MB")

def main():
    # BERT-base configuration
    d_model=768
    num_layers=12
    num_heads=12
    max_len=512
    dropout=0.1
    vocab_size=30522
    d_ff=3072

    model=BERT(vocab_size,d_model,num_layers,num_heads,max_len,dropout,d_ff)

    x=torch.randint(0,vocab_size,size=(1,10))
    logits = model(x)
    print(logits.shape)
    print_model_parameters(model)

if __name__=="__main__":
    main()

    


    

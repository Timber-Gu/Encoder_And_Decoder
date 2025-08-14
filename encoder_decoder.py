import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

# For the Positional Encoding, we use Rotary Position Embedding instead of the Vanilla Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # make sure the dimension is even
        assert d_model % 2 == 0, "Dimension must be even"
        
        # create the dimension indices [0, 2, 4, ..., d_model-2]
        dim_indices = torch.arange(0, d_model, 2)
        
        # create the position indices [0, 1, 2, ..., max_len-1]
        pos_indices = torch.arange(max_len)
        
        # calculate the frequency for each dimension pair (a vector)
        # I know the original paper use a scalar, but in practice, we use a vector
        freqs = 1.0 / (10000 ** (dim_indices.float() / d_model))
        
        # outer product to get the angle for each (position, dimension pair)
        # pos_enc_a's shape: [max_len, d_model/2]
        pos_enc_a = pos_indices[:, None] * freqs[None, :]
        
        # calculate the sin and cos values for each position and dimension pair
        self.register_buffer('sin', pos_enc_a.sin())  # [max_len, d_model/2]
        self.register_buffer('cos', pos_enc_a.cos())  # [max_len, d_model/2]
    
    def forward(self, x):
        # x's shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # get the sin and cos values for the current sequence length
        sin = self.sin[:seq_len]  # [seq_len, d_model/2]
        cos = self.cos[:seq_len]  # [seq_len, d_model/2]
        
        # split the input tensor into two halves, each corresponding to a dimension pair
        # from [batch_size, seq_len, d_model] to [batch_size, seq_len, d_model/2, 2]
        x_reshaped = x.view(batch_size, seq_len, -1, 2)
        
        # separate the two components in the dimension pair
        x1 = x_reshaped[..., 0]  # even dimension
        x2 = x_reshaped[..., 1]  # odd dimension
        
        # apply the rotation transformation
        # rotation matrix: [cos, -sin]
        #                  [sin,  cos]
        rotated_x1 = x1 * cos - x2 * sin  # first component
        rotated_x2 = x1 * sin + x2 * cos  # second component
        
        # combine the two components
        # [batch_size, seq_len, d_model/2, 2] -> [batch_size, seq_len, d_model]
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rotated.view(batch_size, seq_len, -1)
        
        # apply dropout and return
        return self.dropout(rotated)
    
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        # these two parameters help the model to learn the mean and variance of the input
        self.alpha=nn.Parameter(torch.ones(1)) # learnable scale
        self.bias=nn.Parameter(torch.zeros(1)) # learnable bias

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        return self.alpha*(x-mean)/torch.sqrt(var+self.eps)+self.bias

# Here we use the SwiGLU activation function instead of the ReLU in the vanilla Transformer
class FeedForward(nn.Module):
	def __init__(self, d_model: int, d_ff: int, dropout: float):
		super().__init__()
		self.w1 = nn.Linear(d_model, d_ff)   # up projection
		self.w3 = nn.Linear(d_model, d_ff)   # gate
		self.w2 = nn.Linear(d_ff, d_model)   # down projection
		self.dropout = nn.Dropout(dropout)
		self.silu = nn.SiLU()

	def forward(self, x):
		value = self.w1(x)
		gate = self.silu(self.w3(x))         # Swish/SiLU on gate
		return self.w2(self.dropout(value * gate))
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=num_heads
        assert d_model%self.h==0, "d_model must be divisible by num_heads"

        self.d_k=d_model//self.h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        # q: query tensor [batch_size, seq_len_q, d_model]
        query=self.w_q(q)
        # k: key tensor [batch_size, seq_len_k, d_model]
        key=self.w_k(k)
        # v: value tensor [batch_size, seq_len_v, d_model]
        value=self.w_v(v)

        # split the query, key, and value into num_heads heads
        # [batch_size, seq_len_q, d_model] -> [batch_size, num_heads, seq_len_q, d_k]
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        # calculate the attention scores
        scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(self.d_k)

        # apply the mask if provided
        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)
        
        # calculate the attention weights
        attn_weights=torch.softmax(scores,dim=-1)
        # apply the dropout
        attn_weights=self.dropout(attn_weights)
        
        # calculate the output
        output=torch.matmul(attn_weights,value)
        # concatenate the heads
        # [batch_size, num_heads, seq_len_q, d_k] -> [batch_size,seq_len,num_heads,d_k] -> [batch_size, seq_len, d_model]
        output=output.transpose(1,2).contiguous().view(output.shape[0],output.shape[2],-1)
        # apply the final linear transformation
        return self.w_o(output)

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
    
    def forward(self,x,sublayer):
        # add the residual after the normalization
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,feed_forward_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection1=ResidualConnection(dropout)
        self.residual_connection2=ResidualConnection(dropout)

    def forward(self,x,mask=True):
        # we first apply the self-attention block then add the residual connection
        x=self.residual_connection1(x,lambda x:self.self_attention_block(x,x,x,mask))
        # then apply the feed-forward block
        x=self.residual_connection2(x,lambda x:self.feed_forward_block(x))
        return x

# There are N encoder blocks in the encoder (correspond to N layers in the original paper)
class Encoder(nn.Module):
    # takes list of encoder block as input
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,mask=True):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttention,cross_attention_block:MultiHeadAttention,feed_forward_block:FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connection1=ResidualConnection(dropout)
        self.residual_connection2=ResidualConnection(dropout)
        self.residual_connection3=ResidualConnection(dropout)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        # self attention
        x=self.residual_connection1(x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        # cross attention (q,k,v). k,v from Encoder. q from Decoder
        x=self.residual_connection2(x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection3(x,lambda x:self.feed_forward_block(x))
        return x

class Decoder(nn.Module):
    # takes list of decoder blocks as input
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed: InputEmbedding,tgt_embed: InputEmbedding,src_pos: PositionalEncoding,tgt_pos: PositionalEncoding, projection:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection

    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_length:int,tgt_seq_length:int,d_model:int=512,N:int=6,h:int=8,dropout:float=0.1,d_ff:int=2048):
    src_embed=InputEmbedding(vocab_size=src_vocab_size,d_model=d_model)
    tgt_embed=InputEmbedding(vocab_size=tgt_vocab_size,d_model=d_model)
    src_pos=PositionalEncoding(d_model,src_seq_length,dropout=dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_length,dropout=dropout)

    # encoder block
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttention(d_model,h,dropout)
        encoder_feed_forward_block=FeedForward(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,encoder_feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    # decoder block
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttention(d_model,h,dropout)
        decoder_feed_forward_block=FeedForward(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,decoder_feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder=Encoder(encoder_blocks)
    decoder=Decoder(decoder_blocks)

    projection=ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer

def main():
    src_vocab_size=10000
    tgt_vocab_size=20000
    src_seq_length=100
    tgt_seq_length=100
    d_model=512
    N=6
    h=8
    dropout=0.1
    d_ff=2048

    transformer=build_transformer(src_vocab_size,tgt_vocab_size,src_seq_length,tgt_seq_length,d_model,N,h,dropout,d_ff)

    src=torch.randint(0,src_vocab_size,size=(1,src_seq_length))
    tgt=torch.randint(0,tgt_vocab_size,size=(1,tgt_seq_length))

    src_mask=torch.ones(1,1,src_seq_length)
    tgt_mask=torch.ones(1,tgt_seq_length,tgt_seq_length)

    encoder_output=transformer.encode(src,src_mask)
    decoder_output=transformer.decode(encoder_output,src_mask,tgt,tgt_mask) 
    output=transformer.project(decoder_output)

    print(output.shape)

if __name__=="__main__":
    main()
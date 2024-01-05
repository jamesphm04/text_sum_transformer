import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    #initialize dimension of the model
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #Embedding vector size
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a matrix of shape(seq_len, d_model) because this will represent the word itself(d_mode) x its position in the sequenc(seq_len)
        pe = torch.zeros(seq_len, d_model)
        # try to calculate the pe by the following equation: pe = sin(posisition/(10000^(2i/d_model)))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # using log because it more stable ???
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)) #(d_model/2) because we set step = 2
        # apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term) #sin(position*(10000**(2i/d_model)))
        # apply cosin to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) #cosin(position*(10000**(2i/d_model)))
        # apply a batch dimension to the positional embedding at the beginning(index 0)
        pe = pe.unsqueeze(0)
        # register the positional encoding as a buffer, basiclly pe will be saved similar to weight parametter when you save the model 
        self.register_buffer('pe', pe)
        # self.pe = pe #???
        
    def forward(self, x):
        # concate x with positional tensor up to x.shape[1] only, require_grad_ = False dont allow this to be learnable
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features: int,  eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps # avoid devide by or close to zero -> explode
        self.alpha = nn.Parameter(torch.ones(features)) # learnable Multiplied
        self.bias = nn.Parameter(torch.ones(features)) # learnable Added
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean)/(std + self.eps) + self.bias
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
        
    def forward(self, x):
        # x: (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h # number of heads
        # Make sure d_module is divisible by h 
        assert d_model % h == 0 , 'd_model is not divisible by h'
        
        self.d_k = d_model//h # Dimension of vector seen by each head
        
        self.w_q = nn.Linear(d_model, d_model, bias=False) #Wq (d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False) #Wk (d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model, bias=False) #Wv (d_model, d_model)
        
        # Weight of the output
        self.w_o = nn.Linear(d_model, d_model, bias=False) #Wo (d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # apply formula
        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # write a very low value (indicating -inf) to the positions where mask == 0 
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) #(batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) -> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
        
    def forward(self, q, k, v, mask):
        #(batch, seq_len, d_model)*(d_model, d_model) -> (batch, seq_len, d_model)
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)
        
        #(batch, seq_len, d_model) / h -> (batch, seq_len, h, d_k).transpose -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # view will change the dim without changing the data, self.h * self.d_k will give back d_model, transpose will swap pos 1 and 2
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, dropout=self.dropout)
        
        # combine all the heads together 
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k).contiguous (basically execute it inplacely) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # multiply by Wo
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features=features)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        # add src_mask here in order to make <pad> not interact with others
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # Res_1st: first layer is x, second layer is the output of multiheadAttentionBlock, which is receive x, x, x as q, k, v
        x = self.residual_connections[1](x, self.feed_forward_block) # Res_2nd: first layer is the output of Res_1st, second layer is feed_forward
        return x
    
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norms = LayerNormalization(features)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norms(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
        
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) -> None:
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.project(decoder_output)
    
# put it all together 

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int=50, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # src_vocab_size : number of unique words in src vocab
    # tgt_vocab_size : number of unique words in tgt vocab
    # src_seq_len : max number of words in a src sentence
    # tgt_seq_len : max number of words in a tgt sentence
    # d_model : dimension of tokens - size of vectors used to represent each token 
    # N : number of hidden encoder/decoder layers 
    # h : number of heads in MultiAttentionLayer
    # dropout : propability of dropout
    # d_ff : dimension of feedforward - size of hidden layer in Feedforwardlayer
    
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create the encoder blocks 
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # create the projection layer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
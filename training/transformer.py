#transformer->lstm->ppo 
#ppo means proximal policy optimisation
#transformer (multi head attention) would let out 15 readings from 4 sensors influence each other
#lstm adds temporal memory at each timestep
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} must be divisible by nhead {nhead}" 
        #crash the program if this condition is false
        self.d_model = d_model #32
        self.nhead = nhead #4
        self.d_k = d_model // nhead 
        self.W_q = nn.Linear(d_model, d_model) #query
        #4 different projection matrices, nn.Linear(i/p, o/p)
        self.W_k = nn.Linear(d_model, d_model) #key
        self.W_v = nn.Linear(d_model, d_model) #value
        self.W_o = nn.Linear(d_model, d_model) #output
        self.scale = self.d_k ** 0.5  #to normalise 
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None 
    def forward(self, x, store_attention=False):
        #x has the sensor tokens, each of 32 dim
        B, S, _ = x.shape #S is number of sensor features
        Q = self.W_q(x) 
        K = self.W_k(x)  
        V = self.W_v(x)
        #multi head split
        Q = Q.view(B, S, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(B, S, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(B, S, self.nhead, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / self.scale  #attention scores O(n2)
        attn = torch.softmax(scores, dim=-1) 
        attn = self.dropout(attn)
        if store_attention:
            self.attn_weights = attn.detach().cpu()
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model) #better to align tensor as contiguous memory after transpose, cuz view() requires contig
        return self.W_o(out)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), #32 to 128 expansion inside the feed forward network
            nn.GELU(), #GELU allows small negative values, RELU converts all into 0
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), 
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model) #pre-norm, paper said postnorm
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, store_attention = False):
        #layer1
        normed = self.norm1(x)
        attended = self.attention(normed, store_attention)
        x = x + attended #x = prev + new learned info
        #layer2
        normed = self.norm2(x)
        fed_forward = self.ffn(normed)
        x = x+fed_forward #x = prev + processed 
        return x 
class TransformerFeatures(BaseFeaturesExtractor):
    #15 features
    FEATURE_NAMES = [
        "us_left", "us_center", "us_right",
        "speed_pwm", "lane",
        "obstacle_bin", "obstacle_dist",
        "oncoming_bin", "oncoming_dist", "oncoming_speed",
        "lhs_safe", "rhs_safe",
        "traffic_signal", "obstacle_pass", "stop_sign"
    ]
    def __init__(self, observation_space, features_dim=128,d_model=32, nhead=4, num_layers=2,dim_ff=128, dropout=0.1):
        super().__init__(observation_space, features_dim) #observation space is required by sb3 that uses gym to see what data agen gets from env
        self.obs_dim = observation_space.shape[0] 
        self.d_model = d_model #dimensions per token=32
        self.feature_embedding = nn.Linear(1, d_model) #from the input string, 1 token is 1 scalar, we convert scalar to a vector with 32 dimensions
        self.pos_encoding = nn.Parameter(torch.randn(1, self.obs_dim, d_model) * 0.02) #positional encoding, ensuring the context is different even with similar inputs
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(self.obs_dim * d_model, features_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim * 2, features_dim))
        self._initialize_weights()
    def _initialize_weights(self): #xavier's initialisation to maintain the randomness of the initial values what we start with (balanced variance)
        for p in self.parameters():
            if p.dim() > 1: #only applies to weight matrices which have dim more than 1
                nn.init.xavier_uniform_(p)
    def forward(self, observations, store_attention=False):
        B = observations.shape[0]
        x = observations.unsqueeze(-1)   # (B, 15) → (B, 15, 1) , unsqueeze to convert a scalar into a token by adding a new dimension at last posn
        x = self.feature_embedding(x)    # (B, 15, 1) → (B, 15, 32) , linear projection, token wise
        x = x + self.pos_encoding 
        for block in self.blocks: #blocks are repeated transformer layers
            x = block(x, store_attention=store_attention)
        x = x.reshape(B,-1)
        return self.output_proj(x)







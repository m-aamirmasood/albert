import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense LeakyReLU BatchNormalization
from keras.layers import Conv2D Conv2DTranspose Reshape Flatten

class LayerNorm(Layer):
    def __init__(self cfg variance_epsilon=1e-12):
        super().__init__()
        self.gamma = K.ones(cfg.hidden)
        self.beta  = K.zeros(cfg.hidden)
        self.variance_epsilon = variance_epsilon

    def forward(self x):
        u = K.mean(x keepdim=True)
        s = K.mean(K.pow((x - u)2) keepdim=True)
        x = (x - u) / K.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

def gelu(x):
    Px = 0.5 * (1.0 + K.tanh((K.sqrt(2 / (22/7)) * (x + 0.044715 * K.pow(x 3)))))
    return x * Px

class Embeddings(Layer):
    def __init__(self cfg):
        super().__init__()
        self.tok_embed1 = K.Embedding(cfg.vocab_size cfg.embedding)
        self.tok_embed2 = K.Linear(cfg.embedding cfg.hidden)

        self.pos_embed = K.Embedding(cfg.max_len cfg.hidden) # position embedding
        self.seg_embed = K.Embedding(cfg.n_segments cfg.hidden) # segment(token type) embedding

        self.norm = LayerNorm(cfg)

    def forward(self x seg):
        seq_len = x.size(1)
        pos = K.arange(seq_len dtype=\float32\)
        pos = K.expand_dims(pos axis = -1)

        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(e)

class MultiHeadedSelfAttention(Layer):
    def __init__(self cfg):
        super().__init__()
        self.proj_q = Dense(cfg.hidden cfg.hidden)
        self.proj_k = Dense(cfg.hidden cfg.hidden)
        self.proj_v = Dense(cfg.hidden cfg.hidden)
        self.scores = None
        self.n_heads = cfg.n_heads

    def forward(self x mask):
        q, k, v = self.proj_q(x) self.proj_k(x) self.proj_v(x)
        q, k, v = (split_last(x (self.n_heads -1)).transpose(1 2) for x in [q k v])
        scores = q @ k.transpose(-2 -1) / K.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[: None None :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = K.softmax(scores dim=-1)
        h = (scores @ v).transpose(1 2).contiguous()
        h = merge_last(h 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(Layer):
    def __init__(self cfg):
        super().__init__()
        self.fc1 = Dense(cfg.hidden cfg.hidden_ff)
        self.fc2 = Dense(cfg.hidden_ff cfg.hidden)

    def forward(self x):
        return self.fc2(gelu(self.fc1(x)))

class Transformer(Layer):
    def __init__(self cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = Dense(cfg.hidden cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self x seg mask):
        h = self.embed(x seg)

        for _ in range(self.n_layers):
            h = self.attn(h mask)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))

        return h

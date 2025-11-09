import torch
import torch.nn as nn
import math

# --- (1) 相对位置编码 (RPE) 模块 ---
# 我们将实现 T5 风格的相对位置偏置
# T5: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

class T5RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, nhead=4):
        super(T5RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.nhead = nhead
        # 相对位置偏置的嵌入表
        # 我们需要 num_buckets 个桶
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.nhead)

    @staticmethod
    def _t5_bucket(relative_position, num_buckets=32, max_distance=128):
        # T5 论文中的分桶算法
        ret = 0
        n_buckets = num_buckets // 2
        
        # 负位置
        ret += (relative_position < 0).long() * n_buckets
        relative_position = torch.abs(relative_position)

        # 桶
        max_exact = n_buckets // 2
        is_small = relative_position < max_exact

        # 对数分桶，适用于较大的距离
        val_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (n_buckets - max_exact)
        ).long()
        
        # 确保不越界
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, query_len, key_len, device):
        # 计算相对位置矩阵
        q_pos = torch.arange(query_len, dtype=torch.long, device=device)
        k_pos = torch.arange(key_len, dtype=torch.long, device=device)
        
        # [query_len, key_len]
        relative_position = k_pos[None, :] - q_pos[:, None]

        # 分桶
        bucketed_positions = self._t5_bucket(
            relative_position, 
            num_buckets=self.num_buckets, 
            max_distance=self.max_distance
        )
        
        # 查找偏置
        # [query_len, key_len, nhead]
        bias = self.relative_attention_bias(bucketed_positions)
        
        # [1, nhead, query_len, key_len] (准备好广播到 [batch, nhead, query_len, key_len])
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        return bias

# --- (2) 绝对位置编码 (现在是可选的) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- (3) 修改: MultiHeadAttention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # ... (d_k, nhead, d_model 保持不变) ...
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    # (3.1) 修改: forward 签名，接受 pos_bias
    def forward(self, query, key, value, mask=None, pos_bias=None):
        batch_size = query.size(1)
        query_len = query.size(0)
        key_len = key.size(0)
        
        q = self.q_linear(query).view(query_len, batch_size, self.nhead, self.d_k).transpose(0, 1)
        k = self.k_linear(key).view(key_len, batch_size, self.nhead, self.d_k).transpose(0, 1)
        v = self.v_linear(value).view(key_len, batch_size, self.nhead, self.d_k).transpose(0, 1)
        
        # [batch_size, nhead, seq_len, d_k]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # (3.2) 计算分数
        # [batch_size, nhead, query_len, key_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # (3.3) 添加 RPE 偏置
        if pos_bias is not None:
            # pos_bias shape [1, nhead, query_len, key_len]
            scores = scores + pos_bias
            
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1) # [1, 1, query_len, key_len]
            # [batch_size, nhead, query_len, key_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # [batch_size, nhead, query_len, d_k]
        context = torch.matmul(attn, v)
        
        # [batch_size, query_len, nhead, d_k] -> [batch_size, query_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        
        output = self.out_linear(context)
        
        # [query_len, batch_size, d_model]
        return output.transpose(0, 1)

# PositionwiseFeedForward 保持不变
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# --- (4) 修改: EncoderBlock ---
class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # (4.1) 修改: forward 签名，接受 pos_bias
    def forward(self, src, src_mask=None, pos_bias=None):
        src_attn = self.self_attn(src, src, src, src_mask, pos_bias=pos_bias)
        src = src + self.dropout1(src_attn)
        src = self.norm1(src)
        
        src_ffn = self.feed_forward(src)
        src = src + self.dropout2(src_ffn)
        src = self.norm2(src)
        return src

# --- (5) 修改: DecoderBlock ---
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    # (5.1) 修改: forward 签名，接受 pos_bias
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, self_pos_bias=None, cross_pos_bias=None):
        tgt_attn = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, pos_bias=self_pos_bias)
        tgt = tgt + self.dropout1(tgt_attn)
        tgt = self.norm1(tgt)
        
        tgt_cross_attn = self.cross_attn(tgt, memory, memory, mask=memory_mask, pos_bias=cross_pos_bias)
        tgt = tgt + self.dropout2(tgt_cross_attn)
        tgt = self.norm2(tgt)
        
        tgt_ffn = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(tgt_ffn)
        tgt = self.norm3(tgt)
        return tgt

# --- (6) 修改: TransformerSeq2Seq ---
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_encoder_layers, num_decoder_layers, dropout=0.1, 
                 use_rpe=False, rpe_buckets=32, rpe_max_dist=128): # (6.1) 新增 RPE 参数
        super(TransformerSeq2Seq, self).__init__()
        self.d_model = d_model
        self.use_rpe = use_rpe
        
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        if self.use_rpe:
            # (6.2) RPE 模块 (一个模块在所有层之间共享)
            print("Using Relative Position Encoding (RPE)")
            self.rpe_bias = T5RelativePositionBias(
                num_buckets=rpe_buckets, 
                max_distance=rpe_max_dist, 
                nhead=nhead
            )
        else:
            # (6.3) 绝对位置编码 (如果不用 RPE)
            print("Using Absolute Positional Encoding")
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, nhead, d_ff, dropout) for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout_embed = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # ... (保持不变) ...
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.tgt_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        # ... (保持不变) ...
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, tgt_causal_mask=None, memory_padding_mask=None):
        src_len = src.size(0)
        tgt_len = tgt.size(0)
        device = src.device

        # 1. 嵌入
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # (6.4) RPE 或 绝对PE 的分支
        if self.use_rpe:
            # RPE: 计算偏置
            src_pos_bias = self.rpe_bias(src_len, src_len, device)
            tgt_pos_bias = self.rpe_bias(tgt_len, tgt_len, device)
            cross_pos_bias = self.rpe_bias(tgt_len, src_len, device)
        else:
            # 绝对PE: 添加编码
            src_embed = self.pos_encoder(src_embed)
            tgt_embed = self.pos_encoder(tgt_embed)
            src_pos_bias, tgt_pos_bias, cross_pos_bias = None, None, None
            
        src_embed = self.dropout_embed(src_embed)
        tgt_embed = self.dropout_embed(tgt_embed)

        # 2. Encoder
        memory = src_embed
        for layer in self.encoder:
            memory = layer(memory, src_mask=src_padding_mask, pos_bias=src_pos_bias) # (6.5) 传递偏置
            
        # 3. Decoder
        output = tgt_embed
        for layer in self.decoder:
            output = layer(
                output, 
                memory, 
                tgt_mask=tgt_causal_mask, 
                memory_mask=memory_padding_mask,
                self_pos_bias=tgt_pos_bias, # (6.6) 传递偏置
                cross_pos_bias=cross_pos_bias
            )
            
        # 4. Final Output
        output = self.fc_out(output)
        return output
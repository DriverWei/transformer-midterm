import torch
from torch.utils.data import Dataset
import requests
import os

class CharDataset(Dataset):
    def __init__(self, block_size, data_dir="./data"):
        self.block_size = block_size # seq_len
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path = os.path.join(data_dir, "input.txt")
        
        if not os.path.exists(data_path):
            os.makedirs(data_dir, exist_ok=True)
            print("Downloading Tiny Shakespeare dataset...")
            r = requests.get(data_url)
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(r.text)

        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # --- (1) 修改: 添加 <SOS> 标记 ---
        self.chars = sorted(list(set(self.text)))
        
        # 添加特殊 token
        self.SOS_TOKEN = '<SOS>'
        # 确保 <SOS> 不在原始字符集中 (在本项目中它肯定不在)
        if self.SOS_TOKEN not in self.chars:
            self.chars.insert(0, self.SOS_TOKEN) # 将 <SOS> 放在词汇表开头
            
        self.vocab_size = len(self.chars)
        
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # 获取 <SOS> token 的 ID
        self.sos_id = self.stoi[self.SOS_TOKEN]
        self.sos_tensor = torch.tensor([self.sos_id], dtype=torch.long)
        
        # 编码整个数据集 (不包含 <SOS>)
        # 我们只在构造 item 时动态使用 <SOS>
        self.data = torch.tensor([self.stoi[ch] for ch in self.text if ch in self.stoi and ch != self.SOS_TOKEN], dtype=torch.long)
        
        print(f"Dataset updated. Vocab size (including <SOS>): {self.vocab_size}")

    def __len__(self):
        # 保持不变
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # --- (2) 修改: 实现正确的 Teacher Forcing ---
        
        # src_chunk (原始序列)
        # e.g., "hello" (len=L)
        src_chunk = self.data[idx : idx + self.block_size]
        
        # tgt_chunk_full (完整的目标反转序列)
        # e.g., "olleh" (len=L)
        tgt_chunk_full = torch.flip(src_chunk, dims=[0])

        # tgt_input (解码器输入: <SOS> + 目标序列的前 L-1 个)
        # e.g., [<SOS>, 'o', 'l', 'l', 'e'] (len=1 + L-1 = L)
        tgt_input = torch.cat([self.sos_tensor, tgt_chunk_full[:-1]])
        
        # tgt_output (解码器目标: 完整的目标序列)
        # e.g., ['o', 'l', 'l', 'e', 'h'] (len=L)
        tgt_output = tgt_chunk_full

        return src_chunk, tgt_input, tgt_output

    def get_vocab_size(self):
        return self.vocab_size

    def encode(self, string):
        return [self.stoi[c] for c in string]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
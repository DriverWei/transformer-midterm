import torch
import random
import numpy as np
import os
import matplotlib
# 设置 matplotlib 后端为 'Agg'，使其可以在没有 GUI 的服务器上运行
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, path):
    print(f"Saving model checkpoint to {path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }, path)

def load_checkpoint(model, optimizer, path):
    print(f"Loading model checkpoint from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# --- (1) 绘图函数 ---
def plot_metric(steps, values, metric_name, save_dir):
    """
    为单个指标绘制步级曲线并保存到单独的文件。
    
    steps: (list) X 轴的步数
    values: (list) Y 轴的指标值
    metric_name: (str) 指标名称 (e.g., 'train_loss', 'val_loss')
    save_dir: (str) 保存图像的目录
    """
    if not values:
        print(f"Skipping plot for {metric_name} as no data is available.")
        return
        
    save_path = os.path.join(save_dir, f"{metric_name}_step.png")
    print(f"Saving step-wise plot to {save_path}")
    
    fig = plt.figure(figsize=(10, 5))
    
    # 使用标记 'o' 和更小的 'markersize' 来绘制验证集
    if 'val' in metric_name:
        plt.plot(steps, values, label=metric_name, marker='o', markersize=2, linestyle='--')
    else:
        plt.plot(steps, values, label=metric_name)
    
    plt.title(f"{metric_name.replace('_', ' ').title()} vs. Training Steps")
    plt.xlabel('Training Steps')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig) # 释放内存
# 从零实现Transformer课程作业

本项目为《大模型基础与应用》课程的期中作业。

本项目从零开始，仅使用 PyTorch 实现了一个完整的 Transformer Encoder-Decoder 架构，未依赖 `torch.nn.Transformer` 等高级封装。

核心实现包括：
* 完整的 Encoder-Decoder 架构。
* 标准的缩放点积注意力、多头注意力、FFN、残差与层归一化。
* **绝对位置编码 (Absolute Positional Encoding)**。
* **相对位置编码 (Relative Positional Encoding, RPE)** ：实现了T5风格的可学习偏置。
* **训练稳定性技巧**：实现了 AdamW、梯度裁剪 和 `OneCycleLR` 学习率调度器。
* **消融实验框架**：所有高级功能均可通过命令行标志控制，便于进行对比实验。

为了验证完整的 Encoder-Decoder 架构，我们在 Tiny Shakespeare 数据集上构建了一个“序列反转”（Sequence Reversal）的 Seq2Seq 任务。

## 1. 项目结构

```
transformer-midterm/
├── checkpoints/          # 存放训练好的模型检查点 (.pth)
├── data/                 # 存放下载的数据集 (input.txt)
├── results/              # 存放生成的图表 (.png)
├── scripts/
│   └── run.sh            # (核心) 可复现的运行脚本
├── src/
│   ├── dataset.py        # CharDataset 数据集类 (含<SOS>和Teacher Forcing)
│   ├── model.py          # TransformerSeq2Seq 模型架构 (含RPE)
│   └── utils.py          # 辅助函数 (绘图, 随机种子, 模型保存)
├── train.py              # 主训练脚本 (含argparse)
└── requirements.txt      # 依赖列表
```

## 2. 环境配置与安装

本项目在 `Python 3.8` 和 `PyTorch 2.1.0` (CUDA 12.1) 环境下开发和测试。

**（1）创建 Conda 虚拟环境**
```bash
conda create -n transformer_midterm python=3.8
conda activate transformer_midterm
```

**（2）安装依赖**
所有依赖项均在 `requirements.txt` 中：
```bash
pip install -r requirements.txt
```

## 3. 实验复现

### 3.1 数据集

本项目使用 Tiny Shakespeare 数据集。

**您无需手动下载**。首次运行时，`src/dataset.py` 脚本会检查数据是否存在，如果不存在，将自动从网络下载数据集到 `--data_dir` 指定的目录中。

### 3.2 路径修改

本项目中的所有路径均通过命令行参数传递。`scripts/run.sh` 中硬编码了默认的实验路径。

**在运行前，请务必打开 `scripts/run.sh` 文件**，并修改以下三个参数为您自己环境中的绝对路径：
* `--data_dir`
* `--results_dir`
* `--checkpoint_dir`

### 3.3 运行基线实验

基线实验配置为：**启用** RPE、**启用** AdamW、**启用** 学习率调度、**启用** 梯度裁剪。

要**精确复现**报告中的基线结果，请执行：

```bash
./scripts/run.sh
```

`run.sh` 脚本中的**精确命令**如下：
```bash
#!/bin/bash
set -e

echo "Starting Transformer (Seq2Seq) training with RPE and Stability Tricks..."

python train.py \
    --d_model 256 \
    --nhead 4 \
    --d_ff 1024 \
    --num_layers 3 \
    --batch_size 32 \
    --lr 3e-4 \
    --epochs 1 \
    --block_size 128 \
    --dropout 0.1 \
    --seed 42 \
    --save_interval 5 \
    \
    --data_dir "./data" \
    --results_dir "./results" \
    --checkpoint_dir "./checkpoints" \
    \
    --use_rpe \
    --use_scheduler \
    --use_grad_clip \
    --grad_clip 1.0 \
    --optimizer_type 'adamw' \
    --eval_step 250

echo "Training finished."
```

### 3.4 运行消融实验

要进行消融实验，只需修改 `scripts/run.sh` 中的命令行标志。

**关键消融标志：**
* `--use_rpe`：控制是否使用相对位置编码。
* `--use_scheduler`：控制是否使用 `OneCycleLR` 学习率调度器。
* `--use_grad_clip`：控制是否使用梯度裁剪。
* `--optimizer_type`：可设为 `'adamw'` 或 `'adam'`。
* `--d_ff`：可修改此值进行超参数敏感性分析（例如改为 `512`）。

**示例：运行“无RPE（APE）”的消融实验**

1.  打开 `scripts/run.sh`。
2.  **删除** `--use_rpe` 这一行。
3.  （推荐）修改 `--results_dir` 和 `--checkpoint_dir` 到一个新路径（例如 `.../results_no_rpe`），以防覆盖基线结果。
4.  保存并运行 `./scripts/run.sh`。

## 4. 实验结果

所有训练图表（.png文件）将保存在 `--results_dir` 指定的目录中。

训练完成后，您应在结果目录中看到 **4** 张图表：
* `train_loss_step.png` (训练损失 vs. 步数)
* `val_loss_step.png` (验证损失 vs. 步数)
* `learning_rate_step.png` (学习率 vs. 步数)
* `gradient_norm_step.png` (梯度范数 vs. 步数)

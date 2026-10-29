"""cifar10_linear.py — 编程作业：CIFAR-10 线性分类器权重可视化

背景
----
线性分类器训练完成后，权重矩阵 W 的形状为 (10, 3072)。
将每行权重 reshape 为 (3, 32, 32) 并归一化到 [0,1]，
即可还原为一张图像，直观展示分类器为每个类别学习到的"模板"。

任务
----
完成 plot_templates 函数中的权重图像还原与显示逻辑。

输出（保存到 --save 目录）
--------------------------
  training_curve.png   — 训练 loss 与 acc 曲线
  weight_templates.png — 每个类别的权重模板图像

用法
----
    python cifar10_linear.py
    python cifar10_linear.py --save my_dir
    python cifar10_linear.py --epochs 5 --lr 0.05
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 的 10 个类别名称
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 Linear Classifier")
    p.add_argument("--save",   type=str,   default="outputs")  # 输出目录
    p.add_argument("--epochs", type=int,   default=10)          # 训练轮数
    p.add_argument("--lr",     type=float, default=0.001)       # 学习率
    p.add_argument("--batch",  type=int,   default=128)         # 批大小
    p.add_argument("--seed",   type=int,   default=42)          # 随机种子
    return p.parse_args()


def load_cifar10(data_root, batch_size):
    """下载并加载完整 CIFAR-10 数据集，像素值归一化到 [0, 1]。"""
    tf = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"训练样本: {len(train_set)}  |  测试样本: {len(test_set)}")
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一轮，返回每个 batch 的 step 编号、loss 和累计准确率。"""
    model.train()
    steps, losses, accs = [], [], []
    correct = total = 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        # 将图像展平为一维向量，送入设备
        imgs   = imgs.view(imgs.size(0), -1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 统计累计准确率
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        steps.append(batch_idx)
        losses.append(loss.item())
        accs.append(correct / total)
    return steps, losses, accs


@torch.no_grad()
def evaluate(model, loader, device):
    """在测试集上计算整体准确率。"""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.view(imgs.size(0), -1).to(device)
        labels = labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def plot_curves(all_steps, all_losses, all_accs, boundaries, save_path):
    """绘制训练 loss 与 acc 曲线，用虚线标注每轮 epoch 的分界。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(all_steps, all_losses, color="#e74c3c", linewidth=0.8, label="Train Loss")
    ax2.plot(all_steps, all_accs,   color="#3498db", linewidth=0.8, label="Train Acc")

    # 在每个 epoch 末尾画竖向虚线
    for ep, b in enumerate(boundaries, start=1):
        for ax in (ax1, ax2):
            ax.axvline(x=b, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        ax1.text(b + 1, max(all_losses) * 0.98, f"ep{ep}", fontsize=8, color="gray", va="top")
        ax2.text(b + 1, 0.98,                   f"ep{ep}", fontsize=8, color="gray", va="top")

    ax1.set_xlabel("Step"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy"); ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("CIFAR-10 Linear Classifier — Training Curves", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 -> {save_path}")


def plot_templates(model, save_path):
    """【填空】将 10 个类别的权重向量还原为图像，排列成 2×5 网格。

    要求：在循环体内完成每个类别权重的提取、归一化与显示：
      1. 从 W 中取出第 idx 个类别的权重，reshape 为 (3, 32, 32)
      2. 对该权重做 min-max 归一化，将值域映射到 [0, 1]
      3. 将通道顺序从 (C, H, W) 转换为 (H, W, C)，用 ax.imshow 显示

    提示：
      - W 的形状为 (10, 3072)，已在函数开头 reshape 为 (10, 3, 32, 32)
      - 归一化公式：t = (t - t.min()) / (t.max() - t.min() + 1e-8)
      - np.transpose(t, (1, 2, 0)) 将 (C,H,W) → (H,W,C)
      - ax.imshow 显示图像
    """
    # model.weight 形状为 (10, 3072)，reshape 为 (10, C, H, W)
    W = model.weight.data.cpu().numpy().reshape(10, 3, 32, 32)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("CIFAR-10 Linear Classifier - Weight Templates", fontsize=13)
    for idx, ax in enumerate(axes.flat):
        ### 代码开始处 ###
        t = W[idx]  # 取出第idx个类别的权重，形状(3, 32, 32)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)  # min-max归一化到[0,1]
        img = np.transpose(t, (1, 2, 0))  # 转换为(H, W, C)
        ax.imshow(img)
        ### 代码结束处 ###
        ax.set_title(CLASSES[idx], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 -> {save_path}")


def main():
    args = parse_args()

    # 确定输出目录和数据目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, args.save)
    os.makedirs(out_dir, exist_ok=True)
    data_root  = os.path.join(script_dir, "..", "data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    train_loader, test_loader = load_cifar10(data_root, args.batch)

    # 模型：单个全连接层，输入 3072 维，输出 10 类
    model     = nn.Linear(3 * 32 * 32, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    all_steps, all_losses, all_accs = [], [], []
    boundaries  = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        steps, losses, accs = train_epoch(model, train_loader, criterion, optimizer, device)
        # 将本轮 step 编号偏移到全局坐标
        all_steps  += [global_step + s for s in steps]
        all_losses += losses
        all_accs   += accs
        global_step += len(steps)
        boundaries.append(global_step - 1)

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={losses[-1]:.4f}  "
              f"train_acc={accs[-1]:.3f}  "
              f"test_acc={test_acc:.3f}")

    # 绘图并保存，epoch 分界线去掉最后一条（训练已结束）
    plot_curves(all_steps, all_losses, all_accs,
                boundaries[:-1],
                os.path.join(out_dir, "training_curve.png"))
    plot_templates(model, os.path.join(out_dir, "weight_templates.png"))


if __name__ == "__main__":
    main()

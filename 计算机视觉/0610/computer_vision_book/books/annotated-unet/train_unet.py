# 教学演示：U-Net 医学图像分割
# 数据集：Kvasir-SEG（息肉分割），损失函数：Dice Loss
# 支持早停机制（--patience），默认训练 100 轮，结果保存到 --save 目录

import os
import argparse
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# ── 固定随机种子确保可复现 ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# DuckNet 论文使用的随机种子（用于数据集划分，与论文结果对齐）
DUCK_SEED = 58800


# ── 数据集 ──────────────────────────────────────────────────────────
class KvasirDataset(Dataset):
    """Kvasir-SEG 息肉图像分割数据集"""

    def __init__(self, img_dir, mask_dir, img_size=256, img_paths=None):
        """
        img_paths: 可选，传入预先划分好的 Path 列表（用于 train/val/test 分割）
                   若为 None，则自动扫描 img_dir 下所有图像
        """
        if img_paths is not None:
            # 使用外部传入的路径列表（用于 train/val/test 分割后的子集）
            self.img_paths = img_paths
        else:
            img_dir = Path(img_dir)
            self.img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
            self.img_paths = sorted(self.img_paths)
            if not self.img_paths:
                raise FileNotFoundError(
                    f"在 {img_dir} 中未找到图像文件，请先运行:\n"
                    f"  unzip data/kvasir-seg.zip -d data/"
                )
        self.mask_dir = Path(mask_dir)
        # 图像预处理：调整大小、转为张量、归一化
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        # 掩码预处理：调整大小、转为张量（0/1 二值）
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / img_path.name
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return self.img_transform(img), (self.mask_transform(mask) > 0.5).float()


# ── U-Net 模型 ──────────────────────────────────────────────────────
def double_conv(in_ch, out_ch):
    """两层 3×3 卷积 + BN + ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """经典 U-Net，4 层下采样 + 4 层上采样"""

    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__()
        # 编码器（下采样路径）
        self.enc1 = double_conv(in_channels, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.enc4 = double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        # 瓶颈层
        self.bottleneck = double_conv(base * 8, base * 16)
        # 解码器（上采样路径）
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = double_conv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = double_conv(base * 2, base)
        # 输出层（1×1 卷积）
        self.out_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # 瓶颈
        b = self.bottleneck(self.pool(e4))
        # 解码（跳跃连接）
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# ── Dice Loss ───────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Dice Loss，常用于医学图像分割"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


# ── 工具函数 ─────────────────────────────────────────────────────────
def compute_dice(logits, targets, threshold=0.5):
    """计算 Dice 系数（评估指标）"""
    probs = (torch.sigmoid(logits) > threshold).float()
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    return (2.0 * intersection) / (probs.sum() + targets.sum() + 1e-8)


def denormalize(tensor):
    """将归一化的图像张量还原为可显示的 numpy 数组"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1)


# ── 训练一轮 ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


# ── 验证一轮（训练过程使用，只返回 loss 和 Dice）────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        total_dice += compute_dice(logits, masks).item() * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n


# ── 测试集完整指标（Dice / IoU / Precision / Recall）────────────────
@torch.no_grad()
def compute_test_metrics(model, loader, criterion, device, threshold=0.5):
    """在测试集上计算完整评估指标，与 DuckNet 论文报告的指标一致"""
    model.eval()
    total_loss = 0.0
    # 累积所有样本的像素级预测与标签，用于批量计算指标
    all_preds, all_targets = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > threshold).float()
        all_preds.append(preds.cpu().view(-1))
        all_targets.append(masks.cpu().view(-1))

    p = torch.cat(all_preds).bool()
    t = torch.cat(all_targets).bool()
    tp = (p & t).sum().float()
    fp = (p & ~t).sum().float()
    fn = (~p & t).sum().float()
    tn = (~p & ~t).sum().float()

    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)          # Jaccard
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    loss      = total_loss / len(loader.dataset)

    return {"loss": loss, "dice": dice.item(), "iou": iou.item(),
            "precision": precision.item(), "recall": recall.item(),
            "accuracy": accuracy.item()}


# ── 绘制损失曲线 ──────────────────────────────────────────────────────
def plot_losses(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[已保存] 损失曲线 → {save_path}")


# ── 可视化预测结果 ────────────────────────────────────────────────────
@torch.no_grad()
def visualize_predictions(model, dataset, device, save_dir, n=5):
    model.eval()
    indices = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    col_titles = ["Input Image", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12)

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        logit = model(img.unsqueeze(0).to(device))
        pred = (torch.sigmoid(logit) > 0.5).float().squeeze().cpu().numpy()

        axes[row, 0].imshow(denormalize(img))
        axes[row, 1].imshow(mask.squeeze().numpy(), cmap="gray")
        axes[row, 2].imshow(pred, cmap="gray")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "predictions.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[已保存] 预测可视化 → {save_path}")


# ── 主程序 ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Kvasir-SEG")
    parser.add_argument("--data",       default="data/Kvasir-SEG",   help="数据集根目录")
    parser.add_argument("--save",       default="outputs",            help="输出目录")
    parser.add_argument("--epochs",     type=int,   default=100,      help="最大训练轮数")
    parser.add_argument("--batch",      type=int,   default=8,        help="批大小")
    parser.add_argument("--lr",         type=float, default=1e-3,     help="学习率")
    parser.add_argument("--img_size",   type=int,   default=256,      help="输入图像尺寸")
    parser.add_argument("--test_ratio", type=float, default=0.1,      help="测试集比例（对齐 DuckNet：10%%）")
    parser.add_argument("--val_ratio",  type=float, default=0.1,      help="验证集比例（对齐 DuckNet：10%%）")
    parser.add_argument("--patience",   type=int,   default=10,       help="早停耐心轮数")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.save, exist_ok=True)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"[设备] {device}")

    # 数据集路径
    img_dir  = os.path.join(args.data, "images")
    mask_dir = os.path.join(args.data, "masks")

    # 获取全部图像路径，再用两步 train_test_split 划分（对齐 DuckNet 论文协议）
    # 第一步：分出测试集（10%）；第二步：从剩余中分出验证集（≈10%）
    all_paths = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.jpeg"))
    all_paths = sorted(all_paths)

    # val_frac：从 train+val 中再切多少作验证，使验证集占总数约等于 val_ratio
    val_frac = args.val_ratio / (1.0 - args.test_ratio)
    trainval_paths, test_paths = train_test_split(
        all_paths, test_size=args.test_ratio, shuffle=True, random_state=DUCK_SEED
    )
    train_paths, val_paths = train_test_split(
        trainval_paths, test_size=val_frac, shuffle=True, random_state=DUCK_SEED
    )
    print(f"[数据集] 训练: {len(train_paths)} / 验证: {len(val_paths)} / 测试: {len(test_paths)}")

    train_set = KvasirDataset(img_dir, mask_dir, img_size=args.img_size, img_paths=train_paths)
    val_set   = KvasirDataset(img_dir, mask_dir, img_size=args.img_size, img_paths=val_paths)
    test_set  = KvasirDataset(img_dir, mask_dir, img_size=args.img_size, img_paths=test_paths)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=args.batch, shuffle=False, num_workers=2)

    # 模型、损失函数、优化器
    model     = UNet().to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ReduceLROnPlateau：验证损失连续 3 轮不降则 lr × 0.5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 训练循环
    train_losses, val_losses = [], []
    best_dice    = 0.0
    best_val_loss = float("inf")   # 早停监控验证损失
    patience_counter = 0           # 连续未改善的轮数

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 获取当前学习率（供打印参考）
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"Val Dice: {val_dice:.4f}  "
              f"LR: {current_lr:.2e}")

        # 保存最优模型权重（按 Dice 评估）
        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = os.path.join(args.save, "unet_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → 最优模型已保存 (Dice={best_dice:.4f})")

        # 早停：验证损失连续 patience 轮无改善则终止
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[早停] 验证损失连续 {args.patience} 轮未改善，提前终止训练。")
                break

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.save, "unet_final.pth"))

    # 绘制损失曲线
    plot_losses(train_losses, val_losses, os.path.join(args.save, "loss_curve.png"))

    # 加载最优模型，在测试集上输出完整评估指标
    print("\n[测试集评估] 加载最优模型 unet_best.pth ...")
    model.load_state_dict(torch.load(os.path.join(args.save, "unet_best.pth"), map_location=device))
    metrics = compute_test_metrics(model, test_loader, criterion, device)
    print(f"\n{'='*50}")
    print(f"  Test Results (n={len(test_set)})")
    print(f"  Dice      : {metrics['dice']:.4f}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Loss      : {metrics['loss']:.4f}")
    print(f"{'='*50}")

    # 可视化 5 张预测结果（从测试集中随机抽取）
    visualize_predictions(model, test_set, device, args.save, n=5)

    print(f"\n训练完成！共 {len(train_losses)} 轮，最优 Val Dice: {best_dice:.4f}，Test Dice: {metrics['dice']:.4f}")
    print(f"输出目录: {args.save}/")


if __name__ == "__main__":
    main()

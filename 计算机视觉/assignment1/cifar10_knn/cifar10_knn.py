"""cifar10_knn.py — 编程作业：CIFAR-10 KNN 分类器

任务概述
--------
从头实现 K 近邻（KNN）分类器，完成以下三个版本的 L2 距离矩阵计算函数：

  - compute_distances_two_loops  ：双层循环（最直观）
  - compute_distances_one_loop   ：单层循环 + 广播
  - compute_distances_vectorized ：完全向量化（最快）

每完成一处填空，即可直接运行脚本验证该版本的准确率。
全部完成后，对比三个版本的运行时间与结果一致性。

用法
----
    python cifar10_knn.py
    python cifar10_knn.py --k 3 --seed 0
    python cifar10_knn.py --save my_dir
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# CIFAR-10 类别名称
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 KNN Classifier")
    p.add_argument("--k",    type=int, default=5)          # 近邻数
    p.add_argument("--save", type=str, default="outputs")  # 输出目录
    p.add_argument("--seed", type=int, default=42)         # 随机种子
    return p.parse_args()


def load_cifar10(data_root):
    """下载 CIFAR-10，返回原始数据集对象（不做采样）。"""
    tf = transforms.ToTensor()
    train_set = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=tf)
    test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
    return train_set, test_set


def sample_by_class(dataset, n_per_class, seed):
    """按类均匀采样，每类取 n_per_class 张，返回 (X, y) numpy 数组。

    X: (N, 3072) float32，像素值 [0, 1]
    y: (N,)      int
    """
    rng = np.random.default_rng(seed)
    # 按类别收集下标
    class_indices = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected = []
    for indices in class_indices:
        chosen = rng.choice(indices, size=n_per_class, replace=False)
        selected.extend(chosen.tolist())

    # 提取图像和标签，展平为一维向量
    imgs, labels = [], []
    for idx in selected:
        img, label = dataset[idx]
        imgs.append(img.numpy().reshape(-1))   # (3072,)
        labels.append(label)

    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.int32)


# ── KNN 距离矩阵计算（三个版本）────────────────────────────────────────────

def compute_distances_two_loops(X_train, X_test):
    """【填空 1】双层循环版本：逐一计算每对 (测试, 训练) 样本之间的 L2 距离。

    要求：
      - 外层循环遍历测试样本（下标 i）
      - 内层循环遍历训练样本（下标 j）
      - 计算 X_test[i] 与 X_train[j] 之间的欧氏距离，存入 dists[i, j]

    提示：L2 距离公式：dist = sqrt(sum((a - b)^2))
          np.sqrt、np.dot 均可使用

    参数：
      X_train: (n_train, 3072) 训练集特征
      X_test:  (n_test,  3072) 测试集特征
    返回：
      dists:   (n_test, n_train) float32 距离矩阵
    """
    n_test, n_train = X_test.shape[0], X_train.shape[0]
    dists = np.zeros((n_test, n_train), dtype=np.float32)

    ### 代码开始处 ###
    for i in range(n_test):
        for j in range(n_train):
            dists[i, j] = np.sqrt(np.sum((X_test[i] - X_train[j]) ** 2))
    ### 代码结束处 ###

    return dists


def compute_distances_one_loop(X_train, X_test):
    """【填空 2】单层循环版本：对每个测试样本，利用广播一次计算与所有训练样本的距离。

    要求：
      - 只使用一层循环遍历测试样本（下标 i）
      - 利用 NumPy 广播，计算 X_test[i] 与所有训练样本的差值矩阵
      - 对差值按行求平方和后开方，得到 dists[i]

    提示：X_train - X_test[i] 会触发广播，结果形状为 (n_train, 3072)

    参数/返回：同 compute_distances_two_loops
    """
    n_test  = X_test.shape[0]
    n_train = X_train.shape[0]
    dists = np.zeros((n_test, n_train), dtype=np.float32)

    ### 代码开始处 ###
    for i in range(n_test):
        diff = X_train - X_test[i]
        dists[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    ### 代码结束处 ###

    return dists


def compute_distances_vectorized(X_train, X_test):
    """【填空 3】完全向量化版本：利用平方展开公式，用矩阵乘法一次计算所有距离。

    要求：
      - 不使用任何 Python 循环
      - 利用公式展开：||a - b||² = ||a||² - 2·a·bᵀ + ||b||²
      - 将最终结果赋值给 dists

    提示：
      - sq_test  = (X_test  ** 2).sum(axis=1, keepdims=True)   形状 (n_test,  1)
      - sq_train = (X_train ** 2).sum(axis=1, keepdims=True).T  形状 (1, n_train)
      - cross    = X_test @ X_train.T                           形状 (n_test, n_train)
      - 数值误差可能导致极小负值，需在开方前 clip 到 0

    参数/返回：同 compute_distances_two_loops
    """
    n_test  = X_test.shape[0]
    n_train = X_train.shape[0]
    dists = np.zeros((n_test, n_train), dtype=np.float32)

    ### 代码开始处 ###
    sq_test = (X_test ** 2).sum(axis=1, keepdims=True)
    sq_train = (X_train ** 2).sum(axis=1, keepdims=True).T
    cross = X_test @ X_train.T
    dists = np.sqrt(np.maximum(sq_test + sq_train - 2 * cross, 0))
    ### 代码结束处 ###

    return dists


# ── 预测与评估 ────────────────────────────────────────────────────────────

def predict(dists, y_train, k):
    """根据距离矩阵，用 K 近邻多数投票给出预测标签。

    参数：
      dists:   (n_test, n_train) 距离矩阵
      y_train: (n_train,)        训练标签
      k:       int               近邻数
    返回：
      preds:   (n_test,) int32   预测标签
    """
    # 找出每行距离最小的 k 个训练样本下标，形状 (n_test, k)
    nn_indices = np.argpartition(dists, k, axis=1)[:, :k]
    # 取出 k 个近邻的训练标签，形状 (n_test, k)
    nn_labels = y_train[nn_indices]
    # 多数投票：统计每个类别出现次数，取最多的作为预测结果
    preds = np.array([
        np.bincount(row, minlength=10).argmax()
        for row in nn_labels
    ], dtype=np.int32)
    return preds


def accuracy(preds, y_true):
    return (preds == y_true).mean()


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_timing(names, times, accs, save_path):
    """绘制三个版本的运行时间对比柱状图，准确率标注在柱子上方。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(names, times, color=colors, width=0.5)

    # 在柱子顶部标注准确率
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.01,
                f"acc={acc:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Time (seconds)")
    ax.set_title("KNN Distance Computation: Time Comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 -> {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, args.save)
    os.makedirs(out_dir, exist_ok=True)
    data_root  = os.path.join(script_dir, "..", "data")

    # 下载数据
    print("加载 CIFAR-10 ...")
    train_set, test_set = load_cifar10(data_root)

    # 按类均匀采样
    print("按类采样 ...")
    X_train, y_train = sample_by_class(train_set, n_per_class=500, seed=args.seed)
    X_test,  y_test  = sample_by_class(test_set,  n_per_class=20,  seed=args.seed)
    print(f"训练集: {X_train.shape}  |  测试集: {X_test.shape}")

    # 三个版本的 KNN
    versions = [
        ("Two-loop",   compute_distances_two_loops),
        ("One-loop",   compute_distances_one_loop),
        ("Vectorized", compute_distances_vectorized),
    ]

    times, accs, dist_matrices, done_names = [], [], [], []
    for name, fn in versions:
        print(f"\n[{name}] 计算距离矩阵 ...")
        t0 = time.time()
        dists = fn(X_train, X_test)
        elapsed = time.time() - t0

        # 若函数返回 None 或全零矩阵，说明该填空尚未完成，跳过
        if dists is None or not np.any(dists):
            print(f"  [{name}] 尚未实现，跳过。")
            continue

        preds = predict(dists, y_train, k=args.k)
        acc   = accuracy(preds, y_test)
        times.append(elapsed)
        accs.append(acc)
        dist_matrices.append(dists)
        done_names.append(name)
        print(f"  时间: {elapsed:.2f}s  |  准确率: {acc:.4f}")

    if not done_names:
        print("\n尚未完成任何填空，请先实现至少一个距离计算函数。")
        return

    # 打印汇总表
    print("\n{:<14} {:>10} {:>10}".format("Version", "Time(s)", "Accuracy"))
    print("-" * 36)
    for name, t, acc in zip(done_names, times, accs):
        print(f"{name:<14} {t:>10.2f} {acc:>10.4f}")

    # 若多个版本均已完成，验证结果一致性
    if len(dist_matrices) > 1:
        print("\n距离矩阵一致性验证:")
        for i in range(1, len(dist_matrices)):
            ok = np.allclose(dist_matrices[0], dist_matrices[i], atol=1e-4)
            print(f"  {done_names[0]} ≈ {done_names[i]}: {ok}")

    # 若全部完成，保存柱状图
    if len(done_names) == len(versions):
        plot_timing(done_names, times, accs, os.path.join(out_dir, "knn_timing.png"))

    # 助教核对行：输出各已完成版本的准确率（seed=42, k=5 时参考值约为 0.245）
    print("\n" + "=" * 36)
    for name, acc in zip(done_names, accs):
        print(f"[{name}] accuracy = {acc:.4f}")
    print("=" * 36)


if __name__ == "__main__":
    main()

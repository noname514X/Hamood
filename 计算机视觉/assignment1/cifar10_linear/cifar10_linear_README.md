# 编程作业：CIFAR-10 线性分类器权重可视化

## 作业背景

线性分类器训练完成后，权重矩阵 $W$ 的形状为 $(10, 3072)$，每一行对应一个类别的"模板"。将某一行 reshape 为 $(3, 32, 32)$ 并归一化后，即可将其还原为一张图像，直观反映分类器对该类别外观的"期望"。

本作业要求学生完成权重图像的还原与显示逻辑，理解以下概念：

- NumPy 数组的 reshape 与 min-max 归一化
- matplotlib 显示彩色图像时通道维度的顺序要求（H, W, C）

## 任务要求

在 `cifar10_linear.py` 的 `plot_templates` 函数循环体内完成 **1 处填空**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | `t = W[idx]` | 取第 idx 个类别的权重，形状 `(3, 32, 32)` |
| 2 | min-max 归一化 | 将权重值域映射到 `[0, 1]`，公式见提示 |
| 3 | `ax.imshow` | 转换为 `(H, W, C)` 后显示 |

完成填空后运行脚本，`outputs/weight_templates.png` 中应出现 10 个类别的彩色模板图像。

## 运行方式

```bash
# 使用默认参数（epochs=10，lr=0.01，seed=42）
python cifar10_linear.py

# 自定义参数
python cifar10_linear.py --epochs 5 --lr 0.05 --save my_outputs
```

